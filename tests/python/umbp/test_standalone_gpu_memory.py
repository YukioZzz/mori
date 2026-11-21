# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Integration checks against an isolated distributed DRAM standalone server.

Set UMBP_TEST_STANDALONE_ADDRESS to its local Unix socket. The tests require
GPUs and, for GEM, a NIC supported by that allocator. They never clear a pool.
"""

import os
import time
import uuid

import pytest
import torch

from mori.cpp import (
    MemoryLocationType,
    UMBPClient,
    UMBPConfig,
    UMBPStandaloneProcessConfig,
)
from mori.io import hip_gem_lookup, make_hip_gem_mem_pool

ADDRESS = os.environ.get("UMBP_TEST_STANDALONE_ADDRESS")
pytestmark = pytest.mark.skipif(
    not ADDRESS or not torch.cuda.is_available(),
    reason="requires an isolated standalone GPU server",
)


def client():
    config = UMBPConfig()
    standalone = UMBPStandaloneProcessConfig()
    standalone.address = ADDRESS
    standalone.auto_start = False
    standalone.startup_timeout_ms = 10000
    config.standalone_process = standalone
    return UMBPClient(config)


@pytest.mark.parametrize("device", range(torch.cuda.device_count()))
@pytest.mark.parametrize("allocator", ["torch", "gem"])
def test_registered_views_roundtrip_and_reregister(device, allocator):
    torch.cuda.set_device(device)
    pool = make_hip_gem_mem_pool(device) if allocator == "gem" else None

    def empty():
        if pool is None:
            return torch.empty(8 << 20, dtype=torch.uint8, device=f"cuda:{device}")
        with torch.cuda.use_mem_pool(pool, device=device):
            return torch.empty(8 << 20, dtype=torch.uint8, device=f"cuda:{device}")

    source, target = empty(), empty()
    views = [(64 << 10, 1 << 20), (5 << 20, 2 << 20)]
    c = client()
    try:
        for cycle in range(2):
            source.fill_(91 + cycle)
            target.zero_()
            torch.cuda.synchronize(device)
            for tensor in (source, target):
                for offset, size in views:
                    assert c.register_memory(
                        tensor.data_ptr() + offset, size, MemoryLocationType.GPU, device
                    )
                # A second layer may register an already imported allocation/view.
                assert c.register_memory(
                    tensor.data_ptr() + views[0][0],
                    views[0][1],
                    MemoryLocationType.GPU,
                    device,
                )
            keys = [f"standalone-memory-{uuid.uuid4()}" for _ in views]
            sizes = [size for _, size in views]
            pointers = [[source.data_ptr() + offset] for offset, _ in views]
            assert list(
                c.batch_put_ranges_from_ptr(
                    keys, sizes, pointers, [[s] for s in sizes], [[0], [0]]
                )
            ) == [True, True]
            deadline = time.monotonic() + 30
            while not all(c.batch_exists(keys)):
                assert time.monotonic() < deadline, "stored keys not visible"
                time.sleep(0.05)
            destinations = [[target.data_ptr() + offset] for offset, _ in views]
            assert list(
                c.batch_get_ranges_into_ptr(
                    keys, destinations, [[s] for s in sizes], [[0], [0]]
                )
            ) == [True, True]
            torch.cuda.synchronize(device)
            expected = torch.zeros_like(target)
            for offset, size in views:
                expected[offset : offset + size].fill_(91 + cycle)
            assert torch.equal(target, expected), "data or untouched-region mismatch"
            # Standalone deregistration tears down all of this client's regions.
            c.deregister_memory(source.data_ptr() + views[0][0])
    finally:
        c.deregister_memory(source.data_ptr() + views[0][0])


def test_gem_registration_rejects_allocation_overrun():
    torch.cuda.set_device(0)
    pool = make_hip_gem_mem_pool(0)
    with torch.cuda.use_mem_pool(pool, device=0):
        tensor = torch.empty(4 << 20, dtype=torch.uint8, device="cuda:0")
    base, size, device = hip_gem_lookup(tensor.data_ptr())
    c = client()
    assert not c.register_memory(
        tensor.data_ptr(),
        base + size - tensor.data_ptr() + 1,
        MemoryLocationType.GPU,
        device,
    )
    assert c.register_memory(
        tensor.data_ptr(), tensor.nbytes, MemoryLocationType.GPU, device
    )
    c.deregister_memory(tensor.data_ptr())
