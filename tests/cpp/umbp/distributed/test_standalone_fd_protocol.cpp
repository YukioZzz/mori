// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include <dirent.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <string>

#include "umbp/standalone/ipc.h"

namespace mori::umbp::standalone {
namespace {

size_t OpenFdCount() {
  DIR* directory = opendir("/proc/self/fd");
  if (!directory) return 0;
  size_t count = 0;
  while (readdir(directory)) ++count;
  closedir(directory);
  return count;
}

class FdProtocolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets.data()), 0);
    source = open("/dev/null", O_RDONLY | O_CLOEXEC);
    ASSERT_GE(source, 0);
    std::strcpy(message.client_id, "test-client");
    message.worker_base = 4096;
    message.size = 8192;
  }
  void TearDown() override {
    for (int fd : sockets)
      if (fd >= 0) close(fd);
    if (source >= 0) close(source);
  }
  void Send(size_t bytes, size_t descriptors = 1) {
    std::array<int, 2> fds{source, source};
    iovec io{&message, bytes};
    std::array<char, CMSG_SPACE(sizeof(int) * 2)> control{};
    msghdr header{};
    header.msg_iov = &io;
    header.msg_iovlen = 1;
    header.msg_control = control.data();
    header.msg_controllen = CMSG_SPACE(sizeof(int) * descriptors);
    auto* rights = CMSG_FIRSTHDR(&header);
    rights->cmsg_level = SOL_SOCKET;
    rights->cmsg_type = SCM_RIGHTS;
    rights->cmsg_len = CMSG_LEN(sizeof(int) * descriptors);
    std::memcpy(CMSG_DATA(rights), fds.data(), sizeof(int) * descriptors);
    ASSERT_EQ(sendmsg(sockets[0], &header, MSG_NOSIGNAL), static_cast<ssize_t>(bytes));
    ASSERT_EQ(shutdown(sockets[0], SHUT_WR), 0);
  }
  void Reject(size_t bytes, size_t descriptors = 1) {
    const size_t before = OpenFdCount();
    Send(bytes, descriptors);
    FdRegistrationMessage received;
    std::string error;
    EXPECT_EQ(RecvFdRegistration(sockets[1], &received, &error), -1);
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(OpenFdCount(), before) << "received descriptors leaked on rejection";
  }
  std::array<int, 2> sockets{-1, -1};
  int source = -1;
  FdRegistrationMessage message;
};

TEST_F(FdProtocolTest, LegacyHostMessageStillUses88Bytes) {
  static_assert(kFdRegistrationV1Bytes == 88);
  Send(88);
  FdRegistrationMessage received;
  std::string error;
  int fd = RecvFdRegistration(sockets[1], &received, &error);
  ASSERT_GE(fd, 0) << error;
  EXPECT_EQ(received.version, 1);
  EXPECT_EQ(received.device_id, -1);
  EXPECT_EQ(received.worker_base, message.worker_base);
  EXPECT_EQ(received.size, message.size);
  EXPECT_NE(fcntl(fd, F_GETFD) & FD_CLOEXEC, 0);
  close(fd);
}

TEST_F(FdProtocolTest, GpuExtensionPreservesNonzeroDevice) {
  message.version = kGpuFdRegistrationVersion;
  message.device_id = 7;
  Send(sizeof(message));
  FdRegistrationMessage received;
  std::string error;
  int fd = RecvFdRegistration(sockets[1], &received, &error);
  ASSERT_GE(fd, 0) << error;
  EXPECT_EQ(received.device_id, 7);
  EXPECT_EQ(received.version, kGpuFdRegistrationVersion);
  close(fd);
}

TEST_F(FdProtocolTest, TruncatedGpuExtensionClosesFd) {
  message.version = kGpuFdRegistrationVersion;
  message.device_id = 0;
  Reject(kFdRegistrationV1Bytes + 2);
}
TEST_F(FdProtocolTest, NegativeGpuDeviceClosesFd) {
  message.version = kGpuFdRegistrationVersion;
  Reject(sizeof(message));
}
TEST_F(FdProtocolTest, UnknownVersionClosesFd) {
  message.version = 99;
  Reject(kFdRegistrationV1Bytes);
}
TEST_F(FdProtocolTest, ReservedExtensionClosesFd) {
  message.version = kGpuFdRegistrationVersion;
  message.device_id = 0;
  message.reserved = 1;
  Reject(sizeof(message));
}
TEST_F(FdProtocolTest, UnexpectedExtraDescriptorClosesBoth) { Reject(kFdRegistrationV1Bytes, 2); }
TEST_F(FdProtocolTest, TruncatedHeaderClosesFd) { Reject(16); }
}  // namespace
}  // namespace mori::umbp::standalone
