// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <errno.h>
#include <sched.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <zmq.hpp>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "infiniband/verbs.h"
#include "internal/verbs_attribute.h"
#include "public/flags.h"
#include "public/introspection.h"
#include "public/rdma_memblock.h"
#include "public/status_matchers.h"
#include "public/verbs_helper_suite.h"
#include "public/verbs_util.h"
#include "unit/loopback_fixture.h"

using namespace std::chrono_literals;

namespace rdma_unit_test {
namespace {

// TODO(author1): Add tests stressing SGEs.
// TODO(author1): Send with insufficient recv buffering (RC and UD).
// TODO(author2): Add QP error state check for relevant testcases.

using ::testing::AnyOf;
using ::testing::Each;
using ::testing::Ne;
using ::testing::NotNull;

class Peer2PeerRcQpTest : public LoopbackFixture {
 private:
  zmq::context_t comm_context{1};
  zmq::socket_t comm_socket{};

 public:
  static constexpr char kLocalBufferContent = 'a';
  static constexpr char kRemoteBufferContent = 'b';
  static constexpr int kLargePayloadPages = 128;
  static constexpr int kPages = 1;

  void SetUp() override {
    LoopbackFixture::SetUp();
    if (!Introspection().SupportsRcQp()) {
      GTEST_SKIP() << "Nic does not support RC QP";
    }
    if (verbs_util::peer_mode()) {
      auto peer_port = absl::GetFlag(FLAGS_peer_port);
      auto server_ip = absl::GetFlag(FLAGS_server_ip);
      if (verbs_util::is_client()) {
        comm_socket = {comm_context, zmq::socket_type::req};
        comm_socket.connect("tcp://" + server_ip + ":" +
                            std::to_string(peer_port));
      } else {
        comm_socket = {comm_context, zmq::socket_type::rep};
        comm_socket.bind("tcp://*:" + std::to_string(peer_port));
      }
    }
  }

 protected:
  void synchronise() {
    if (!verbs_util::peer_mode()) {
      return;
    }
    static int sync_count = 0;
    std::string msg = "sync" + std::to_string(sync_count);
    zmq::message_t reply{};
    if (verbs_util::is_client()) {
      comm_socket.send(zmq::buffer(msg), zmq::send_flags::none);
      if (!comm_socket.recv(reply, zmq::recv_flags::none) ||
          (reply.to_string() != msg)) {
        LOG(FATAL) << "Wrong Sync message received '" << reply.to_string()
                   << "' expected '" << msg << "'";
      }
    } else {
      if (!comm_socket.recv(reply, zmq::recv_flags::none) ||
          (reply.to_string() != msg)) {
        LOG(FATAL) << "Wrong Sync message received '" << reply.to_string()
                   << "' expected '" << msg << "'";
      }
      comm_socket.send(zmq::buffer(msg), zmq::send_flags::none);
    }
    sync_count++;
  }

  absl::StatusOr<std::pair<Client, Client>> CreateConnectedClientsPair(
      int pages = kPages, QpInitAttribute qp_init_attr = QpInitAttribute(),
      QpAttribute qp_attr = QpAttribute()) {
    struct verbs_util::conn_attr local_host {};
    struct verbs_util::conn_attr remote_host {};
    Client local{};
    Client remote{};

    if (!verbs_util::peer_mode() || verbs_util::is_client()) {
      ASSIGN_OR_RETURN(local, CreateClient(IBV_QPT_RC, pages, qp_init_attr));
      std::fill_n(local.buffer.data(), local.buffer.size(),
                  kLocalBufferContent);
    }
    if (!verbs_util::peer_mode() || verbs_util::is_server()) {
      ASSIGN_OR_RETURN(remote, CreateClient(IBV_QPT_RC, pages, qp_init_attr));
      std::fill_n(remote.buffer.data(), remote.buffer.size(),
                  kRemoteBufferContent);
    }

    // Execute Tests in Loopback mode
    if (!verbs_util::peer_mode()) {
      RETURN_IF_ERROR(ibv_.ModifyRcQpResetToRts(local.qp, local.port_attr,
                                                remote.port_attr.gid,
                                                remote.qp->qp_num, qp_attr));
      RETURN_IF_ERROR(ibv_.ModifyRcQpResetToRts(remote.qp, remote.port_attr,
                                                local.port_attr.gid,
                                                local.qp->qp_num, qp_attr));
      local.remote.addr = remote.buffer.data();
      local.remote.size = remote.buffer.size();
      local.remote.rkey = remote.mr->rkey;
      remote.remote.addr = local.buffer.data();
      remote.remote.size = local.buffer.size();
      remote.remote.rkey = local.mr->rkey;
    } else if (verbs_util::is_client()) {
      local_host.psn = lrand48() & 0xffffff;
      local_host.gid = local.port_attr.gid;
      local_host.lid = local.port_attr.attr.lid;
      local_host.qpn = local.qp->qp_num;
      local_host.port = local.port_attr.port;
      local_host.remote_addr = local.buffer.data();
      local_host.remote_buf_size = local.buffer.size();
      local_host.rkey = local.mr->rkey;
      verbs_util::RunPeerClient(local_host, remote_host, comm_socket);
      RETURN_IF_ERROR(ibv_.ModifyRcQpResetToRts(
          local.qp, local.port_attr, remote_host.gid, remote_host.qpn,
          remote_host.port, qp_attr));
      local.remote.addr = remote_host.remote_addr;
      local.remote.size = remote_host.remote_buf_size;
      local.remote.rkey = remote_host.rkey;
    } else {
      local_host.psn = lrand48() & 0xffffff;
      local_host.gid = remote.port_attr.gid;
      local_host.lid = remote.port_attr.attr.lid;
      local_host.qpn = remote.qp->qp_num;
      local_host.port = remote.port_attr.port;
      local_host.remote_addr = remote.buffer.data();
      local_host.remote_buf_size = remote.buffer.size();
      local_host.rkey = remote.mr->rkey;
      verbs_util::RunPeerServer(local_host, remote_host, comm_socket);
      RETURN_IF_ERROR(ibv_.ModifyRcQpResetToRts(
          remote.qp, remote.port_attr, remote_host.gid, remote_host.qpn,
          remote_host.port, qp_attr));
      remote.remote.addr = remote_host.remote_addr;
      remote.remote.size = remote_host.remote_buf_size;
      remote.remote.rkey = remote_host.rkey;
    }

    return std::make_pair(local, remote);
  }

  // Returns a valid inline size and a invalid inline size based on trial
  // and error creating qp's.
  absl::StatusOr<std::pair<uint32_t, uint32_t>> DetermineInlineLimits() {
    QpInitAttribute qp_init_attr;
    static constexpr std::array kInlineTestSize{64,   128,  256,  512,
                                                1024, 4096, 16536};
    qp_init_attr.set_max_inline_data(kInlineTestSize[0]);
    ASSIGN_OR_RETURN(Client local,
                     CreateClient(IBV_QPT_RC, kPages, qp_init_attr));
    for (int idx = 1; idx < static_cast<int>(kInlineTestSize.size()); ++idx) {
      ibv_qp_init_attr init_attr =
          qp_init_attr.set_max_inline_data(kInlineTestSize[idx])
              .GetAttribute(local.cq, local.cq, IBV_QPT_RC);
      ibv_qp* qp = ibv_.CreateQp(local.pd, init_attr);
      if (qp == nullptr) {
        return std::make_pair(kInlineTestSize[idx - 1], kInlineTestSize[idx]);
      }
    }
    return absl::InvalidArgumentError("unable to find inline size limit.");
  }
};

TEST_F(Peer2PeerRcQpTest, Send) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge sge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &sge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    verbs_util::PostSend(local.qp, send);
  }
  synchronise();

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.byte_len, local.buffer.size());
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    EXPECT_EQ(completion.wc_flags, 0);
    EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
  }
  synchronise();
}

TEST_F(Peer2PeerRcQpTest, SendEmptySgl) {
  if (!Introspection().AllowsEmptySgl()) {
    GTEST_SKIP() << "Device does not allow empty SGL.";
  }
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, nullptr, /*num_sge=*/0);
    verbs_util::PostRecv(remote.qp, recv);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, nullptr, /*num_sge=*/0);
    verbs_util::PostSend(local.qp, send);
  }
  synchronise();

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.byte_len, 0);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    EXPECT_EQ(completion.wc_flags, 0);
  }
  synchronise();
}

TEST_F(Peer2PeerRcQpTest, UnsignaledSend) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    send.send_flags = send.send_flags & ~IBV_SEND_SIGNALED;
    verbs_util::PostSend(local.qp, send);
  }
  synchronise();

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.byte_len, local.buffer.size());
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    EXPECT_EQ(completion.wc_flags, 0);
    EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_wc completion;
    EXPECT_EQ(ibv_poll_cq(local.cq, 1, &completion), 0);
  }
  synchronise();
}

// Send a 64MB chunk from local to remote
TEST_F(Peer2PeerRcQpTest, SendLargeChunk) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote),
                       CreateConnectedClientsPair(kLargePayloadPages));

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_SEND);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RECV);
  EXPECT_EQ(completion.byte_len, local.buffer.size());
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
  EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
}

TEST_F(Peer2PeerRcQpTest, SendInlineData) {
  uint32_t valid_inline_size, invalid_inline_size;
  ASSERT_OK_AND_ASSIGN(std::tie(valid_inline_size, invalid_inline_size),
                       DetermineInlineLimits());
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(
      std::tie(local, remote),
      CreateConnectedClientsPair(
          kPages, QpInitAttribute().set_max_inline_data(valid_inline_size)));
  ASSERT_GE(remote.buffer.size(), valid_inline_size)
      << "receiver buffer too small";

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  // a vector which is not registered to pd or mr
  auto data_src = std::make_unique<std::vector<uint8_t>>(valid_inline_size);
  std::fill(data_src->begin(), data_src->end(), 'c');
  ibv_sge lsge{
      .addr = reinterpret_cast<uint64_t>(data_src->data()),
      .length = valid_inline_size,
      .lkey = 0xDEADBEEF,  // random bad keys
  };
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  send.send_flags |= IBV_SEND_INLINE;
  verbs_util::PostSend(local.qp, send);
  (*data_src)[0] = kLocalBufferContent;  // source can be modified immediately
  data_src.reset();  // delete the source buffer immediately after post_send()

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_SEND);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RECV);
  EXPECT_EQ(completion.byte_len, valid_inline_size);
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
  EXPECT_THAT(absl::MakeSpan(remote.buffer.data(), valid_inline_size),
              Each('c'));
  EXPECT_THAT(absl::MakeSpan(remote.buffer.data() + valid_inline_size,
                             remote.buffer.data() + remote.buffer.size()),
              Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, SendExceedMaxInlineData) {
  uint32_t valid_inline_size, invalid_inline_size;
  ASSERT_OK_AND_ASSIGN(std::tie(valid_inline_size, invalid_inline_size),
                       DetermineInlineLimits());
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(
      std::tie(local, remote),
      CreateConnectedClientsPair(
          kPages, QpInitAttribute().set_max_inline_data(valid_inline_size)));
  ASSERT_GE(remote.buffer.size(), valid_inline_size)
      << "receiver buffer too small";
  const uint32_t actual_max_inline_data =
      verbs_util::GetQpCap(local.qp).max_inline_data;
  // a vector which is not registered to pd or mr
  auto data_src =
      std::make_unique<std::vector<uint8_t>>(actual_max_inline_data + 10);
  std::fill(data_src->begin(), data_src->end(), 'c');

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge{
      .addr = reinterpret_cast<uint64_t>(data_src->data()),
      .length = static_cast<uint32_t>(data_src->size()),
      .lkey = 0xDEADBEEF,  // random bad keys
  };

  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  send.send_flags |= IBV_SEND_INLINE;
  ibv_send_wr* bad_wr;
  EXPECT_THAT(ibv_post_send(local.qp, &send, &bad_wr),
              AnyOf(EPERM, ENOMEM, EINVAL));
  EXPECT_TRUE(verbs_util::ExpectNoCompletion(local.cq));
}

TEST_F(Peer2PeerRcQpTest, SendInlineDataInvalidOp) {
  constexpr uint32_t kProposedMaxInlineData = 64;
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(
      std::tie(local, remote),
      CreateConnectedClientsPair(kPages, QpInitAttribute().set_max_inline_data(
                                             kProposedMaxInlineData)));
  ASSERT_GE(remote.buffer.size(), kProposedMaxInlineData)
      << "receiver buffer too small";
  ibv_qp_attr attr;
  ibv_qp_init_attr init_attr;
  ASSERT_EQ(ibv_query_qp(local.qp, &attr, IBV_QP_CAP, &init_attr), 0);
  // a vector which is not registered to pd or mr
  auto data_src =
      std::make_unique<std::vector<uint8_t>>(kProposedMaxInlineData);
  std::fill(data_src->begin(), data_src->end(), 'c');

  ibv_sge lsge{
      .addr = reinterpret_cast<uint64_t>(data_src->data()),
      .length = init_attr.cap.max_inline_data,
      .lkey = 0xDEADBEEF,  // random bad keys
  };

  // Inline data is only for send and RDMA write. Post a read here.
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &lsge, /*num_sge=*/1, local.remote.addr, local.remote.rkey);
  read.send_flags |= IBV_SEND_INLINE;
  ibv_send_wr* bad_wr;
  // Undefined behavior according to spec.
  int result = ibv_post_send(local.qp, &read, &bad_wr);
  if (result == 0) {
    (*data_src)[0] = kLocalBufferContent;  // source can be modified immediately
    data_src.reset();  // delete the source buffer immediately after post_send()

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_THAT(completion.status,
                AnyOf(IBV_WC_LOC_QP_OP_ERR, IBV_WC_LOC_PROT_ERR));
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  } else {
    EXPECT_EQ(result, EINVAL);
  }
}

TEST_F(Peer2PeerRcQpTest, SendImmData) {
  // The immediate data should be in network byte order according to the type
  // But we are just lazy here and assume that the two sides have the same
  // endianness.
  const uint32_t kImm = 0xBADDCAFE;
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    send.opcode = IBV_WR_SEND_WITH_IMM;
    send.imm_data = kImm;
    verbs_util::PostSend(local.qp, send);
  }
  synchronise();

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.byte_len, remote.buffer.size());
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    EXPECT_NE(completion.wc_flags & IBV_WC_WITH_IMM, 0);
    EXPECT_EQ(kImm, completion.imm_data);

    EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
  }
  synchronise();
}

TEST_F(Peer2PeerRcQpTest, SendRecvOnlyImmData) {
  const uint32_t kImm = 0xBADDCAFE;
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_recv_wr recv;
  ibv_send_wr send;
  ibv_wc completion;

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    recv = verbs_util::CreateRecvWr(/*wr_id=*/0, nullptr, /*num_sge=*/0);
    verbs_util::PostRecv(remote.qp, recv);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    send = verbs_util::CreateSendWr(/*wr_id=*/1, nullptr, /*num_sge=*/0);
    send.opcode = IBV_WR_SEND_WITH_IMM;
    send.imm_data = kImm;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    EXPECT_NE(completion.wc_flags & IBV_WC_WITH_IMM, 0);
    EXPECT_EQ(kImm, completion.imm_data);
  }
  synchronise();

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    /* Modify the immediate value and send back */
    recv = verbs_util::CreateRecvWr(/*wr_id=*/0, nullptr, /*num_sge=*/0);
    verbs_util::PostRecv(local.qp, recv);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    send = verbs_util::CreateSendWr(/*wr_id=*/1, nullptr, /*num_sge=*/0);
    send.opcode = IBV_WR_SEND_WITH_IMM;
    send.imm_data = completion.imm_data * 2;
    verbs_util::PostSend(remote.qp, send);

    ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    EXPECT_NE(completion.wc_flags & IBV_WC_WITH_IMM, 0);
    EXPECT_EQ(kImm * 2, completion.imm_data);
  }
  synchronise();
}

TEST_F(Peer2PeerRcQpTest, SendWithInvalidate) {
  if (!Introspection().SupportsType2()) {
    GTEST_SKIP() << "Needs type 2 MW and SendWithInvalidate.";
  }
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  // Bind a Type 2 MW on remote.
  static constexpr uint32_t rkey = 0xBA;
  ibv_mw* mw = ibv_.AllocMw(remote.pd, IBV_MW_TYPE_2);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType2MwBind(
                  remote.qp, mw, remote.buffer.span(), rkey, remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);
  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  send.opcode = IBV_WR_SEND_WITH_INV;
  send.invalidate_rkey = mw->rkey;
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_SEND);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RECV);
  EXPECT_EQ(completion.byte_len, remote.buffer.size());
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
  EXPECT_EQ(completion.wc_flags, IBV_WC_WITH_INV);
  EXPECT_EQ(mw->rkey, completion.invalidated_rkey);

  EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));

  // Check that rkey is invalid.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(/*wr_id=*/1, &sge, /*num_sge=*/1,
                                              remote.buffer.data(), mw->rkey);
  read.wr_id = 2;
  verbs_util::PostSend(local.qp, read);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_ACCESS_ERR;
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, expected);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 2);
}

TEST_F(Peer2PeerRcQpTest, SendWithInvalidateEmptySgl) {
  if (!Introspection().SupportsType2()) {
    GTEST_SKIP() << "Needs type 2 MW and SendWithInvalidate.";
  }
  if (!Introspection().AllowsEmptySgl()) {
    GTEST_SKIP() << "NIC does not allow empty SGL.";
  }
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  // Bind a Type 2 MW on remote.
  static constexpr uint32_t rkey = 0xBA;
  ibv_mw* mw = ibv_.AllocMw(remote.pd, IBV_MW_TYPE_2);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType2MwBind(
                  remote.qp, mw, remote.buffer.span(), rkey, remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));

  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, nullptr, /*num_sge=*/0);
  verbs_util::PostRecv(remote.qp, recv);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, nullptr, /*num_sge=*/0);
  send.opcode = IBV_WR_SEND_WITH_INV;
  send.invalidate_rkey = mw->rkey;
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_SEND);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RECV);
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
  EXPECT_EQ(completion.wc_flags, IBV_WC_WITH_INV);
  EXPECT_EQ(mw->rkey, completion.invalidated_rkey);

  // Check that rkey is invalid.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(/*wr_id=*/1, &sge, /*num_sge=*/1,
                                              remote.buffer.data(), mw->rkey);
  read.wr_id = 2;
  verbs_util::PostSend(local.qp, read);
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_REM_ACCESS_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 2);
}

TEST_F(Peer2PeerRcQpTest, SendWithInvalidateNoBuffer) {
  if (!Introspection().SupportsType2()) {
    GTEST_SKIP() << "Needs type 2 MW and SendWithInvalidate.";
  }
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  // Bind a Type 2 MW on remote.
  static constexpr uint32_t rkey = 0xBA;
  ibv_mw* mw = ibv_.AllocMw(remote.pd, IBV_MW_TYPE_2);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType2MwBind(
                  remote.qp, mw, remote.buffer.span(), rkey, remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  send.opcode = IBV_WR_SEND_WITH_INV;
  send.invalidate_rkey = mw->rkey;
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_RNR_RETRY_EXC_ERR;
  EXPECT_EQ(completion.status, expected);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

TEST_F(Peer2PeerRcQpTest, SendWithInvalidateBadRkey) {
  if (!Introspection().SupportsType2()) {
    GTEST_SKIP() << "Needs type 2 MW and SendWithInvalidate.";
  }
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  // Bind a Type 2 MW on remote.
  static constexpr uint32_t rkey = 0xBA;
  ibv_mw* mw = ibv_.AllocMw(remote.pd, IBV_MW_TYPE_2);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType2MwBind(
                  remote.qp, mw, remote.buffer.span(), rkey, remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  send.opcode = IBV_WR_SEND_WITH_INV;
  // Invalid rkey.
  send.invalidate_rkey = (mw->rkey + 10) * 5;
  verbs_util::PostSend(local.qp, send);

  // When a send with invalidate failed, the responder should not send out a
  // NAK code, see IB spec 9.9.6.3 responder error behavior (class J error). The
  // requester should not send out a completion in this case.
  EXPECT_TRUE(verbs_util::ExpectNoCompletion(local.cq));
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(remote.cq));
  // General memory management error undefined in ibverbs.
  EXPECT_THAT(completion.status, Ne(IBV_WC_SUCCESS));
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, recv.wr_id);
}

TEST_F(Peer2PeerRcQpTest, SendWithInvalidateType1Rkey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(
      std::tie(local, remote),
      CreateConnectedClientsPair(kPages, QpInitAttribute(),
                                 QpAttribute().set_timeout(absl::Seconds(1))));
  // Bind a Type 1 MW on remote.
  ibv_mw* mw = ibv_.AllocMw(remote.pd, IBV_MW_TYPE_1);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType1MwBind(remote.qp, mw,
                                             remote.buffer.span(), remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  send.opcode = IBV_WR_SEND_WITH_INV;
  // Type1 rkey.
  send.invalidate_rkey = mw->rkey;
  verbs_util::PostSend(local.qp, send);

  // When a send with invalidate failed, the responder should not send out a
  // NAK code, see IB spec 9.9.6.3 responder error behavior (class J error). The
  // requester should not send out a completion in this case.
  EXPECT_TRUE(verbs_util::ExpectNoCompletion(local.cq));
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(remote.cq));
  // Memory management error (IB Spec 11.6.2) undefined for ibverbs.
  EXPECT_THAT(completion.status, Ne(IBV_WC_SUCCESS));
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
}

// Send with Invalidate targeting another QPs MW.
TEST_F(Peer2PeerRcQpTest, SendWithInvalidateWrongQp) {
  if (!Introspection().SupportsType2()) {
    GTEST_SKIP() << "Needs type 2 MW and SendWithInvalidate.";
  }
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  // Bind a Type 2 MW on local qp.
  static constexpr uint32_t rkey = 0xBA;
  ibv_mw* mw = ibv_.AllocMw(local.pd, IBV_MW_TYPE_2);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType2MwBind(local.qp, mw, local.buffer.span(),
                                             rkey, local.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  send.opcode = IBV_WR_SEND_WITH_INV;
  send.invalidate_rkey = mw->rkey;
  verbs_util::PostSend(local.qp, send);

  // When a send with invalidate failed, the responder should not send out a
  // NAK code, see IB spec 9.9.6.3 responder error behavior (class J error). The
  // requester should not send out a completion in this case.
  EXPECT_TRUE(verbs_util::ExpectNoCompletion(local.cq));
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(remote.cq));
  // General memory management error undefined in ibverbs.
  EXPECT_THAT(completion.status, Ne(IBV_WC_SUCCESS));
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, recv.wr_id);
}

TEST_F(Peer2PeerRcQpTest, SendWithTooSmallRecv) {
  // Recv buffer is to small to fit the whole buffer.
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  const uint32_t kRecvLength = local.buffer.span().size() - 1;
  sge.length = kRecvLength;
  ibv_recv_wr recv = verbs_util::CreateRecvWr(/*wr_id=*/0, &sge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(local.qp, send);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_REM_INV_REQ_ERR);
  EXPECT_EQ(local.qp->qp_num, completion.qp_num);
  EXPECT_EQ(completion.wr_id, 1);

  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_LEN_ERR);
  EXPECT_EQ(remote.qp->qp_num, completion.qp_num);
  EXPECT_EQ(completion.wr_id, 0);

  EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_ERR);
}

TEST_F(Peer2PeerRcQpTest, SendRnr) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_RNR_RETRY_EXC_ERR;
  EXPECT_EQ(completion.status, expected);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

TEST_F(Peer2PeerRcQpTest, SendRnrInfiniteRetries) {
  // 7 is the magic number for infinite retries. Set before modifying QP.
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(
      std::tie(local, remote),
      CreateConnectedClientsPair(kPages, QpInitAttribute(),
                                 QpAttribute().set_rnr_retry(7)));

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(local.qp, send);
  if (Introspection().GeneratesRetryExcOnConnTimeout()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_RETRY_EXC_ERR);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  } else {
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(local.cq));
  }
}

TEST_F(Peer2PeerRcQpTest, BadSendAddr) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    --sge.addr;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &sge, /*num_sge=*/1);
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
    EXPECT_EQ(completion.wr_id, 1);

    /* Ensure local Qp has moved to error state */
    EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));
  }
  synchronise();
}

TEST_F(Peer2PeerRcQpTest, BadRecvAddr) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  --rsge.addr;
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);
  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(local.qp, send);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_REM_OP_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
}

TEST_F(Peer2PeerRcQpTest, RecvOnDeregisteredRegion) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);
  // Deregister the MR before send is posted.
  ASSERT_EQ(ibv_.DeregMr(remote.mr), 0);
  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  // The op code isn't set by the hardware in the error case.
  EXPECT_EQ(completion.status, IBV_WC_REM_OP_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
  EXPECT_THAT(remote.buffer.span(), Each(kRemoteBufferContent));

  EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_ERR);
}

TEST_F(Peer2PeerRcQpTest, RecvPayloadExceedMr) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_mr* shorter_recv_mr = ibv_.RegMr(
      remote.pd, remote.buffer.subblock(0, remote.buffer.size() - 32));
  ASSERT_THAT(shorter_recv_mr, NotNull());
  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), shorter_recv_mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);
  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(local.qp, send);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_REM_OP_ERR);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion2,
                       verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion2.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion2.wr_id, 0);
  EXPECT_EQ(completion2.status, IBV_WC_LOC_PROT_ERR);
}

TEST_F(Peer2PeerRcQpTest, SendBufferExceedMr) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length += 32;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    verbs_util::PostSend(local.qp, send);
  }
  synchronise();

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
    EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);

    /* Ensure local Qp has moved to error state */
    EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));
  }
  synchronise();
}

TEST_F(Peer2PeerRcQpTest, RecvBufferExceedMr) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_mr* recv_mr =
      ibv_.RegMr(remote.pd, remote.buffer.subblock(0, remote.buffer.size()));
  ASSERT_THAT(recv_mr, NotNull());
  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), recv_mr);
  rsge.length += 32;
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);
  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(local.qp, send);

  // Behavior for recv SGL longer than the MR (but the payload is not) is not
  // *strictly* defined by the IBTA spec.
  // On Mellanox NICs, we expect the recv WR result in a success.
  // On MEV, we expect the recv WR result in IBV_WC_LOC_PROT_ERR, and the
  // send fail with IBV_WC_REM_OP_ERR
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(completion.status, AnyOf(IBV_WC_SUCCESS, IBV_WC_REM_OP_ERR));

  ASSERT_OK_AND_ASSIGN(ibv_wc completion2,
                       verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion2.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion2.wr_id, 0);
  EXPECT_THAT(completion2.status, AnyOf(IBV_WC_SUCCESS, IBV_WC_LOC_PROT_ERR));
}

TEST_F(Peer2PeerRcQpTest, BadRecvLkey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  // Change lkey to be invalid.
  rsge.lkey = (rsge.lkey + 10) * 5;
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);
  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(local.qp, send);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_REM_OP_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
}

TEST_F(Peer2PeerRcQpTest, SendInvalidLkey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  // Change lkey to be invalid.
  sge.lkey = (sge.lkey + 10) * 5;
  ibv_send_wr send = verbs_util::CreateSendWr(/*wr_id=*/1, &sge, /*num_sge=*/1);
  verbs_util::PostSend(local.qp, send);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  // Existing hardware does not set opcode on error.
  EXPECT_EQ(completion.wr_id, 1);
}

TEST_F(Peer2PeerRcQpTest, UnsignaledSendError) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  // Change lkey to be invalid.
  sge.lkey = (sge.lkey + 10) * 5;
  ibv_send_wr send = verbs_util::CreateSendWr(/*wr_id=*/1, &sge, /*num_sge=*/1);
  send.send_flags = send.send_flags & ~IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, send);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
}

TEST_F(Peer2PeerRcQpTest, BasicRead) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(local.buffer.span(), Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, BasicReadLargePayload) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote),
                       CreateConnectedClientsPair(kLargePayloadPages));
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(local.buffer.span(), Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, UnsignaledRead) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  read.send_flags = read.send_flags & ~IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, read);
  read.send_flags = read.send_flags | IBV_SEND_SIGNALED;
  read.wr_id = 2;
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 2);
}

TEST_F(Peer2PeerRcQpTest, QpSigAll) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(
      std::tie(local, remote),
      CreateConnectedClientsPair(kPages, QpInitAttribute().set_sq_sig_all(1)));

  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  read.send_flags = read.send_flags & ~IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
}

TEST_F(Peer2PeerRcQpTest, Type1MWRead) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  // Bind a Type 1 MW on remote.
  ibv_mw* mw = ibv_.AllocMw(remote.pd, IBV_MW_TYPE_1);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType1MwBind(remote.qp, mw,
                                             remote.buffer.span(), remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(/*wr_id=*/1, &sge, /*num_sge=*/1,
                                              remote.buffer.data(), mw->rkey);
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(local.buffer.span(), Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, Type2MWRead) {
  if (!Introspection().SupportsType2()) {
    GTEST_SKIP() << "Needs type 2 MW.";
  }
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  // Bind a Type 2 MW on remote.
  static constexpr uint32_t rkey = 0xBA;
  ibv_mw* mw = ibv_.AllocMw(remote.pd, IBV_MW_TYPE_2);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType2MwBind(
                  remote.qp, mw, remote.buffer.span(), rkey, remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(/*wr_id=*/1, &sge, /*num_sge=*/1,
                                              remote.buffer.data(), mw->rkey);
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(local.buffer.span(), Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, Type1MWUnbind) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  // Bind a Type 1 MW on remote.
  ibv_mw* mw = ibv_.AllocMw(remote.pd, IBV_MW_TYPE_1);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType1MwBind(remote.qp, mw,
                                             remote.buffer.span(), remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));
  // Rebind to get a new rkey.
  uint32_t original_rkey = mw->rkey;
  ASSERT_THAT(verbs_util::ExecuteType1MwBind(remote.qp, mw,
                                             remote.buffer.span(), remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));
  EXPECT_NE(original_rkey, mw->rkey);

  // Issue a read with the new rkey.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(/*wr_id=*/1, &sge, /*num_sge=*/1,
                                              remote.buffer.data(), mw->rkey);
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(local.buffer.span(), Each(kRemoteBufferContent));

  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_ACCESS_ERR;
  // Issue a read with the old rkey.
  read.wr.rdma.rkey = original_rkey;
  verbs_util::PostSend(local.qp, read);
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.status, expected);
}

TEST_F(Peer2PeerRcQpTest, ReadInvalidLkey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  // Change lkey to be invalid.
  sge.lkey = (sge.lkey + 10) * 5;
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  // Existing hardware does not set this on error.
  // EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  // Verify buffer is unchanged.
  EXPECT_THAT(local.buffer.span(), Each(kLocalBufferContent));
}

TEST_F(Peer2PeerRcQpTest, UnsignaledReadError) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  // Change lkey to be invalid.
  sge.lkey = (sge.lkey + 10) * 5;
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  read.send_flags = read.send_flags & ~IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  // Existing hardware does not set this on error.
  // EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

TEST_F(Peer2PeerRcQpTest, ReadInvalidRkey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  // Change rkey to be invalid.
  read.wr.rdma.rkey = (read.wr.rdma.rkey + 10) * 5;
  verbs_util::PostSend(local.qp, read);

  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_ACCESS_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, expected);
  // Existing hardware does not set this on error.
  // EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  // Verify buffer is unchanged.
  EXPECT_THAT(local.buffer.span(), Each(kLocalBufferContent));
}

TEST_F(Peer2PeerRcQpTest, ReadInvalidRKeyAndInvalidLKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  // Invalidate the lkey.
  sge.lkey = (sge.lkey + 10) * 5;
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  // Invalidate the rkey too.
  read.wr.rdma.rkey = (read.wr.rdma.rkey + 10) * 5;
  verbs_util::PostSend(local.qp, read);

  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_ACCESS_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, expected);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(local.buffer.span(), Each(kLocalBufferContent));
}

TEST_F(Peer2PeerRcQpTest, BadReadAddrLocal) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());

  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  --sge.addr;

  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);

  EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));

  /* Ensure local QP has moved to error state */
  EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_RTS);
}

TEST_F(Peer2PeerRcQpTest, BadReadAddrRemote) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());

  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data() - 32,
      remote.mr->rkey);
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_REM_ACCESS_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);

  EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));

  /* Ensure both QPs have moved to error state */
  EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_ERR);
}

TEST_F(Peer2PeerRcQpTest, BasicWrite) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
}

TEST_F(Peer2PeerRcQpTest, BasicWriteLargePayload) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote),
                       CreateConnectedClientsPair(kLargePayloadPages));
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
}

TEST_F(Peer2PeerRcQpTest, UnsignaledWrite) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  write.send_flags = write.send_flags & ~IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, write);
  write.wr_id = 2;
  write.send_flags = write.send_flags | IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 2);
}

TEST_F(Peer2PeerRcQpTest, WriteInlineData) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  const size_t kWriteSize = 36;
  ASSERT_GE(remote.buffer.size(), kWriteSize) << "receiver buffer too small";
  // a vector which is not registered to pd or mr
  auto data_src = std::make_unique<std::vector<uint8_t>>(kWriteSize);
  std::fill(data_src->begin(), data_src->end(), 'c');

  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  sge.addr = reinterpret_cast<uint64_t>(data_src->data());
  sge.length = kWriteSize;
  sge.lkey = 0xDEADBEEF;  // random bad keys
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  write.send_flags |= IBV_SEND_INLINE;
  verbs_util::PostSend(local.qp, write);
  (*data_src)[0] = kLocalBufferContent;  // source can be modified immediately
  data_src.reset();  // src buffer can be deleted immediately after post_send()

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(absl::MakeSpan(remote.buffer.data(), kWriteSize), Each('c'));
  EXPECT_THAT(absl::MakeSpan(remote.buffer.data() + kWriteSize,
                             remote.buffer.data() + remote.buffer.size()),
              Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, WriteZeroByteWithImmData) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  const uint32_t kImm = 0xBADDCAFE;

  // Create a dummy buffer which shouldn't be touched.
  const unsigned int kDummyBufSize = 100;
  RdmaMemBlock dummy_buf =
      ibv_.AllocAlignedBuffer(/*pages=*/1).subblock(0, kDummyBufSize);
  ASSERT_EQ(kDummyBufSize, dummy_buf.size());
  memset(dummy_buf.data(), 'd', dummy_buf.size());
  ibv_mr* dummy_mr = ibv_.RegMr(remote.pd, dummy_buf);

  // WRITE_WITH_IMM requires a RR. Use the dummy buf for sg_list.
  ibv_sge rsge = verbs_util::CreateSge(dummy_buf.span(), dummy_mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  // Post zero sge write to remote.buffer.
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, nullptr, /*num_sge=*/0, remote.buffer.data(),
      remote.mr->rkey);
  write.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  write.imm_data = kImm;
  verbs_util::PostSend(local.qp, write);

  // Verify WRITE completion.
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);

  // Verify RECV completion.
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RECV_RDMA_WITH_IMM);
  EXPECT_EQ(completion.byte_len, 0);
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
  EXPECT_NE(completion.wc_flags & IBV_WC_WITH_IMM, 0);
  EXPECT_EQ(completion.imm_data, kImm);

  // Verify that data written to the correct buffer.
  EXPECT_THAT(remote.buffer.span(), Each(kRemoteBufferContent));
  EXPECT_THAT(dummy_buf.span(), Each('d'));
}

TEST_F(Peer2PeerRcQpTest, WriteImmData) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  const uint32_t kImm = 0xBADDCAFE;

  // Create a dummy buffer which shouldn't be touched.
  const unsigned int kDummyBufSize = 100;
  RdmaMemBlock dummy_buf =
      ibv_.AllocAlignedBuffer(/*pages=*/1).subblock(0, kDummyBufSize);
  ASSERT_EQ(kDummyBufSize, dummy_buf.size());
  memset(dummy_buf.data(), 'd', dummy_buf.size());
  ibv_mr* dummy_mr = ibv_.RegMr(remote.pd, dummy_buf);

  // WRITE_WITH_IMM requires a RR. Use the dummy buf for sg_list.
  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  rsge.addr = reinterpret_cast<uint64_t>(dummy_buf.data());
  rsge.length = dummy_buf.size();
  rsge.lkey = dummy_mr->lkey;
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  // Post write to remote.buffer.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  write.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  write.imm_data = kImm;
  verbs_util::PostSend(local.qp, write);

  // Verify WRITE completion.
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);

  // Verify RECV completion.
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RECV_RDMA_WITH_IMM);
  EXPECT_EQ(completion.byte_len, local.buffer.size());
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
  EXPECT_NE(completion.wc_flags & IBV_WC_WITH_IMM, 0);
  EXPECT_EQ(kImm, completion.imm_data);

  // Verify that data written to the correct buffer.
  EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
  EXPECT_THAT(dummy_buf.span(), Each('d'));
}

TEST_F(Peer2PeerRcQpTest, WriteImmDataInvalidRKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  const uint32_t kImm = 0xBADDCAFE;

  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, nullptr, /*num_sge=*/0);
  verbs_util::PostRecv(remote.qp, recv);

  // Post write to remote.buffer.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(),
      /*rkey=*/0xDEADBEEF);
  write.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  write.imm_data = kImm;
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_REM_ACCESS_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
  EXPECT_THAT(completion.status,
              AnyOf(IBV_WC_LOC_ACCESS_ERR, IBV_WC_LOC_PROT_ERR));
  EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 0);
  EXPECT_THAT(remote.buffer.span(), Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, WriteImmDataRnR) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  const uint32_t kImm = 0xBADDCAFE;

  // Post write to remote.buffer.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  write.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  write.imm_data = kImm;
  verbs_util::PostSend(local.qp, write);

  // Verify WRITE completion.
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_RNR_RETRY_EXC_ERR;
  EXPECT_EQ(completion.status, expected);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

TEST_F(Peer2PeerRcQpTest, Type1MWWrite) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_mw* mw = ibv_.AllocMw(remote.pd, IBV_MW_TYPE_1);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType1MwBind(remote.qp, mw,
                                             remote.buffer.span(), remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), mw->rkey);
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
}

TEST_F(Peer2PeerRcQpTest, Type2MWWrite) {
  if (!Introspection().SupportsType2()) {
    GTEST_SKIP() << "Needs type 2 MW.";
  }
  Client local, remote;
  constexpr uint32_t rkey = 0xBA;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_mw* mw = ibv_.AllocMw(remote.pd, IBV_MW_TYPE_2);
  ASSERT_THAT(mw, NotNull());
  ASSERT_THAT(verbs_util::ExecuteType2MwBind(
                  remote.qp, mw, remote.buffer.span(), rkey, remote.mr),
              IsOkAndHolds(IBV_WC_SUCCESS));
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), mw->rkey);
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
}

TEST_F(Peer2PeerRcQpTest, WriteInvalidLkey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  // Change lkey to be invalid.
  sge.lkey = (sge.lkey + 10) * 5;
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  // Existing hardware does not set this on error.
  // EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  // Verify buffer is unchanged.
  EXPECT_THAT(remote.buffer.span(), Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, WriteInvalidRkey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  // Change rkey to be invalid.
  write.wr.rdma.rkey = (write.wr.rdma.rkey + 10) * 5;
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_REM_ACCESS_ERR);
  // Existing hardware does not set this on error.
  // EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  // Verify buffer is unchanged.
  EXPECT_THAT(remote.buffer.span(), Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, UnsignaledWriteInvalidRkey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  // Change rkey to be invalid.
  write.wr.rdma.rkey = (write.wr.rdma.rkey + 10) * 5;
  write.send_flags = write.send_flags & ~IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_REM_ACCESS_ERR);
  // Existing hardware does not set this on error.
  // EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  // Verify buffer is unchanged.
  EXPECT_THAT(remote.buffer.span(), Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, WriteInvalidRKeyAndInvalidLKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  // Invalidate the lkey.
  sge.lkey = (sge.lkey + 10) * 5;
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  // Also invalidate the rkey.
  write.wr.rdma.rkey = (write.wr.rdma.rkey + 10) * 5;
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  // On a write the local key is checked first.
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(remote.buffer.span(), Each(kRemoteBufferContent));
}

TEST_F(Peer2PeerRcQpTest, BadWriteAddrLocal) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());

  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  --sge.addr;
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);

  EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));

  /* Ensure local QP has moved to error state */
  EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_RTS);
}

TEST_F(Peer2PeerRcQpTest, BadWriteAddrRemote) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());

  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data() - 32,
      remote.mr->rkey);
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_REM_ACCESS_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);

  EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));

  /* Ensure both QPs have moved to error state */
  EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_ERR);
}

TEST_F(Peer2PeerRcQpTest, FetchAddInvalidSize) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  // The local SGE will be used to store the value before the update.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 9;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      1);
  verbs_util::PostSend(local.qp, fetch_add);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  if (completion.status == IBV_WC_SUCCESS) {
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    // The remote should be incremented by 1.
    EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 3);
    // The local buffer should be the same as the original remote.
    EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 2);
  } else {
    EXPECT_THAT(completion.status,
                AnyOf(IBV_WC_LOC_LEN_ERR, IBV_WC_REM_ACCESS_ERR));
    // Local and remote buffer should be unchanged.
    EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
    EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
  }
}

TEST_F(Peer2PeerRcQpTest, FetchAddNoOp) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  // The local SGE will be used to store the value before the update.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      0);
  verbs_util::PostSend(local.qp, fetch_add);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);

  // The remote should remain b/c we added 0.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  // The local buffer should be the same as the remote.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 2);
}

// Note: Multiple SGEs for Atomics are specifically not supported by the IBTA
// spec, but some Mellanox NICs and its successors choose to
// support it.
TEST_F(Peer2PeerRcQpTest, FetchAddSplitSgl) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  static constexpr uint64_t kSrcContent = 0xAAAAAAAAAAAAAAAA;
  static constexpr uint64_t kDstContent = 2;
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = kSrcContent;
  *reinterpret_cast<uint64_t*>((local.buffer.data() + 8)) = kSrcContent;
  *reinterpret_cast<uint64_t*>((local.buffer.data() + 16)) = kSrcContent;
  *reinterpret_cast<uint64_t*>((local.buffer.data() + 24)) = kSrcContent;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = kDstContent;
  // The local SGE will be used to store the value before the update.
  // Going to read 0x2 from the remote and split it into 2 to store on the local
  // side:
  // 0x00 --------
  // 0x10 xxxx----
  // 0x20 xxxx----
  // 0x30 --------
  ibv_sge sgl[2];
  sgl[0] = verbs_util::CreateSge(local.buffer.subspan(8, 4), local.mr);
  sgl[1] = verbs_util::CreateSge(local.buffer.subspan(16, 4), local.mr);
  static constexpr uint64_t kIncrementAmount = 15;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, sgl, /*num_sge=*/2, remote.buffer.data(), remote.mr->rkey,
      kIncrementAmount);
  ibv_send_wr* bad_wr = nullptr;
  int result =
      ibv_post_send(local.qp, const_cast<ibv_send_wr*>(&fetch_add), &bad_wr);
  // Some NICs do not allow 2 SG entries
  if (result) return;

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  if (completion.status == IBV_WC_SUCCESS) {
    // Check destination.
    uint64_t value = *reinterpret_cast<uint64_t*>(remote.buffer.data());
    EXPECT_EQ(value, kDstContent + kIncrementAmount);
    // Check source.
    value = *reinterpret_cast<uint64_t*>(local.buffer.data());
    EXPECT_EQ(value, kSrcContent);
    value = *reinterpret_cast<uint64_t*>(local.buffer.data() + 8);
    uint64_t fetched = kDstContent;
    uint64_t expected = kSrcContent;
    memcpy(&expected, &fetched, 4);
    EXPECT_EQ(value, expected);
    value = *reinterpret_cast<uint64_t*>(local.buffer.data() + 16);
    expected = kSrcContent;
    memcpy(&expected, reinterpret_cast<uint8_t*>(&fetched) + 4, 4);
    EXPECT_EQ(value, expected);
    value = *reinterpret_cast<uint64_t*>(local.buffer.data() + 24);
    EXPECT_EQ(value, kSrcContent);
  }
}

TEST_F(Peer2PeerRcQpTest, UnsignaledFetchAdd) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  // The local SGE will be used to store the value before the update.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      10);
  fetch_add.send_flags = fetch_add.send_flags & ~IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, fetch_add);
  ibv_send_wr fetch_add2 = verbs_util::CreateFetchAddWr(
      /*wr_id=*/2, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      1);
  verbs_util::PostSend(local.qp, fetch_add2);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 2);
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  // Remote = 2 (orig) + 10 + 1 = 14.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 13);
  // Local = 2 (orig) + 10 = 12.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 12);
}

TEST_F(Peer2PeerRcQpTest, FetchAddIncrementBy1) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  // The local SGE will be used to store the value before the update.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      /*compare_add=*/1);
  verbs_util::PostSend(local.qp, fetch_add);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);

  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 3);
  // The local buffer should be the same as the remote before update.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 2);
}

// This tests increments by a value that is larger than 32 bits.
TEST_F(Peer2PeerRcQpTest, FetchAddLargeIncrement) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      68719476736);
  verbs_util::PostSend(local.qp, fetch_add);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);

  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 68719476738);
  // The local buffer should be the same as the remote before the add.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 2);
}

TEST_F(Peer2PeerRcQpTest, FetchAddUnaligned) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;

  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // The buffer is always an index in the first half of the 16 byte
  // vector so we can increment by 1 to get an unaligned address with enough
  // space.
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data() + 1,
      remote.mr->rkey, 1);
  verbs_util::PostSend(local.qp, fetch_add);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_INV_REQ_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, expected);

  // The buffers should be unmodified.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
}

TEST_F(Peer2PeerRcQpTest, FetchAddInvalidLKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  sge.lkey = sge.lkey * 13 + 7;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      1);
  verbs_util::PostSend(local.qp, fetch_add);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);

  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
  // Some providers will still carry out the add on remote, so remote buffer
  // might be incremented.
  EXPECT_THAT(*(reinterpret_cast<uint64_t*>(remote.buffer.data())),
              AnyOf(2, 3));
}

TEST_F(Peer2PeerRcQpTest, UnsignaledFetchAddError) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // There may be a more standard way to corrupt such a key.
  sge.lkey = sge.lkey * 13 + 7;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      1);
  fetch_add.send_flags = fetch_add.send_flags & ~IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, fetch_add);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
}

TEST_F(Peer2PeerRcQpTest, FetchAddInvalidRKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // There may be a more standard way to corrupt such a key.
  uint32_t corrupted_rkey = remote.mr->rkey * 13 + 7;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), corrupted_rkey,
      1);
  verbs_util::PostSend(local.qp, fetch_add);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_ACCESS_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, expected);

  // The buffers should not have changed.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, FetchAddInvalidLKeyAndInvalidRKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // Corrupt the rkey.
  uint32_t corrupted_rkey = remote.mr->rkey * 13 + 7;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), corrupted_rkey,
      1);
  // Also corrupt the lkey.
  sge.lkey = sge.lkey * 13 + 7;
  verbs_util::PostSend(local.qp, fetch_add);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_ACCESS_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, expected);

  // The buffers should not have changed.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, FetchAddUnalignedInvalidLKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // There may be a more standard way to corrupt such a key.
  sge.lkey = sge.lkey * 13 + 7;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data() + 1,
      remote.mr->rkey, 1);
  verbs_util::PostSend(local.qp, fetch_add);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_INV_REQ_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  // Our implementation checks the key first. The hardware may check the
  // alignment first.
  EXPECT_EQ(completion.status, expected);

  // The buffers should not have changed.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, FetchAddUnalignedInvalidRKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // There may be a more standard way to corrupt such a key.
  uint32_t corrupted_rkey = remote.mr->rkey * 13 + 7;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data() + 1,
      corrupted_rkey, 1);
  verbs_util::PostSend(local.qp, fetch_add);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_INV_REQ_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  // Our implementation will check the key first. The hardware may or may not
  // behave the same way.
  EXPECT_EQ(completion.status, expected);

  // The buffers should not have changed.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, CompareSwapNotEqualNoSwap) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      1, 3);
  verbs_util::PostSend(local.qp, cmp_swp);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);

  // The remote buffer should not have changed because the compare value != 2.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  // The local buffer should have had the remote value written back even though
  // the comparison wasn't true.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 2);
}

TEST_F(Peer2PeerRcQpTest, CompareSwapEqualWithSwap) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      2, 3);
  verbs_util::PostSend(local.qp, cmp_swp);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);

  // The remote buffer should get updated.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 3);
  // The local buffer should have had the remote value written back.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 2);
}

TEST_F(Peer2PeerRcQpTest, UnsignaledCompareSwap) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      2, 3);
  cmp_swp.send_flags = cmp_swp.send_flags & ~IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, cmp_swp);
  cmp_swp =
      verbs_util::CreateCompSwapWr(/*wr_id=*/1, &sge, /*num_sge=*/1,
                                   remote.buffer.data(), remote.mr->rkey, 3, 2);
  cmp_swp.wr_id = 2;
  verbs_util::PostSend(local.qp, cmp_swp);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 2);
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);

  // The remote buffer should get updated.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  // The local buffer should have had the remote value written back.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 3);
}

TEST_F(Peer2PeerRcQpTest, UnsignaledCompareSwapError) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // Corrupt the lkey.
  sge.lkey = sge.lkey * 13 + 7;
  ASSERT_EQ(sge.length, 8U);
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      2, 3);
  cmp_swp.send_flags = cmp_swp.send_flags & ~IBV_SEND_SIGNALED;
  verbs_util::PostSend(local.qp, cmp_swp);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);

  // The remote buffer should be updated because the lkey is not eagerly
  // checked.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 3);
  // The local buffer should not be updated because of the invalid lkey.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, CompareSwapInvalidLKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // Corrupt the lkey.
  sge.lkey = sge.lkey * 13 + 7;
  ASSERT_EQ(sge.length, 8U);
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      2, 3);
  verbs_util::PostSend(local.qp, cmp_swp);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);

  // The remote buffer should be updated because the lkey is not eagerly
  // checked.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 3);
  // The local buffer should not be updated because of the invalid lkey.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, CompareSwapInvalidRKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // Corrupt the lkey.
  ASSERT_EQ(sge.length, 8U);
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(),
      remote.mr->rkey + 7 * 10, 2, 3);
  verbs_util::PostSend(local.qp, cmp_swp);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_ACCESS_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, expected);

  // The remote buffer should not have changed because it will be caught by the
  // invalid rkey.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, CompareSwapInvalidRKeyAndInvalidLKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  sge.lkey = sge.lkey * 7 + 10;
  ASSERT_EQ(sge.length, 8U);
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(),
      remote.mr->rkey + 7 * 10, 2, 3);
  verbs_util::PostSend(local.qp, cmp_swp);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_ACCESS_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, expected);

  // The buffers shouldn't change because the rkey will get caught.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, CompareSwapInvalidSize) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 9), local.mr);

  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      2, 3);
  verbs_util::PostSend(local.qp, cmp_swp);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, IBV_WC_LOC_LEN_ERR);

  // The buffers should not be updated.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, CompareSwapUnaligned) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // The data gets moved to an invalid location.
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data() + 1,
      remote.mr->rkey, 2, 3);
  verbs_util::PostSend(local.qp, cmp_swp);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_INV_REQ_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, expected);

  // No buffers should change because the alignment will get caught.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, CompareSwapUnalignedInvalidRKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // The data gets moved to an invalid location and the rkey is corrupted
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data() + 1,
      remote.mr->rkey * 10 + 3, 2, 3);
  verbs_util::PostSend(local.qp, cmp_swp);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_INV_REQ_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, expected);

  // No buffers should change because the alignment will get caught.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, CompareSwapUnalignedInvalidLKey) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  // Corrupt the lkey.
  sge.lkey = sge.lkey * 10 + 7;
  ASSERT_EQ(sge.length, 8U);
  // The data gets moved to an invalid location.
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data() + 1,
      remote.mr->rkey, 2, 3);
  verbs_util::PostSend(local.qp, cmp_swp);
  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_INV_REQ_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, expected);

  // No buffers should change because the alignment will get caught.
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);
}

TEST_F(Peer2PeerRcQpTest, SgePointerChase) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  {
    // This scope causes the SGE to be cleaned up.
    ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    ibv_send_wr read =
        verbs_util::CreateReadWr(/*wr_id=*/1, &sge, /*num_sge=*/1,
                                 remote.buffer.data(), remote.mr->rkey);
    verbs_util::PostSend(local.qp, read);
  }
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_THAT(local.buffer.span(), Each(kRemoteBufferContent));
}

// Should be qp-fatal under custom transport and roce meaning that the queue
// pair will be transitioned to the error state.
TEST_F(Peer2PeerRcQpTest, RemoteFatalError) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  sge.length = 8;
  // The data gets moved to an invalid location.
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data() + 1,
      remote.mr->rkey, 2, 3);

  verbs_util::PostSend(local.qp, cmp_swp);

  enum ibv_wc_status expected = Introspection().GeneratesRetryExcOnConnTimeout()
                                    ? IBV_WC_RETRY_EXC_ERR
                                    : IBV_WC_REM_INV_REQ_ERR;
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.status, expected);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(remote.buffer.data())), 2);
  EXPECT_EQ(*(reinterpret_cast<uint64_t*>(local.buffer.data())), 1);

  // Both should have transitioned now that the completion has been received.
  EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_ERR);

  // Reset the buffer values.
  memset(local.buffer.data(), kLocalBufferContent, local.buffer.size());
  memset(remote.buffer.data(), kRemoteBufferContent, remote.buffer.size());

  // Create a second WR that should return an error.
  ibv_sge write_sg = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write =
      verbs_util::CreateWriteWr(/*wr_id=*/2, &write_sg, /*num_sge=*/1,
                                remote.buffer.data(), remote.mr->rkey);
  write.wr_id = 2;
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(local.cq));
  // This write should not have landed.
  EXPECT_EQ(completion.status, IBV_WC_WR_FLUSH_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 2);
  EXPECT_THAT(remote.buffer.span(), Each(kRemoteBufferContent));

  EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_ERR);

  ibv_wc result;
  EXPECT_EQ(ibv_poll_cq(local.cq, 1, &result), 0);
  EXPECT_EQ(ibv_poll_cq(remote.cq, 1, &result), 0);
}

TEST_F(Peer2PeerRcQpTest, QueryQpInitialState) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  ibv_qp_attr attr;
  ibv_qp_init_attr init_attr;
  ASSERT_EQ(ibv_query_qp(local.qp, &attr, IBV_QP_STATE, &init_attr), 0);
  EXPECT_EQ(attr.qp_state, IBV_QPS_RTS);

  ASSERT_EQ(ibv_query_qp(remote.qp, &attr, IBV_QP_STATE, &init_attr), 0);
  EXPECT_EQ(attr.qp_state, IBV_QPS_RTS);
}

// A set of testcases for when the remote qp is not in RTS state.
struct RemoteQpStateTestParameter {
  std::string name;
  ibv_wr_opcode opcode;
  ibv_qp_state remote_state;
};

class RemoteRcQpStateTest
    : public Peer2PeerRcQpTest,
      public testing::WithParamInterface<RemoteQpStateTestParameter> {
 protected:
  void SetUp() override { Peer2PeerRcQpTest::SetUp(); }

  absl::Status BringUpClientQp(Client& client, ibv_qp_state target_qp_state,
                               ibv_gid remote_gid, uint32_t remote_qpn) {
    if (target_qp_state == IBV_QPS_RESET) {
      return absl::OkStatus();
    }
    int result_code =
        ibv_.ModifyRcQpResetToInit(client.qp, client.port_attr.port);
    if (result_code != 0) {
      return absl::InternalError(absl::StrFormat(
          "Modify QP from RESET to INIT failed (%d)", result_code));
    }
    if (target_qp_state == IBV_QPS_INIT) {
      return absl::OkStatus();
    }
    result_code = ibv_.ModifyRcQpInitToRtr(client.qp, client.port_attr,
                                           remote_gid, remote_qpn);
    if (result_code != 0) {
      return absl::InternalError(absl::StrFormat(
          "Modify QP from INIT to RTR failed (%d)", result_code));
    }
    if (target_qp_state == IBV_QPS_RTR) {
      return absl::OkStatus();
    }
    result_code = ibv_.ModifyRcQpRtrToRts(
        client.qp, QpAttribute().set_timeout(absl::Seconds(1)));
    if (result_code != 0) {
      return absl::InternalError(absl::StrFormat(
          "Modify QP from RTR to RTS failed (%d)", result_code));
    }
    if (target_qp_state == IBV_QPS_RTS) {
      return absl::OkStatus();
    }
    if (target_qp_state == IBV_QPS_ERR) {
      return ibv_.ModifyQpToError(client.qp);
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Does not support QP in ", target_qp_state, "state."));
  }
};

TEST_P(RemoteRcQpStateTest, RemoteRcQpStateTests) {
  RemoteQpStateTestParameter param = GetParam();
  ASSERT_OK_AND_ASSIGN(Client local, CreateClient(IBV_QPT_RC));
  ASSERT_OK_AND_ASSIGN(Client remote, CreateClient(IBV_QPT_RC));
  ASSERT_OK(BringUpClientQp(local, IBV_QPS_RTS, remote.port_attr.gid,
                            remote.qp->qp_num));
  ASSERT_OK(BringUpClientQp(remote, param.remote_state, local.port_attr.gid,
                            local.qp->qp_num));

  switch (param.remote_state) {
    case IBV_QPS_RTR:
    case IBV_QPS_RTS:
      EXPECT_THAT(ExecuteRdmaOp(local, remote, param.opcode), IBV_WC_SUCCESS);
      break;
    case IBV_QPS_ERR:
      EXPECT_THAT(ExecuteRdmaOp(local, remote, param.opcode),
                  AnyOf(IBV_WC_RETRY_EXC_ERR, IBV_WC_REM_OP_ERR));
      break;
    default:
      EXPECT_THAT(ExecuteRdmaOp(local, remote, param.opcode),
                  IBV_WC_RETRY_EXC_ERR);
  }
}

std::vector<RemoteQpStateTestParameter> GenerateRemoteQpStateParameters(
    std::vector<ibv_wr_opcode> op_types, std::vector<ibv_qp_state> qp_states) {
  std::vector<RemoteQpStateTestParameter> params;
  for (const auto& op_type : op_types) {
    for (const auto& qp_state : qp_states) {
      auto OpToString = [](ibv_wr_opcode opcode) {
        switch (opcode) {
          case IBV_WR_RDMA_READ:
            return "Read";
          case IBV_WR_RDMA_WRITE:
            return "Write";
          case IBV_WR_ATOMIC_FETCH_AND_ADD:
            return "FetchAndAdd";
          case IBV_WR_ATOMIC_CMP_AND_SWP:
            return "CompareAndSwap";
          default:
            return "Unknown";
        }
      };
      auto QpStateToString = [](ibv_qp_state qp_state) {
        switch (qp_state) {
          case IBV_QPS_RESET:
            return "Reset";
          case IBV_QPS_INIT:
            return "Init";
          case IBV_QPS_RTR:
            return "Rtr";
          case IBV_QPS_RTS:
            return "Rts";
          case IBV_QPS_ERR:
            return "Err";
          default:
            return "Unknown";
        }
      };
      RemoteQpStateTestParameter param{
          .name = absl::StrCat("RemoteQp", QpStateToString(qp_state),
                               OpToString(op_type), "Test"),
          .opcode = op_type,
          .remote_state = qp_state,
      };
      params.push_back(param);
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(
    RemoteRcQpStateTest, RemoteRcQpStateTest,
    testing::ValuesIn(GenerateRemoteQpStateParameters(
        {IBV_WR_RDMA_READ, IBV_WR_RDMA_WRITE, IBV_WR_ATOMIC_FETCH_AND_ADD,
         IBV_WR_ATOMIC_CMP_AND_SWP},
        {IBV_QPS_RESET, IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS, IBV_QPS_ERR})),
    [](const testing::TestParamInfo<RemoteRcQpStateTest::ParamType>& info) {
      return info.param.name;
    });

TEST_F(Peer2PeerRcQpTest, WriteBatchedWr) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote),
                       CreateConnectedClientsPair(kLargePayloadPages));

  uint32_t batch_size = verbs_util::GetQpCap(local.qp).max_send_wr;
  uint32_t sge_size = local.buffer.size() / batch_size;

  // Address in each sge points to buffer of equal size
  // Offset for each buffer local and remote = wr_id * size, here wr_id = i
  // Length of each buffer = size
  // sge[i] is used to create WR write_batch[i]
  std::vector<ibv_sge> sge(batch_size);
  std::vector<ibv_send_wr> write_batch(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    sge[i] =
        verbs_util::CreateSge(local.buffer.subspan(
                                  /*offset=*/i * sge_size, /*size=*/sge_size),
                              local.mr);
    write_batch[i] = verbs_util::CreateWriteWr(
        /*wr_id=*/i, &sge[i], /*num_sge=*/1,
        remote.buffer.data() + i * sge_size, remote.mr->rkey);
    write_batch[i].next = (i != batch_size - 1) ? &write_batch[i + 1] : nullptr;
  }

  verbs_util::PostSend(local.qp, write_batch[0]);

  uint32_t completions = 0;
  while (completions < batch_size) {
    absl::StatusOr<ibv_wc> completion = verbs_util::WaitForCompletion(local.cq);
    if (completion.ok()) {
      EXPECT_EQ(completion->status, IBV_WC_SUCCESS);
      EXPECT_EQ(completion->qp_num, local.qp->qp_num);
      EXPECT_EQ(completion->wr_id, completions++);
    }
  }
}

TEST_F(Peer2PeerRcQpTest, SendRecvBatchedWr) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote),
                       CreateConnectedClientsPair(kLargePayloadPages));

  uint32_t send_queue_size = verbs_util::GetQpCap(local.qp).max_send_wr;
  uint32_t recv_queue_size = verbs_util::GetQpCap(remote.qp).max_recv_wr;
  uint32_t batch_size = std::min(send_queue_size, recv_queue_size);
  uint32_t sge_size = local.buffer.size() / batch_size;

  // Address in each sge points to buffer of equal size
  // Offset for each buffer = wr_id * size, here wr_id = i
  // length of each buffer = size
  // lsge[i] is used to create WR send_batch[i]
  // Similar batching is done for recv WRs
  std::vector<ibv_sge> lsge(batch_size);
  std::vector<ibv_send_wr> send_batch(batch_size);
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    for (uint32_t i = 0; i < batch_size; ++i) {
      lsge[i] =
          verbs_util::CreateSge(local.buffer.subspan(
                                    /*offset=*/i * sge_size, /*size=*/sge_size),
                                local.mr);
      send_batch[i] = verbs_util::CreateSendWr(/*wr_id=*/i, &lsge[i],
                                               /*num_sge=*/1);
      send_batch[i].next = (i != batch_size - 1) ? &send_batch[i + 1] : nullptr;
    }
    verbs_util::PostSend(local.qp, send_batch[0]);
  }
  synchronise();

  std::vector<ibv_sge> rsge(batch_size);
  std::vector<ibv_recv_wr> recv_batch(batch_size);
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    for (uint32_t i = 0; i < batch_size; ++i) {
      rsge[i] =
          verbs_util::CreateSge(remote.buffer.subspan(
                                    /*offset=*/i * sge_size, /*size=*/sge_size),
                                remote.mr);
      recv_batch[i] = verbs_util::CreateRecvWr(/*wr_id=*/i, &rsge[i],
                                               /*num_sge=*/1);
      recv_batch[i].next = (i != batch_size - 1) ? &recv_batch[i + 1] : nullptr;
    }
    verbs_util::PostRecv(remote.qp, recv_batch[0]);
  }
  synchronise();

  uint32_t send_completions = 0;
  uint32_t recv_completions = 0;

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    while (send_completions < batch_size) {
      absl::StatusOr<ibv_wc> completion =
          verbs_util::WaitForCompletion(local.cq);
      if (completion.ok()) {
        EXPECT_EQ(completion->status, IBV_WC_SUCCESS);
        EXPECT_EQ(completion->qp_num, local.qp->qp_num);
        EXPECT_EQ(completion->wr_id, send_completions++);
      }
    }
  }
  synchronise();
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    while (recv_completions < batch_size) {
      absl::StatusOr<ibv_wc> completion =
          verbs_util::WaitForCompletion(remote.cq);
      if (completion.ok()) {
        EXPECT_EQ(completion->status, IBV_WC_SUCCESS);
        EXPECT_EQ(completion->qp_num, remote.qp->qp_num);
        EXPECT_EQ(completion->wr_id, recv_completions++);
      }
    }
  }
  synchronise();
}

TEST_F(Peer2PeerRcQpTest, FullSubmissionQueue) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  uint32_t send_queue_size = verbs_util::GetQpCap(local.qp).max_send_wr;
  if (send_queue_size % 2) {
    // The test assume local send queue has even size in order to hit its
    // boundary. If not, we round it down and this test will not hit QP's
    // boundary.
    --send_queue_size;
  }
  uint32_t batch_size = send_queue_size / 2;

  // Basic Read.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  // Submit a batch at a time.
  std::vector<ibv_send_wr> submissions(batch_size, read);
  for (uint32_t i = 0; i < batch_size - 1; ++i) {
    submissions[i].next = &submissions[i + 1];
  }
  submissions[batch_size - 1].next = nullptr;

  // At this value we should issue more work.
  uint32_t target_outstanding = send_queue_size - batch_size;
  std::vector<ibv_wc> completions(batch_size);
  uint32_t outstanding = 0;
  uint32_t total = 0;
  // Issue 20 batches of work.
  while (total < batch_size * 20) {
    if (outstanding <= target_outstanding) {
      verbs_util::PostSend(local.qp, submissions[0]);
      outstanding += batch_size;
    }
    // Wait a little.
    sched_yield();
    absl::SleepFor(absl::Milliseconds(20));
    // Poll completions
    int count = ibv_poll_cq(local.cq, batch_size, completions.data());
    total += count;
    outstanding -= count;
  }

  // Wait for outstanding to avoid any leaks...
  while (outstanding > 0) {
    // Wait a little.
    sched_yield();
    absl::SleepFor(absl::Milliseconds(20));
    // Poll completions
    int count = ibv_poll_cq(local.cq, batch_size, completions.data());
    outstanding -= count;
  }
}

// This test issues 2 reads. The first read has an invalid lkey that will send
// the requester into an error state. The second read is a valid read that will
// not land because the qp will be in an error state.
TEST_F(Peer2PeerRcQpTest, FlushErrorPollTogether) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());

  ibv_sge read_sg = verbs_util::CreateSge(local.buffer.span(), local.mr);
  read_sg.lkey = read_sg.lkey * 10 + 7;
  ibv_send_wr read =
      verbs_util::CreateReadWr(/*wr_id=*/1, &read_sg, /*num_sge=*/1,
                               remote.buffer.data(), remote.mr->rkey);
  verbs_util::PostSend(local.qp, read);

  ibv_sge read_sg_2 = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read_2 =
      verbs_util::CreateWriteWr(/*wr_id=*/1, &read_sg_2, /*num_sge=*/1,
                                remote.buffer.data(), remote.mr->rkey);
  read_2.wr_id = 2;
  verbs_util::PostSend(local.qp, read_2);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
  // Verify buffer is unchanged.
  EXPECT_THAT(local.buffer.span(), Each(kLocalBufferContent));
  EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_RTS);

  ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_WR_FLUSH_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 2);
  EXPECT_THAT(local.buffer.span(), Each(kLocalBufferContent));

  EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
  EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_RTS);
}

class RnrRecoverTest
    : public Peer2PeerRcQpTest,
      public testing::WithParamInterface<verbs_util::IbvOperations> {};

TEST_P(RnrRecoverTest, RnrRecoverTests) {
  verbs_util::IbvOperations operation = GetParam();

  uint32_t valid_inline_size, invalid_inline_size;
  ASSERT_OK_AND_ASSIGN(std::tie(valid_inline_size, invalid_inline_size),
                       DetermineInlineLimits());

  // 7 is the magic number for infinite retries.
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(
      std::tie(local, remote),
      CreateConnectedClientsPair(
          kPages, QpInitAttribute().set_max_inline_data(valid_inline_size),
          QpAttribute().set_rnr_retry(7)));

  const uint32_t kImm = 0xBADDCAFE;

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);

  if (operation == verbs_util::IbvOperations::SendInline) {
    // a vector which is not registered to pd or mr
    auto data_src = std::make_unique<std::vector<uint8_t>>(valid_inline_size);
    std::fill(data_src->begin(), data_src->end(), 'c');

    // Modify lsge to send Inline data with addr of the vector
    lsge.addr = reinterpret_cast<uint64_t>(data_src->data());
    lsge.length = valid_inline_size;
    lsge.lkey = 0xDEADBEEF;  // random bad keys
  }

  ibv_send_wr lwqe =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  ibv_recv_wr rwqe =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);

  switch (operation) {
    case verbs_util::IbvOperations::WriteWithImm:
      lwqe = verbs_util::CreateWriteWr(/*wr_id=*/1, &lsge, /*num_sge=*/1,
                                       remote.buffer.data(), remote.mr->rkey);
      lwqe.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      lwqe.imm_data = kImm;

      rwqe = verbs_util::CreateRecvWr(/*wr_id=*/0, nullptr, /*num_sge=*/0);
      break;

    case verbs_util::IbvOperations::SendInline:
      lwqe.send_flags |= IBV_SEND_INLINE;
      break;

    case verbs_util::IbvOperations::SendWithImm:
      lwqe.opcode = IBV_WR_SEND_WITH_IMM;
      lwqe.imm_data = kImm;
      break;
  }

  verbs_util::PostSend(local.qp, lwqe);

  uint32_t i = 0;
  uint32_t cq_polling_tries = 5;

  while (1) {
    if (i++ == cq_polling_tries) {
      verbs_util::PostRecv(remote.qp, rwqe);
      VLOG(1) << "Posted Recv WR";
    }

    absl::StatusOr<ibv_wc> lcompletion =
        verbs_util::WaitForCompletion(local.cq);
    absl::StatusOr<ibv_wc> rcompletion =
        verbs_util::WaitForCompletion(remote.cq);

    if (lcompletion.ok() || rcompletion.ok()) {
      VLOG(1) << absl::StrCat("Polled CQ try ", i, " - Completion Received.");

      EXPECT_EQ(lcompletion->status, IBV_WC_SUCCESS);
      EXPECT_EQ(rcompletion->status, IBV_WC_SUCCESS);

      EXPECT_EQ(lcompletion->wr_id, 1);
      EXPECT_EQ(rcompletion->wr_id, 0);

      EXPECT_EQ(lcompletion->qp_num, local.qp->qp_num);
      EXPECT_EQ(rcompletion->qp_num, remote.qp->qp_num);

      if (operation == verbs_util::IbvOperations::SendWithImm ||
          operation == verbs_util::IbvOperations::WriteWithImm) {
        EXPECT_NE(rcompletion->wc_flags & IBV_WC_WITH_IMM, 0);
        EXPECT_EQ(kImm, rcompletion->imm_data);
      }

      break;
    } else {
      VLOG(1) << absl::StrCat("Polled CQ try ", i, " - ")
              << lcompletion.status();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    RnrRecoverTest, RnrRecoverTest,
    testing::Values(verbs_util::IbvOperations::Send,
                    verbs_util::IbvOperations::SendWithImm,
                    verbs_util::IbvOperations::SendInline,
                    verbs_util::IbvOperations::WriteWithImm),
    [](const testing::TestParamInfo<RnrRecoverTest::ParamType>& info) {
      std::string name = [info]() {
        switch (info.param) {
          case verbs_util::IbvOperations::Send:
            return "Send";
          case verbs_util::IbvOperations::SendInline:
            return "SendInline";
          case verbs_util::IbvOperations::SendWithImm:
            return "SendWithImm";
          case verbs_util::IbvOperations::WriteWithImm:
            return "WriteWithImm";
          default:
            return "Unknown";
        }
      }();
      return name;
    });

class RcQpRecoverTest
    : public Peer2PeerRcQpTest,
      public testing::WithParamInterface<verbs_util::IbvOperations> {
 protected:
  static constexpr uint32_t kImm = 0xBADDCAFE;
  void MoveQpErrorState(Client& local, Client& remote) {
    // Moves RC QP pair to error state.
    ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    ibv_send_wr read = verbs_util::CreateReadWr(
        /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data() - 32,
        remote.mr->rkey);
    verbs_util::PostSend(local.qp, read);
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_REM_ACCESS_ERR);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));
    // Ensure both QPs have moved to error state
    ASSERT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
    ASSERT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_ERR);
  }

  void RecoverQpErrorState(Client& local, Client& remote) {
    // Recovers RC QP pair from Error state and moves it to RTS state.
    ibv_qp_attr attr;
    attr.qp_state = IBV_QPS_RESET;
    ibv_qp_attr_mask attr_mask = IBV_QP_STATE;
    ASSERT_EQ(ibv_modify_qp(local.qp, &attr, attr_mask), 0);
    ASSERT_EQ(ibv_modify_qp(remote.qp, &attr, attr_mask), 0);
    QpInitAttribute qp_init_attr = QpInitAttribute();
    QpAttribute qp_attr = QpAttribute();
    ASSERT_OK(ibv_.ModifyRcQpResetToRts(local.qp, local.port_attr,
                                        remote.port_attr.gid, remote.qp->qp_num,
                                        qp_attr));
    ASSERT_OK(ibv_.ModifyRcQpResetToRts(remote.qp, remote.port_attr,
                                        local.port_attr.gid, local.qp->qp_num,
                                        qp_attr));
    ASSERT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_RTS);
    ASSERT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_RTS);
  }

  void ValidateLocalCompletion(Client& local,
                               verbs_util::IbvOperations operation) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    if (operation == verbs_util::IbvOperations::Send ||
        operation == verbs_util::IbvOperations::SendInline ||
        operation == verbs_util::IbvOperations::SendWithImm) {
      EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    } else {
      EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
    }
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }

  void ValidateRemoteCompletion(Client& remote,
                                verbs_util::IbvOperations operation) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    if (operation == verbs_util::IbvOperations::SendWithImm ||
        operation == verbs_util::IbvOperations::WriteWithImm) {
      EXPECT_NE(completion.wc_flags & IBV_WC_WITH_IMM, 0);
      EXPECT_EQ(kImm, completion.imm_data);
    } else if (operation == verbs_util::IbvOperations::Send) {
      EXPECT_EQ(completion.wc_flags, 0);
    }
  }
};

TEST_P(RcQpRecoverTest, RcQpRecoverTests) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  const size_t inline_size = verbs_util::GetQpCap(local.qp).max_inline_data;
  MoveQpErrorState(local, remote);
  // Recover QPs from error state
  RecoverQpErrorState(local, remote);
  verbs_util::IbvOperations operation = GetParam();
  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  // A vector which is not registered to pd or mr
  auto data_src = std::make_unique<std::vector<uint8_t>>(inline_size);
  if (operation == verbs_util::IbvOperations::SendInline ||
      operation == verbs_util::IbvOperations::WriteInline) {
    std::fill(data_src->begin(), data_src->end(), 'c');
    // Modify lsge to send Inline data with addr of the vector
    lsge.addr = reinterpret_cast<uint64_t>(data_src->data());
    lsge.length = inline_size;
    lsge.lkey = 0xDEADBEEF;  // random bad keys
  }
  if (operation != verbs_util::IbvOperations::Write &&
      operation != verbs_util::IbvOperations::WriteInline) {
    ibv_recv_wr recv = verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge,
                                                /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  ibv_send_wr wr;
  switch (operation) {
    case verbs_util::IbvOperations::Send:
    case verbs_util::IbvOperations::SendInline:
    case verbs_util::IbvOperations::SendWithImm:
      if (operation == verbs_util::IbvOperations::SendWithImm) {
        wr = verbs_util::CreateSendWithImmWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
        wr.imm_data = kImm;
      } else {
        wr = verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
        if (operation == verbs_util::IbvOperations::SendInline) {
          wr.send_flags |= IBV_SEND_INLINE;
        }
      }
      verbs_util::PostSend(local.qp, wr);
      ValidateLocalCompletion(local, operation);
      ValidateRemoteCompletion(remote, operation);
      if (operation == verbs_util::IbvOperations::SendInline) {
        EXPECT_THAT(absl::MakeSpan(remote.buffer.data(), inline_size),
                    Each('c'));
        EXPECT_THAT(absl::MakeSpan(remote.buffer.data() + inline_size,
                                   remote.buffer.data() + remote.buffer.size()),
                    Each(kRemoteBufferContent));
      } else {
        EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
      }
      break;
    case verbs_util::IbvOperations::Write:
    case verbs_util::IbvOperations::WriteInline:
    case verbs_util::IbvOperations::WriteWithImm:
      if (operation == verbs_util::IbvOperations::WriteWithImm) {
        wr = verbs_util::CreateWriteWithImmWr(/*wr_id=*/1, &lsge, /*num_sge=*/1,
                                              remote.buffer.data(),
                                              remote.mr->rkey);
        wr.imm_data = kImm;
      } else {
        wr = verbs_util::CreateWriteWr(/*wr_id=*/1, &lsge, /*num_sge=*/1,
                                       remote.buffer.data(), remote.mr->rkey);
        if (operation == verbs_util::IbvOperations::WriteInline) {
          wr.send_flags |= IBV_SEND_INLINE;
        }
      }
      verbs_util::PostSend(local.qp, wr);
      ValidateLocalCompletion(local, operation);
      if (operation == verbs_util::IbvOperations::WriteWithImm) {
        ValidateRemoteCompletion(remote, operation);
      }
      if (operation == verbs_util::IbvOperations::WriteInline) {
        EXPECT_THAT(absl::MakeSpan(remote.buffer.data(), inline_size),
                    Each('c'));
        EXPECT_THAT(absl::MakeSpan(remote.buffer.data() + inline_size,
                                   remote.buffer.data() + remote.buffer.size()),
                    Each(kRemoteBufferContent));
      } else {
        EXPECT_THAT(remote.buffer.span(), Each(kLocalBufferContent));
      }
  }
}
INSTANTIATE_TEST_SUITE_P(
    RcQpRecoverTest, RcQpRecoverTest,
    testing::Values(verbs_util::IbvOperations::Send,
                    verbs_util::IbvOperations::SendWithImm,
                    verbs_util::IbvOperations::SendInline,
                    verbs_util::IbvOperations::Write,
                    verbs_util::IbvOperations::WriteInline,
                    verbs_util::IbvOperations::WriteWithImm),
    [](const testing::TestParamInfo<RcQpRecoverTest::ParamType>& info) {
      std::string name = [info]() {
        switch (info.param) {
          case verbs_util::IbvOperations::Send:
            return "Send";
          case verbs_util::IbvOperations::SendInline:
            return "SendInline";
          case verbs_util::IbvOperations::SendWithImm:
            return "SendWithImm";
          case verbs_util::IbvOperations::Write:
            return "Write";
          case verbs_util::IbvOperations::WriteInline:
            return "WriteInline";
          case verbs_util::IbvOperations::WriteWithImm:
            return "WriteWithImm";
          default:
            return "Unknown";
        }
      }();
      return name;
    });
}  // namespace
}  // namespace rdma_unit_test
