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
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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

namespace rdma_unit_test {
namespace {

// TODO(author1): Add tests stressing SGEs.
// TODO(author1): Send with insufficient recv buffering (RC and UD).
// TODO(author2): Add QP error state check for relevant testcases.

using ::testing::AnyOf;
using ::testing::Each;
using ::testing::Ne;
using ::testing::NotNull;

class LoopbackMrcQpTest : public LoopbackFixture {
 public:
  static constexpr char kLocalBufferContent = 'a';
  static constexpr char kRemoteBufferContent = 'b';
  static constexpr int kLargePayloadPages = 128;
  static constexpr int kPages = 1;

  void SetUp() override {
    LoopbackFixture::SetUp();
    if (!Introspection().SupportsMrcQp()) {
      GTEST_SKIP() << "Nic does not support MRC QP";
    }
  }

 protected:
  absl::StatusOr<std::pair<Client, Client>> CreateConnectedClientsPair(
      int pages = kPages, QpInitAttribute qp_init_attr = QpInitAttribute(),
      QpAttribute qp_attr = QpAttribute()) {
    struct verbs_util::conn_attr local_host, remote_host;
    int rc;

    ASSIGN_OR_RETURN(Client local,
                     CreateClient(IBV_QPT_RC, pages, qp_init_attr));
    std::fill_n(local.buffer.data(), local.buffer.size(), kLocalBufferContent);
    ASSIGN_OR_RETURN(Client remote,
                     CreateClient(IBV_QPT_RC, pages, qp_init_attr));
    std::fill_n(remote.buffer.data(), remote.buffer.size(),
                kRemoteBufferContent);

    // Execute Tests in Loopback mode
    if (!verbs_util::peer_mode()) {
      RETURN_IF_ERROR(ibv_.ModifyRcQpResetToRts(local.qp, local.port_attr,
                                                remote.port_attr.gid,
                                                remote.qp->qp_num, qp_attr));
      RETURN_IF_ERROR(ibv_.ModifyRcQpResetToRts(remote.qp, remote.port_attr,
                                                local.port_attr.gid,
                                                local.qp->qp_num, qp_attr));
      return std::make_pair(local, remote);
    }

    // Execute Tests in Peer To Peer mode
    local_host.psn = lrand48() & 0xffffff;
    if (verbs_util::is_client()) {
      local_host.gid = local.port_attr.gid;
      local_host.lid = local.port_attr.attr.lid;
      local_host.qpn = local.qp->qp_num;
      rc = verbs_util::RunClient(local_host, remote_host);
      if (rc) {
        LOG(FATAL) << "Failed to get remote conn attributes, Err:" << rc;
      }
      rc = connect_peer(local.qp, local_host.psn, remote_host,
                        remote.port_attr.port, remote.port_attr.gid_index);
    } else {
      local_host.gid = remote.port_attr.gid;
      local_host.lid = remote.port_attr.attr.lid;
      local_host.qpn = remote.qp->qp_num;
      rc = verbs_util::RunServer(local_host, remote_host);
      if (rc) {
        LOG(FATAL) << "Failed to get remote conn attributes, Err:" << rc;
      }
      rc = connect_peer(remote.qp, local_host.psn, remote_host,
                        remote.port_attr.port, remote.port_attr.gid_index);
    }
    if (rc) {
      LOG(FATAL) << "Failed to connect Peer node, Err:" << rc;
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

TEST_F(LoopbackMrcQpTest, BasicWrite) {
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

TEST_F(LoopbackMrcQpTest, BasicWriteLargePayload) {
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

TEST_F(LoopbackMrcQpTest, UnsignaledWrite) {
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

TEST_F(LoopbackMrcQpTest, WriteSmallInlineData) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateConnectedClientsPair());
  const size_t kWriteSize = 32;
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

TEST_F(LoopbackMrcQpTest, WriteLargeInlineData) {
  uint32_t valid_inline_size, invalid_inline_size;
  ASSERT_OK_AND_ASSIGN(std::tie(valid_inline_size, invalid_inline_size),
                       DetermineInlineLimits());
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(
      std::tie(local, remote),
      CreateConnectedClientsPair(
          kPages, QpInitAttribute().set_max_inline_data(valid_inline_size)));
  ASSERT_GE(remote.buffer.size(), valid_inline_size) << "receiver buffer too small";
  // a vector which is not registered to pd or mr
  auto data_src = std::make_unique<std::vector<uint8_t>>(valid_inline_size);
  std::fill(data_src->begin(), data_src->end(), 'c');

  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  sge.addr = reinterpret_cast<uint64_t>(data_src->data());
  sge.length = valid_inline_size;
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
  EXPECT_THAT(absl::MakeSpan(remote.buffer.data(), valid_inline_size), Each('c'));
  EXPECT_THAT(absl::MakeSpan(remote.buffer.data() + valid_inline_size,
                             remote.buffer.data() + remote.buffer.size()),
              Each(kRemoteBufferContent));
}

TEST_F(LoopbackMrcQpTest, WriteExceedMaxInlineData) {
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

  ibv_sge sge{
      .addr = reinterpret_cast<uint64_t>(data_src->data()),
      .length = static_cast<uint32_t>(data_src->size()),
      .lkey = 0xDEADBEEF,  // random bad keys
  };

  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  write.send_flags |= IBV_SEND_INLINE;
  ibv_send_wr* bad_wr;
  EXPECT_THAT(ibv_post_send(local.qp, &write, &bad_wr),
              AnyOf(EPERM, ENOMEM, EINVAL));
  EXPECT_TRUE(verbs_util::ExpectNoCompletion(local.cq));
}

TEST_F(LoopbackMrcQpTest, WriteZeroByteWithImmData) {
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

TEST_F(LoopbackMrcQpTest, WriteImmData) {
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

TEST_F(LoopbackMrcQpTest, WriteImmDataInvalidRKey) {
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

TEST_F(LoopbackMrcQpTest, WriteImmDataRnR) {
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

TEST_F(LoopbackMrcQpTest, Type1MWWrite) {
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

TEST_F(LoopbackMrcQpTest, Type2MWWrite) {
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

TEST_F(LoopbackMrcQpTest, WriteInvalidLkey) {
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

TEST_F(LoopbackMrcQpTest, WriteInvalidRkey) {
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

TEST_F(LoopbackMrcQpTest, UnsignaledWriteInvalidRkey) {
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

TEST_F(LoopbackMrcQpTest, WriteInvalidRKeyAndInvalidLKey) {
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

TEST_F(LoopbackMrcQpTest, BadWriteAddrLocal) {
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

TEST_F(LoopbackMrcQpTest, BadWriteAddrRemote) {
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

TEST_F(LoopbackMrcQpTest, QueryQpInitialState) {
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

class RemoteMrcQpStateTest
    : public LoopbackMrcQpTest,
      public testing::WithParamInterface<RemoteQpStateTestParameter> {
 protected:
  void SetUp() override { LoopbackMrcQpTest::SetUp(); }

  absl::Status BringUpClientQp(Client& client, ibv_qp_state target_qp_state,
                               ibv_gid remote_gid, uint32_t remote_qpn, uint32_t timeout_ms) {
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
        client.qp, QpAttribute().set_timeout(absl::Milliseconds(timeout_ms)));
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

TEST_P(RemoteMrcQpStateTest, RemoteMrcQpStateTests) {
  RemoteQpStateTestParameter param = GetParam();
  ASSERT_OK_AND_ASSIGN(Client local, CreateClient(IBV_QPT_RC));
  ASSERT_OK_AND_ASSIGN(Client remote, CreateClient(IBV_QPT_RC));
  ASSERT_OK(BringUpClientQp(local, IBV_QPS_RTS, remote.port_attr.gid,
                            remote.qp->qp_num, /*timeout_ms*/150));
  ASSERT_OK(BringUpClientQp(remote, param.remote_state, local.port_attr.gid,
                            local.qp->qp_num, /*timeout_ms*/150));

  ASSERT_OK_AND_ASSIGN(ibv_wc_status result, ExecuteRdmaOp(local, remote, param.opcode, absl::Seconds(4)));
  switch (param.remote_state) {
    case IBV_QPS_RTR:
    case IBV_QPS_RTS:
      EXPECT_EQ(result, IBV_WC_SUCCESS);
      break;
    case IBV_QPS_ERR:
      EXPECT_THAT(result, AnyOf(IBV_WC_RETRY_EXC_ERR, IBV_WC_REM_OP_ERR));
      break;
    default:
      EXPECT_EQ(result, IBV_WC_RETRY_EXC_ERR);
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
    RemoteMrcQpStateTest, RemoteMrcQpStateTest,
    testing::ValuesIn(GenerateRemoteQpStateParameters(
        {IBV_WR_RDMA_READ, IBV_WR_RDMA_WRITE, IBV_WR_ATOMIC_FETCH_AND_ADD,
         IBV_WR_ATOMIC_CMP_AND_SWP},
        {IBV_QPS_RESET, IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS, IBV_QPS_ERR})),
    [](const testing::TestParamInfo<RemoteMrcQpStateTest::ParamType>& info) {
      return info.param.name;
    });

TEST_F(LoopbackMrcQpTest, WriteBatchedWr) {
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
    ASSERT_OK(completion);
    EXPECT_EQ(completion->status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion->qp_num, local.qp->qp_num);
    EXPECT_EQ(completion->wr_id, completions++);
  }
}

TEST_F(LoopbackMrcQpTest, SendRecvBatchedWr) {
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

  uint32_t send_completions = 0;
  uint32_t recv_completions = 0;

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    while (send_completions < batch_size) {
      ASSERT_OK_AND_ASSIGN(ibv_wc completion, verbs_util::WaitForCompletion(local.cq));
      EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
      EXPECT_EQ(completion.qp_num, local.qp->qp_num);
      EXPECT_EQ(completion.wr_id, send_completions++);
    }
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    while (recv_completions < batch_size) {
      ASSERT_OK_AND_ASSIGN(ibv_wc completion, verbs_util::WaitForCompletion(remote.cq));
      EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
      EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
      EXPECT_EQ(completion.wr_id, recv_completions++);
    }
  }
}

// This test issues 2 reads. The first read has an invalid lkey that will send
// the requester into an error state. The second read is a valid read that will
// not land because the qp will be in an error state.
TEST_F(LoopbackMrcQpTest, FlushErrorPollTogether) {
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

class MrcRnrRecoverTest
    : public LoopbackMrcQpTest,
      public testing::WithParamInterface<verbs_util::IbvOperations> {};

TEST_P(MrcRnrRecoverTest, MrcRnrRecoverTests) {
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

  uint32_t cq_polling_tries = 5;

  for (uint32_t i = 0; i < cq_polling_tries; i++) {
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(local.cq));
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));
  }
  verbs_util::PostRecv(remote.qp, rwqe);
  VLOG(1) << "Posted Recv WR";

  ASSERT_OK_AND_ASSIGN(ibv_wc lcompletion,
      verbs_util::WaitForCompletion(local.cq));
  ASSERT_OK_AND_ASSIGN(ibv_wc rcompletion,
      verbs_util::WaitForCompletion(remote.cq));

  EXPECT_EQ(lcompletion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(rcompletion.status, IBV_WC_SUCCESS);

  EXPECT_EQ(lcompletion.wr_id, 1);
  EXPECT_EQ(rcompletion.wr_id, 0);

  EXPECT_EQ(lcompletion.qp_num, local.qp->qp_num);
  EXPECT_EQ(rcompletion.qp_num, remote.qp->qp_num);

  if (operation == verbs_util::IbvOperations::SendWithImm ||
      operation == verbs_util::IbvOperations::WriteWithImm) {
    EXPECT_NE(rcompletion.wc_flags & IBV_WC_WITH_IMM, 0);
    EXPECT_EQ(kImm, rcompletion.imm_data);
  }
}

INSTANTIATE_TEST_SUITE_P(
    MrcRnrRecoverTest, MrcRnrRecoverTest,
    testing::Values(verbs_util::IbvOperations::WriteWithImm),
    [](const testing::TestParamInfo<MrcRnrRecoverTest::ParamType>& info) {
      std::string name = [info]() {
        switch (info.param) {
          case verbs_util::IbvOperations::WriteWithImm:
            return "WriteWithImm";
          default:
            return "Unknown";
        }
      }();
      return name;
    });

class MrcQpRecoverTest
    : public LoopbackMrcQpTest,
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
    EXPECT_EQ(completion.opcode, IBV_WC_RDMA_WRITE);
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
    if (operation == verbs_util::IbvOperations::WriteWithImm) {
      EXPECT_NE(completion.wc_flags & IBV_WC_WITH_IMM, 0);
      EXPECT_EQ(kImm, completion.imm_data);
    }
  }
};

TEST_P(MrcQpRecoverTest, MrcQpRecoverTests) {
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
    MrcQpRecoverTest, MrcQpRecoverTest,
    testing::Values(verbs_util::IbvOperations::Write,
                    verbs_util::IbvOperations::WriteInline,
                    verbs_util::IbvOperations::WriteWithImm),
    [](const testing::TestParamInfo<MrcQpRecoverTest::ParamType>& info) {
      std::string name = [info]() {
        switch (info.param) {
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
