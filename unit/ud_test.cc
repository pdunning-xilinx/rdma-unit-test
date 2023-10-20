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

#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <sched.h>
#include <sys/socket.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "infiniband/verbs.h"
#include "internal/handle_garble.h"
#include "internal/verbs_attribute.h"
#include "public/rdma_memblock.h"
#include "public/status_matchers.h"
#include "public/verbs_helper_suite.h"
#include "public/verbs_util.h"
#include "unit/loopback_fixture.h"

namespace rdma_unit_test {
namespace {

// TODO(author1): UD send to invalid dst IP (via AH).
// TODO(author1): UD with GRH split across many SGE.
// TODO(author1): Send between RC and UD.
// TODO(author1): UD Send with first SGE with invalid lkey.

using ::testing::AnyOf;
using ::testing::Each;
using ::testing::NotNull;

class LoopbackUdQpTest : public LoopbackFixture {
 public:
  static constexpr int kQKey = 200;
  static constexpr char kLocalBufferContent = 'a';
  static constexpr char kRemoteBufferContent = 'b';

 protected:
  absl::StatusOr<std::pair<Client, Client>> CreateUdClientsPair(
      size_t pages = 1, QpInitAttribute qp_init_attr = QpInitAttribute()) {
    struct verbs_util::conn_attr local_host, remote_host;
    int rc;

    ASSIGN_OR_RETURN(Client local, CreateClient(IBV_QPT_UD, pages,
                                                qp_init_attr));
    std::fill_n(local.buffer.data(), local.buffer.size(), kLocalBufferContent);
    ASSIGN_OR_RETURN(Client remote, CreateClient(IBV_QPT_UD, pages,
                                                 qp_init_attr));
    std::fill_n(remote.buffer.data(), remote.buffer.size(),
                kRemoteBufferContent);
    // Execute Tests in Loopback mode
    RETURN_IF_ERROR(ibv_.ModifyUdQpResetToRts(local.qp, kQKey));
    RETURN_IF_ERROR(ibv_.ModifyUdQpResetToRts(remote.qp, kQKey));

    // Execute Tests in Peer To Peer mode, to update local/remote gid
    // and qp_num with the peer host/device information
    if (verbs_util::peer_mode()) {
      local_host.psn = lrand48() & 0xffffff;
      if (verbs_util::is_client()) {
        local_host.gid = local.port_attr.gid;
        local_host.lid = local.port_attr.attr.lid;
        local_host.qpn = local.qp->qp_num;
        rc = verbs_util::RunClient(local_host, remote_host);
	if (rc) {
	  LOG(FATAL) << "Failed to get remote conn attributes, Err:" << rc;
	}
        remote.port_attr.gid = remote_host.gid;
        remote.qp->qp_num = remote_host.qpn;
      } else {
        local_host.gid = remote.port_attr.gid;
        local_host.lid = remote.port_attr.attr.lid;
        local_host.qpn = remote.qp->qp_num;
        rc = verbs_util::RunServer(local_host, remote_host);
	if (rc) {
	  LOG(FATAL) << "Failed to get remote conn attributes, Err:" << rc;
	}
        local.port_attr.gid = remote_host.gid;
        local.qp->qp_num = remote_host.qpn;
      }
    }

    return std::make_pair(local, remote);
  }

  // In ROCE2.0, the GRH is replaced by IP headers.
  // ipv4: the first 20 bytes of the GRH buffer is undefined and last 20 bytes
  // are the ipv4 headers.
  // ipv6: the whole GRH is replaced by an ipv6 header.
  iphdr ExtractIp4Header(void* buffer) {
    // TODO(author2): Verify checksum.
    iphdr iphdr;
    memcpy(&iphdr, static_cast<uint8_t*>(buffer) + 20, sizeof(iphdr));
    return iphdr;
  }
};

TEST_F(LoopbackUdQpTest, Send) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.
  constexpr int kGrhHeaderBytes = 40;
  Client local, remote;
  ibv_wc completion;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kPayloadLength + sizeof(ibv_grh);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length = kPayloadLength;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index, remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.byte_len, (kGrhHeaderBytes + kPayloadLength));
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    absl::Span<uint8_t> recv_payload =
        remote.buffer.subspan(sizeof(ibv_grh), kPayloadLength);
    EXPECT_THAT(recv_payload, Each(kLocalBufferContent));
  }
}

TEST_F(LoopbackUdQpTest, SendImmData) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.
  constexpr int kGrhHeaderBytes = sizeof(ibv_grh);
  const uint32_t kImm = 0xBADDCAFE;

  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kPayloadLength + kGrhHeaderBytes;
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length = kPayloadLength;
    ibv_send_wr send =
        verbs_util::CreateSendWithImmWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    send.imm_data = kImm;
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index,
                               remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.byte_len, (kGrhHeaderBytes + kPayloadLength));
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    absl::Span<uint8_t> recv_payload =
        remote.buffer.subspan(sizeof(ibv_grh), kPayloadLength);
    EXPECT_THAT(recv_payload, Each(kLocalBufferContent));
    EXPECT_NE(completion.wc_flags & IBV_WC_WITH_IMM, 0);
    EXPECT_EQ(kImm, completion.imm_data);
  }
}

TEST_F(LoopbackUdQpTest, SendInlineData) {
  constexpr int kGrhHeaderBytes = sizeof(ibv_grh);
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());
  uint32_t kPayloadLength = verbs_util::GetQpCap(local.qp).max_inline_data;

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kPayloadLength + kGrhHeaderBytes;
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    // a vector which is not registered to pd or mr
    auto data_src = std::make_unique<std::vector<uint8_t>>(kPayloadLength);
    std::fill(data_src->begin(), data_src->end(), 'c');
    ibv_sge lsge{
        .addr = reinterpret_cast<uint64_t>(data_src->data()),
        .length = kPayloadLength,
        .lkey = 0xDEADBEEF,  // random bad keys
    };

    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    send.send_flags |= IBV_SEND_INLINE;
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index,
                               remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.byte_len, (kGrhHeaderBytes + kPayloadLength));
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    absl::Span<uint8_t> recv_payload =
        remote.buffer.subspan(sizeof(ibv_grh), kPayloadLength);
    EXPECT_THAT(recv_payload, Each('c'));
  }
}

TEST_F(LoopbackUdQpTest, SendLargerThanMtu) {
  constexpr size_t kPages = 20;
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair(kPages));
  const int kPayloadLength =
      verbs_util::VerbsMtuToInt(local.port_attr.attr.active_mtu) + 10;
  ASSERT_GT(kPages * kPageSize, kPayloadLength);

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  rsge.length = kPayloadLength + sizeof(ibv_grh);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  lsge.length = kPayloadLength;
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  send.wr.ud.ah = ah;
  send.wr.ud.remote_qpn = remote.qp->qp_num;
  send.wr.ud.remote_qkey = kQKey;
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_LEN_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

TEST_F(LoopbackUdQpTest, SendEmptySgl) {
  constexpr int kGrhHeaderBytes = sizeof(ibv_grh);
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kGrhHeaderBytes;
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, nullptr, /*num_sge=*/0);
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index,
                               remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.byte_len, kGrhHeaderBytes);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
  }
}

TEST_F(LoopbackUdQpTest, SendOnlyImmData) {
  constexpr int kGrhHeaderBytes = sizeof(ibv_grh);
  const uint32_t kImm = 0xBADDCAFE;

  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kGrhHeaderBytes;
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_send_wr send =
        verbs_util::CreateSendWithImmWr(/*wr_id=*/1, nullptr, /*num_sge=*/0);
    send.imm_data = kImm;
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index,
                               remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.byte_len, kGrhHeaderBytes);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    EXPECT_NE(completion.wc_flags & IBV_WC_WITH_IMM, 0);
    EXPECT_EQ(kImm, completion.imm_data);
  }
}

TEST_F(LoopbackUdQpTest, SendRnr) {
  static constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  lsge.length = kPayloadLength;
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  send.wr.ud.ah = ah;
  send.wr.ud.remote_qpn = remote.qp->qp_num;
  send.wr.ud.remote_qkey = kQKey;
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

/* According to ROCE v2 Annex A17.9.2, when setting traffic class,
   it should be correctly reflected in GRH, and should be the same on both
   ends. */
TEST_F(LoopbackUdQpTest, SendTrafficClass) {
  constexpr int kPayloadLength = 1000;
  constexpr uint8_t traffic_class = 0xff;
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kPayloadLength + sizeof(ibv_grh);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length = kPayloadLength;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    // Set with customized traffic class.
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index, remote.port_attr.gid,
                               AhAttribute().set_traffic_class(traffic_class));
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc send_completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(send_completion.status, IBV_WC_SUCCESS);
  }

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc recv_completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(recv_completion.status, IBV_WC_SUCCESS);

    int ip_family = verbs_util::GetIpAddressType(local.port_attr.gid);
    ASSERT_NE(ip_family, -1);
    // On RoCE 2.0, the GRH (global routing header) is replaced by the IP header.
    if (ip_family == AF_INET) {
      // The ipv4 header is located at the lower bytes bits of the GRH. The higher
      // 20 bytes are undefined.
      // According to IPV4 header format and ROCEV2 Annex A17.4.5.2
      // Last 2 bits might be used for ECN
      iphdr ipv4_hdr = ExtractIp4Header(remote.buffer.data());
      uint8_t actual_traffic_class = ipv4_hdr.tos;
      EXPECT_EQ(actual_traffic_class & 0xfc, traffic_class & 0xfc);
      return;
    }
    // According to IPV6 header format and ROCEV2 Annex A17.4.5.2
    // Last 2 bits might be used for ECN
    ibv_grh* grh = reinterpret_cast<ibv_grh*>(remote.buffer.data());
    // 4 bits version, 8 bits traffic class, 20 bits flow label.
    uint32_t version_tclass_flow = ntohl(grh->version_tclass_flow);
    uint8_t actual_traffic_class = version_tclass_flow >> 20 & 0xfc;
    EXPECT_EQ(actual_traffic_class & 0xfc, traffic_class & 0xfc);
  }
}

TEST_F(LoopbackUdQpTest, SendHopLimit) {
  constexpr int kPayloadLength = 1000;
  constexpr uint8_t hop_limit = 0xff;
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kPayloadLength + sizeof(ibv_grh);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length = kPayloadLength;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    // Set with customized hop limit.
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index, remote.port_attr.gid,
                               AhAttribute().set_hop_limit(hop_limit));
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    int ip_family = verbs_util::GetIpAddressType(remote.port_attr.gid);
    ASSERT_NE(ip_family, -1);

    if (ip_family == AF_INET) {
      iphdr ipv4_hdr = ExtractIp4Header(remote.buffer.data());
      uint8_t actual_ttl = ipv4_hdr.ttl;

      if (FLAGS_logtostderr)
        verbs_util::PrintIpHeader(&ipv4_hdr, ip_family);
      VLOG(1) << absl::StrCat("Configured hop limit: ", hop_limit);
      VLOG(1) << absl::StrCat("Actual ttl: ", actual_ttl);
      EXPECT_LE(actual_ttl, hop_limit);
      return;
    }
    ibv_grh* grh = reinterpret_cast<ibv_grh*>(remote.buffer.data());
    uint8_t actual_hop_limit = grh->hop_limit;

    if (FLAGS_logtostderr)
      verbs_util::PrintIpHeader(grh, ip_family);
    VLOG(1) << absl::StrCat("Configured hop limit: ", hop_limit);
    VLOG(1) << absl::StrCat("Actual hop limit: ", actual_hop_limit);
    EXPECT_LE(actual_hop_limit, hop_limit);
  }
}

TEST_F(LoopbackUdQpTest, SendFlowLabel) {
  constexpr int kPayloadLength = 1000;
  constexpr uint32_t flow_label = 0xabcde;
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kPayloadLength + sizeof(ibv_grh);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length = kPayloadLength;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    // Set with customized flow label.
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index, remote.port_attr.gid,
                               AhAttribute().set_flow_label(flow_label));
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    int ip_family = verbs_util::GetIpAddressType(remote.port_attr.gid);
    ASSERT_NE(ip_family, -1);

    if (ip_family == AF_INET6) {
      ibv_grh* grh = reinterpret_cast<ibv_grh*>(remote.buffer.data());
      // 4 bits version, 8 bits traffic class, 20 bits flow label.
      uint32_t version_tclass_flow = ntohl(grh->version_tclass_flow);
      uint32_t actual_flow_label = version_tclass_flow & 0xfffff;
      if (FLAGS_logtostderr)
        verbs_util::PrintIpHeader(grh, ip_family);
      EXPECT_EQ(actual_flow_label, flow_label);
    }
  }
}

TEST_F(LoopbackUdQpTest, SendWithTooSmallRecv) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());
  const uint32_t recv_length = local.buffer.span().size() / 2;
  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  rsge.length = recv_length + sizeof(ibv_grh);
  // Recv buffer is to small to fit the whole buffer.
  rsge.length -= 1;
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  lsge.length = kPayloadLength;
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  send.wr.ud.ah = ah;
  send.wr.ud.remote_qpn = remote.qp->qp_num;
  send.wr.ud.remote_qkey = kQKey;
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

TEST_F(LoopbackUdQpTest, BadSendLkey) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  rsge.length = kPayloadLength + sizeof(ibv_grh);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  lsge.length = kPayloadLength;
  // Garble the local sge key
  lsge.lkey = (lsge.lkey + 10) * 5;
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  send.wr.ud.ah = ah;
  send.wr.ud.remote_qpn = remote.qp->qp_num;
  send.wr.ud.remote_qkey = kQKey;
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(completion.opcode, IBV_WC_SEND);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);

  EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));
}

TEST_F(LoopbackUdQpTest, BadRecvLkey) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kPayloadLength + sizeof(ibv_grh);
    // Garble the remote sge key
    rsge.lkey = (rsge.lkey + 10) * 5;
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length = kPayloadLength;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index,
                               remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
  }
}

TEST_F(LoopbackUdQpTest, BadSendAddr) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.

  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  rsge.length = kPayloadLength + sizeof(ibv_grh);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  lsge.length = kPayloadLength;
  // Modify the local sge address to invalid address
  --lsge.addr;
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  send.wr.ud.ah = ah;
  send.wr.ud.remote_qpn = remote.qp->qp_num;
  send.wr.ud.remote_qkey = kQKey;
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(completion.opcode, IBV_WC_SEND);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);

  EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));
}

TEST_F(LoopbackUdQpTest, BadRecvAddr) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.

  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kPayloadLength + sizeof(ibv_grh);
    // Modify the remote sge address to invalid address
    --rsge.addr;
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length = kPayloadLength;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index,
                               remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
  }
}

TEST_F(LoopbackUdQpTest, SendInvalidAh) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  rsge.length = kPayloadLength + sizeof(ibv_grh);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);

  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  lsge.length = kPayloadLength;
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  HandleGarble garble(ah->handle);
  send.wr.ud.ah = ah;
  send.wr.ud.remote_qpn = remote.qp->qp_num;
  send.wr.ud.remote_qkey = kQKey;
  verbs_util::PostSend(local.qp, send);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  if (completion.status == IBV_WC_SUCCESS) {
    // Some provider will ignore the handle of ibv_ah and only examine its
    // PD and ibv_grh.
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
    ASSERT_OK_AND_ASSIGN(completion, verbs_util::WaitForCompletion(remote.cq));
    EXPECT_THAT(completion.status, AnyOf(IBV_WC_SUCCESS));
    EXPECT_EQ(IBV_WC_RECV, completion.opcode);
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    absl::Span<uint8_t> recv_payload =
        remote.buffer.subspan(sizeof(ibv_grh), kPayloadLength);
    EXPECT_THAT(recv_payload, Each(kLocalBufferContent));
  } else {
    // Otherwise packet won't be sent.
    EXPECT_EQ(completion.status, IBV_WC_LOC_QP_OP_ERR);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);

    EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));
    absl::Span<uint8_t> recv_payload =
        remote.buffer.subspan(sizeof(ibv_grh), kPayloadLength);
    EXPECT_THAT(recv_payload, Each(kRemoteBufferContent));
  }
}

TEST_F(LoopbackUdQpTest, SendInvalidQpn) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kPayloadLength + sizeof(ibv_grh);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length = kPayloadLength;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index, remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = 0xDEADBEEF;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));
    absl::Span<uint8_t> recv_payload =
        remote.buffer.subspan(sizeof(ibv_grh), kPayloadLength);
    EXPECT_THAT(recv_payload, Each(kRemoteBufferContent));
  }
}

TEST_F(LoopbackUdQpTest, SendInvalidQKey) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    rsge.length = kPayloadLength + sizeof(ibv_grh);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
    verbs_util::PostRecv(remote.qp, recv);
  }

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length = kPayloadLength;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index, remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    // MSB of remote_qkey must be unset for SR's QKey to be effective.
    send.wr.ud.remote_qkey = 0xDEADBEEF & 0x7FFFFFF;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote.cq));
    absl::Span<uint8_t> recv_payload =
        remote.buffer.subspan(sizeof(ibv_grh), kPayloadLength);
    EXPECT_THAT(recv_payload, Each(kRemoteBufferContent));
  }
}

TEST_F(LoopbackUdQpTest, SendRecvBatchedWr) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote),
                       CreateUdClientsPair(/*pages=*/128));

  uint32_t send_queue_size = verbs_util::GetQpCap(local.qp).max_send_wr;
  uint32_t recv_queue_size = verbs_util::GetQpCap(remote.qp).max_recv_wr;
  uint32_t batch_size = std::min(send_queue_size, recv_queue_size);
  uint32_t active_mtu =
      verbs_util::VerbsMtuToInt(local.port_attr.attr.active_mtu);
  // Choose rsge_size as the min between active_mtu and allocable buffer
  // for each sge after equal division.
  uint32_t rsge_size = std::min(active_mtu,
                                (uint32_t)local.buffer.size() / batch_size);
  uint32_t lsge_size = rsge_size - sizeof(ibv_grh); // Additional 40B for rsge is for the GRH

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    std::vector<ibv_sge> rsge(batch_size);
    std::vector<ibv_recv_wr> recv_batch(batch_size);
    // Address in each sge points to buffer of equal size
    // Offset for each buffer = wr_id * size, here wr_id = i
    // length of each buffer = size
    // rsge[i] is used to create WR recv_batch[i]
    // Similar batching is done for send WRs
    for (uint32_t i = 0; i < batch_size; ++i) {
      rsge[i] = verbs_util::CreateSge(remote.buffer.subspan(
          /*offset=*/i*rsge_size, /*size=*/rsge_size), remote.mr);
      recv_batch[i] = verbs_util::CreateRecvWr(/*wr_id=*/i, &rsge[i],
                                               /*num_sge=*/1);
      recv_batch[i].next = (i != batch_size-1) ? &recv_batch[i + 1] : nullptr;
    }
    verbs_util::PostRecv(remote.qp, recv_batch[0]);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    std::vector<ibv_sge> lsge(batch_size);
    std::vector<ibv_send_wr> send_batch(batch_size);
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index,
                               remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    for (uint32_t i = 0; i < batch_size; ++i) {
      lsge[i] = verbs_util::CreateSge(local.buffer.subspan(
          /*offset=*/i*lsge_size, /*size=*/lsge_size), local.mr);
      send_batch[i] = verbs_util::CreateSendWr(/*wr_id=*/i, &lsge[i],
                                               /*num_sge=*/1);
      send_batch[i].wr.ud.ah = ah;
      send_batch[i].wr.ud.remote_qpn = remote.qp->qp_num;
      send_batch[i].wr.ud.remote_qkey = kQKey;
      send_batch[i].next = (i != batch_size-1) ? &send_batch[i + 1] : nullptr;
    }
    verbs_util::PostSend(local.qp, send_batch[0]);

    uint32_t send_completions = 0;
    for (uint32_t i=0; i < batch_size; ++i) {
      ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                           verbs_util::WaitForCompletion(local.cq));
      EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
      EXPECT_EQ(completion.qp_num, local.qp->qp_num);
      EXPECT_EQ(completion.wr_id, send_completions++);
    }
    EXPECT_EQ(send_completions, batch_size);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    // Wait for few seconds
    if (verbs_util::peer_mode()) {
      absl::SleepFor(absl::Seconds(2));
    }
    uint32_t recv_completions = 0;
    for (uint32_t i=0; i < batch_size; ++i) {
      ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                           verbs_util::WaitForCompletion(remote.cq));
      EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
      EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
      EXPECT_EQ(completion.wr_id, recv_completions++);
    }
    EXPECT_EQ(recv_completions, batch_size);
  }
}

// Read not supported on UD.
TEST_F(LoopbackUdQpTest, Read) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  read.wr.ud.ah = ah;
  read.wr.ud.remote_qkey = kQKey;
  read.wr.ud.remote_qpn = remote.qp->qp_num;
  verbs_util::PostSend(local.qp, read);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_QP_OP_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

// Tests polling multiple CQE in a single call.
TEST_F(LoopbackUdQpTest, PollMultipleCqe) {
  Client local, remote;
  ibv_send_wr send;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());
  static constexpr int kNumCompletions = 5;
  static constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge recv_sge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
    recv_sge.length = kPayloadLength + sizeof(ibv_grh);
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, &recv_sge, /*num_sge=*/1);
    for (int i = 0; i < kNumCompletions; ++i) {
      verbs_util::PostRecv(remote.qp, recv);
    }
  }

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge send_sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    send_sge.length = kPayloadLength;
    send = verbs_util::CreateSendWr(/*wr_id=*/1, &send_sge, /*num_sge=*/1);
    ibv_ah* ah = ibv_.CreateLoopbackAh(local.pd, remote.port_attr);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    for (int i = 0; i < kNumCompletions; ++i) {
      verbs_util::PostSend(local.qp, send);
    }
  }

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    // Wait for recv completions.
    for (int i = 0; i < kNumCompletions; ++i) {
      ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                           verbs_util::WaitForCompletion(remote.cq));
      EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
      EXPECT_EQ(completion.opcode, IBV_WC_RECV);
      EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    }
  }

  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    int count = 0;
    ibv_wc result[kNumCompletions + 1];
    // This is inherantly racy, just because recv posted doesn't mean send
    // completion is there yet.  Allow up to 1 second for completions to arrive.
    absl::Time stop = absl::Now() + absl::Seconds(1);
    while (count != kNumCompletions && absl::Now() < stop) {
      count += ibv_poll_cq(local.cq, kNumCompletions + 1, &result[count]);
    }
    ASSERT_EQ(kNumCompletions, count)
        << "Missing completions see comment above about potential race.";
    // Spot check last completion.
    ibv_wc& completion = result[kNumCompletions - 1];
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(send.wr_id, completion.wr_id);
  }
}

// Write not supported on Ud.
TEST_F(LoopbackUdQpTest, Write) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());
  ibv_sge sge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  ibv_send_wr write = verbs_util::CreateWriteWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey);
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  write.wr.ud.ah = ah;
  write.wr.ud.remote_qkey = kQKey;
  write.wr.ud.remote_qpn = remote.qp->qp_num;
  verbs_util::PostSend(local.qp, write);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_QP_OP_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

// FetchAndAdd not supported on UD.
TEST_F(LoopbackUdQpTest, FetchAdd) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  // The local SGE will be used to store the value before the update.
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  ibv_send_wr fetch_add = verbs_util::CreateFetchAddWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      0);
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  fetch_add.wr.ud.ah = ah;
  fetch_add.wr.ud.remote_qkey = kQKey;
  fetch_add.wr.ud.remote_qpn = remote.qp->qp_num;
  verbs_util::PostSend(local.qp, fetch_add);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_QP_OP_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

// CompareAndSwap not supported on UD.
TEST_F(LoopbackUdQpTest, CompareSwap) {
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());
  *reinterpret_cast<uint64_t*>(local.buffer.data()) = 1;
  *reinterpret_cast<uint64_t*>(remote.buffer.data()) = 2;
  ibv_sge sge = verbs_util::CreateSge(local.buffer.subspan(0, 8), local.mr);
  sge.length = 8;
  ibv_send_wr cmp_swp = verbs_util::CreateCompSwapWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, remote.buffer.data(), remote.mr->rkey,
      2, 3);
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  cmp_swp.wr.ud.ah = ah;
  cmp_swp.wr.ud.remote_qkey = kQKey;
  cmp_swp.wr.ud.remote_qpn = remote.qp->qp_num;
  verbs_util::PostSend(local.qp, cmp_swp);

  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local.cq));
  EXPECT_EQ(completion.status, IBV_WC_LOC_QP_OP_ERR);
  EXPECT_EQ(completion.qp_num, local.qp->qp_num);
  EXPECT_EQ(completion.wr_id, 1);
}

TEST_F(LoopbackUdQpTest, RecvGrhSplitSgl) {
  constexpr int kPayloadLength = 1000;  // Sub-MTU length for UD.
  constexpr int kGrhHeaderBytes = sizeof(ibv_grh);
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair(
      /*pages=*/1,QpInitAttribute().set_max_recv_sge(2)));

  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ibv_sge rsgl[2];
    // First 30 bytes of GRH will be accommodated in rsgl[0]
    // Reamining 10 bytes along with payload in rsgl[1]
    rsgl[0] = verbs_util::CreateSge(remote.buffer.subspan(0,30),remote.mr);
    rsgl[1] = verbs_util::CreateSge(remote.buffer.subspan(30), remote.mr);
    rsgl[1].length = kPayloadLength + 10;
    ibv_recv_wr recv =
        verbs_util::CreateRecvWr(/*wr_id=*/0, rsgl, /*num_sge=*/2);
    verbs_util::PostRecv(remote.qp, recv);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_client()) {
    ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
    lsge.length = kPayloadLength;
    ibv_send_wr send =
        verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
    ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                               local.port_attr.gid_index,
                               remote.port_attr.gid);
    ASSERT_THAT(ah, NotNull());
    send.wr.ud.ah = ah;
    send.wr.ud.remote_qpn = remote.qp->qp_num;
    send.wr.ud.remote_qkey = kQKey;
    verbs_util::PostSend(local.qp, send);

    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(local.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_SEND);
    EXPECT_EQ(completion.qp_num, local.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 1);
  }
  if (!verbs_util::peer_mode() || verbs_util::is_server()) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(remote.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.opcode, IBV_WC_RECV);
    EXPECT_EQ(completion.byte_len, (kGrhHeaderBytes + kPayloadLength));
    EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
    EXPECT_EQ(completion.wr_id, 0);
    absl::Span<uint8_t> recv_payload =
        remote.buffer.subspan(kGrhHeaderBytes, kPayloadLength);
    EXPECT_THAT(recv_payload, Each(kLocalBufferContent));

    int ip_family = verbs_util::GetIpAddressType(remote.port_attr.gid);
    ASSERT_NE(ip_family, -1);
    if (FLAGS_logtostderr) {
      if (ip_family == AF_INET) {
        verbs_util::PrintIpHeader(remote.buffer.data() + 20, ip_family);
      }
      else{
        verbs_util::PrintIpHeader(remote.buffer.data(), ip_family);
      }
    }
  }
}

class AdvancedLoopbackTest : public RdmaVerbsFixture {
 public:
  struct BasicSetup {
    RdmaMemBlock buffer;
    ibv_context* context;
    PortAttribute port_attr;
    ibv_pd* pd;
    ibv_mr* mr;
  };

  absl::StatusOr<BasicSetup> CreateBasicSetup() {
    BasicSetup setup;
    setup.buffer = ibv_.AllocBuffer(/*pages=*/1);
    ASSIGN_OR_RETURN(setup.context, ibv_.OpenDevice());
    setup.port_attr = ibv_.GetPortAttribute(setup.context);
    setup.pd = ibv_.AllocPd(setup.context);
    if (!setup.pd) {
      return absl::InternalError("Failed to allocate pd.");
    }
    setup.mr = ibv_.RegMr(setup.pd, setup.buffer);
    if (!setup.mr) {
      return absl::InternalError("Failed to register mr.");
    }
    return setup;
  }
};

// TODO(author2): Use LoopbackFixture and CreateClient.
TEST_F(AdvancedLoopbackTest, RcSendToUd) {
  constexpr size_t kPayloadLength = 1000;
  constexpr int kQKey = 200;
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_cq* local_cq = ibv_.CreateCq(setup.context);
  ASSERT_THAT(local_cq, NotNull());
  ibv_cq* remote_cq = ibv_.CreateCq(setup.context);
  ASSERT_THAT(remote_cq, NotNull());
  ibv_qp* local_qp = ibv_.CreateQp(setup.pd, local_cq, IBV_QPT_RC);
  ASSERT_THAT(local_qp, NotNull());
  ibv_qp* remote_qp = ibv_.CreateQp(setup.pd, remote_cq, IBV_QPT_UD);
  ASSERT_THAT(remote_qp, NotNull());
  ASSERT_OK(ibv_.ModifyRcQpResetToRts(
      local_qp, setup.port_attr, setup.port_attr.gid, remote_qp->qp_num,
      QpAttribute().set_timeout(absl::Seconds(1))));
  ASSERT_OK(ibv_.ModifyUdQpResetToRts(remote_qp, kQKey));

  ibv_sge rsge = verbs_util::CreateSge(
      setup.buffer.span().subspan(0, kPayloadLength + sizeof(ibv_grh)),
      setup.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote_qp, recv);
  ibv_sge lsge = verbs_util::CreateSge(
      setup.buffer.span().subspan(0, kPayloadLength), setup.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/0, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(local_qp, send);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local_cq));
  EXPECT_EQ(completion.status, IBV_WC_RETRY_EXC_ERR);
  EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote_cq));
}

TEST_F(AdvancedLoopbackTest, UdSendToRc) {
  constexpr size_t kPayloadLength = 1000;
  constexpr int kQKey = 200;
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_cq* local_cq = ibv_.CreateCq(setup.context);
  ASSERT_THAT(local_cq, NotNull());
  ibv_cq* remote_cq = ibv_.CreateCq(setup.context);
  ASSERT_THAT(remote_cq, NotNull());
  ibv_qp* local_qp = ibv_.CreateQp(setup.pd, local_cq, IBV_QPT_UD);
  ASSERT_THAT(local_qp, NotNull());
  ibv_qp* remote_qp = ibv_.CreateQp(setup.pd, remote_cq, IBV_QPT_RC);
  ASSERT_THAT(remote_qp, NotNull());
  ASSERT_OK(ibv_.ModifyUdQpResetToRts(local_qp, kQKey));
  ASSERT_OK(ibv_.ModifyRcQpResetToRts(
      remote_qp, setup.port_attr, setup.port_attr.gid, local_qp->qp_num,
      QpAttribute().set_timeout(absl::Seconds(1))));
  ibv_ah* ah = ibv_.CreateLoopbackAh(setup.pd, setup.port_attr);
  ASSERT_THAT(ah, NotNull());

  ibv_sge rsge = verbs_util::CreateSge(
      setup.buffer.span().subspan(0, kPayloadLength + sizeof(ibv_grh)),
      setup.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote_qp, recv);
  ibv_sge lsge = verbs_util::CreateSge(
      setup.buffer.span().subspan(0, kPayloadLength), setup.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/0, &lsge, /*num_sge=*/1);
  send.wr.ud.ah = ah;
  send.wr.ud.remote_qpn = remote_qp->qp_num;
  send.wr.ud.remote_qkey = kQKey;
  verbs_util::PostSend(local_qp, send);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(local_cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_TRUE(verbs_util::ExpectNoCompletion(remote_cq));
}

class UdQpRecoverTest
    : public LoopbackUdQpTest,
      public testing::WithParamInterface<verbs_util::IbvOperations> {
  protected:
   int ErrUdQp(ibv_qp* qp) const {
     // Modify QP State to ERR state
     ibv_qp_attr attr;
     attr.qp_state = IBV_QPS_ERR;
     ibv_qp_attr_mask attr_mask = IBV_QP_STATE;
     return ibv_modify_qp(qp, &attr, attr_mask);
   }

   void MoveQpErrorState(Client &local, Client &remote) {
     // Moves UD QP pair to error state.
     constexpr int kPayloadLength = 1000;
     ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
     rsge.length = kPayloadLength + sizeof(ibv_grh);
     rsge.lkey = (rsge.lkey + 10) * 5;
     ibv_recv_wr recv =
           verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
     verbs_util::PostRecv(remote.qp, recv);
     ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
     lsge.length = kPayloadLength;
     ibv_send_wr send =
         verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
     ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                                local.port_attr.gid_index,
                                remote.port_attr.gid);
     ASSERT_THAT(ah, NotNull());
     send.wr.ud.ah = ah;
     send.wr.ud.remote_qpn = remote.qp->qp_num;
     send.wr.ud.remote_qkey = kQKey;
     verbs_util::PostSend(local.qp, send);
     ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                          verbs_util::WaitForCompletion(local.cq));
     EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
     EXPECT_EQ(completion.qp_num, local.qp->qp_num);
     EXPECT_EQ(completion.wr_id, 1);
     ASSERT_OK_AND_ASSIGN(completion,
                          verbs_util::WaitForCompletion(remote.cq));
     EXPECT_EQ(completion.status, IBV_WC_LOC_PROT_ERR);
     EXPECT_EQ(completion.qp_num, remote.qp->qp_num);
     EXPECT_EQ(completion.wr_id, 0);
     EXPECT_EQ(ErrUdQp(local.qp), 0);
     // Ensure both QPs have moved to error state
     EXPECT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_ERR);
     EXPECT_EQ(verbs_util::GetQpState(local.qp), IBV_QPS_ERR);
   }
   void RecoverQpErrorState(Client &local, Client &remote) {
     // Recovers UD QP pair from Error state and moves it to RTS state.
     ibv_qp_attr attr;
     attr.qp_state = IBV_QPS_RESET;
     ibv_qp_attr_mask attr_mask = IBV_QP_STATE;
     ASSERT_EQ(ibv_modify_qp(local.qp, &attr, attr_mask), 0);
     ASSERT_EQ(ibv_modify_qp(remote.qp, &attr, attr_mask), 0);
     ASSERT_OK(ibv_.ModifyUdQpResetToRts(local.qp, kQKey));
     ASSERT_OK(ibv_.ModifyUdQpResetToRts(remote.qp, kQKey));
     ASSERT_EQ(verbs_util::GetQpState(local.qp),  IBV_QPS_RTS);
     ASSERT_EQ(verbs_util::GetQpState(remote.qp), IBV_QPS_RTS);
   }
};

TEST_P(UdQpRecoverTest, UdQpRecoverTests) {
  int kPayloadLength = 1000;  // Sub-MTU length for UD.
  constexpr int kGrhHeaderBytes = sizeof(ibv_grh);
  const uint32_t kImm = 0xBADDCAFE;
  // data_src is to store the address of inline data
  std::unique_ptr<std::vector<uint8_t>> data_src;
  Client local, remote;
  ASSERT_OK_AND_ASSIGN(std::tie(local, remote), CreateUdClientsPair());
  // Move QPs to error state.
  MoveQpErrorState(local, remote);
  // Recover QPs from error state
  RecoverQpErrorState(local, remote);
  verbs_util::IbvOperations operation = GetParam();
  ibv_sge rsge = verbs_util::CreateSge(remote.buffer.span(), remote.mr);
  rsge.length = kPayloadLength + kGrhHeaderBytes;
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/0, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(remote.qp, recv);
  ibv_sge lsge = verbs_util::CreateSge(local.buffer.span(), local.mr);
  if (operation == verbs_util::IbvOperations::SendInline) {
    kPayloadLength = verbs_util::GetQpCap(local.qp).max_inline_data;
  }
  lsge.length = kPayloadLength;
  ibv_send_wr send;
  switch (operation) {
    case verbs_util::IbvOperations::Send:
      send = verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
      break;
    case verbs_util::IbvOperations::SendInline:
      // A vector which is not registered to pd or mr
      data_src = std::make_unique<std::vector<uint8_t>>(kPayloadLength);
      // Modify lsge to send Inline data with addr of the vector
      std::fill(data_src->begin(), data_src->end(), 'a');
      lsge.addr = reinterpret_cast<uint64_t>(data_src->data());
      lsge.lkey = 0xDEADBEEF;  // random bad keys
      send = verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
      send.send_flags |= IBV_SEND_INLINE;
      break;
    case verbs_util::IbvOperations::SendWithImm:
      send = verbs_util::CreateSendWithImmWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
      send.imm_data = kImm;
      break;
  }
  ibv_ah* ah = ibv_.CreateAh(local.pd, local.port_attr.port,
                             local.port_attr.gid_index, remote.port_attr.gid);
  ASSERT_THAT(ah, NotNull());
  send.wr.ud.ah = ah;
  send.wr.ud.remote_qpn = remote.qp->qp_num;
  send.wr.ud.remote_qkey = kQKey;
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
  EXPECT_EQ(completion.byte_len, (kGrhHeaderBytes + kPayloadLength));
  absl::Span<uint8_t> recv_payload =
      remote.buffer.subspan(sizeof(ibv_grh), kPayloadLength);
  EXPECT_THAT(recv_payload, Each(kLocalBufferContent));
  if (operation == verbs_util::IbvOperations::SendWithImm) {
    EXPECT_NE(completion.wc_flags & IBV_WC_WITH_IMM, 0);
    EXPECT_EQ(kImm, completion.imm_data);
  }
}
INSTANTIATE_TEST_SUITE_P(
    UdQpRecoverTest, UdQpRecoverTest,
    testing::Values(verbs_util::IbvOperations::Send,
                    verbs_util::IbvOperations::SendWithImm,
                    verbs_util::IbvOperations::SendInline),
    [](const testing::TestParamInfo<UdQpRecoverTest::ParamType>& info) {
    std::string name = [info]() {
      switch (info.param) {
        case verbs_util::IbvOperations::Send: return "Send";
        case verbs_util::IbvOperations::SendInline: return "SendInline";
        case verbs_util::IbvOperations::SendWithImm: return "SendWithImm";
        default: return "Unknown";
      }
    }();
      return name;
    });
}  // namespace
}  // namespace rdma_unit_test
