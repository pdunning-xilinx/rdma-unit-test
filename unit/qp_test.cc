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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "infiniband/verbs.h"
#include "internal/handle_garble.h"
#include "internal/verbs_attribute.h"
#include "public/introspection.h"
#include "public/rdma_memblock.h"
#include "public/status_matchers.h"
#include "public/verbs_helper_suite.h"
#include "public/verbs_util.h"
#include "unit/loopback_fixture.h"
#include "unit/rdma_verbs_fixture.h"

namespace rdma_unit_test {

using ::testing::AnyOf;
using ::testing::IsNull;
using ::testing::NotNull;

class QpTest : public RdmaVerbsFixture {
 public:
  int InitRcQP(ibv_qp* qp) const {
    PortAttribute port_attr = ibv_.GetPortAttribute(qp->context);
    ibv_qp_attr mod_init = QpAttribute().GetRcResetToInitAttr(port_attr.port);
    int mask = QpAttribute().GetRcResetToInitMask();
    return ibv_modify_qp(qp, &mod_init, mask);
  }

  int InitUdQP(ibv_qp* qp, uint32_t qkey) const {
    PortAttribute port_attr = ibv_.GetPortAttribute(qp->context);
    ibv_qp_attr mod_init = {};
    mod_init.qp_state = IBV_QPS_INIT;
    mod_init.pkey_index = 0;
    mod_init.port_num = port_attr.port;
    mod_init.qkey = qkey;
    constexpr int kInitMask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY;
    return ibv_modify_qp(qp, &mod_init, kInitMask);
  }

 protected:
  static constexpr ibv_qp_cap kBasicQpCap = {.max_send_wr = 10,
                                             .max_recv_wr = 1,
                                             .max_send_sge = 1,
                                             .max_recv_sge = 1,
                                             .max_inline_data = 36};

  struct BasicSetup {
    ibv_context* context;
    PortAttribute port_attr;
    ibv_pd* pd;
    ibv_cq* cq;
    ibv_qp_init_attr basic_attr;
  };

  absl::StatusOr<BasicSetup> CreateBasicSetup() {
    BasicSetup setup;
    ASSIGN_OR_RETURN(setup.context, ibv_.OpenDevice());
    setup.port_attr = ibv_.GetPortAttribute(setup.context);
    setup.pd = ibv_.AllocPd(setup.context);
    if (!setup.pd) {
      return absl::InternalError("Failed to allocate pd.");
    }
    setup.cq = ibv_.CreateCq(setup.context);
    if (!setup.cq) {
      return absl::InternalError("Failed to create cq.");
    }
    setup.basic_attr =
        QpInitAttribute().GetAttribute(setup.cq, setup.cq, IBV_QPT_RC);
    return setup;
  }

  // Takes a bit-mask of arbitrary qp attributes and returns a vector with
  // each attribute that is set or'd with IBV_QP_STATE so that it can be
  // directly used in a ibv_modify_qp invocation.
  std::vector<int> CreateMaskCombinations(int required_attributes) {
    std::vector<int> masks;
    // Ensure the IBV_QP_STATE bit is set.  This will result in the first
    // vector element being no attributes set.
    required_attributes |= IBV_QP_STATE;
    // Traverse the bitmask from bit0..binN and find each set bit which
    // corresponds to an attribute that is set and add it the vector.
    for (unsigned int bit = 0; bit < 8 * sizeof(required_attributes); ++bit) {
      if ((1 << bit) & required_attributes)
        masks.push_back(1 << bit | IBV_QP_STATE);
    }
    return masks;
  }
};

TEST_F(QpTest, Create) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.extension().CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());
  EXPECT_EQ(qp->pd, setup.pd);
  EXPECT_EQ(qp->send_cq, setup.basic_attr.send_cq);
  EXPECT_EQ(qp->recv_cq, setup.basic_attr.recv_cq);
  EXPECT_EQ(qp->srq, setup.basic_attr.srq);
  EXPECT_EQ(qp->qp_type, setup.basic_attr.qp_type);
  EXPECT_EQ(qp->state, IBV_QPS_RESET);
  EXPECT_EQ(ibv_destroy_qp(qp), 0);
}

TEST_F(QpTest, CreateWithInlineMax) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  // Some arbitrary values. max_inline_data is rarely over 1k.
  for (uint32_t max_inline : {36, 96, 216, 580, 1000}) {
    ibv_qp_init_attr attr =
        QpInitAttribute().GetAttribute(setup.cq, setup.cq, IBV_QPT_RC);
    attr.cap.max_inline_data = max_inline;
    ibv_qp* qp = ibv_.CreateQp(setup.pd, attr);
    if (qp != nullptr) {
      EXPECT_GE(attr.cap.max_inline_data, max_inline);
      LOG(INFO) << "Try creating QP with max_inline_data " << max_inline
                << ", get actual value " << attr.cap.max_inline_data << ".";
    } else {
      LOG(INFO) << "Creation failed with max_inline_data = " << max_inline
                << ".";
    }
  }
}

TEST_F(QpTest, CreateWithInvalidSendCq) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_cq* cq = ibv_.CreateCq(setup.context);
  ASSERT_THAT(cq, NotNull());
  HandleGarble garble(cq->handle);
  ibv_qp_init_attr attr =
      QpInitAttribute().GetAttribute(setup.cq, setup.cq, IBV_QPT_RC);
  attr.send_cq = cq;
  EXPECT_THAT(ibv_.CreateQp(setup.pd, attr), IsNull());
}

TEST_F(QpTest, CreateWithInvalidRecvCq) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_cq* cq = ibv_.CreateCq(setup.context);
  ASSERT_THAT(cq, NotNull());
  HandleGarble garble(cq->handle);
  ibv_qp_init_attr attr =
      QpInitAttribute().GetAttribute(setup.cq, setup.cq, IBV_QPT_RC);
  attr.recv_cq = cq;
  EXPECT_THAT(ibv_.CreateQp(setup.pd, attr), IsNull());
}

TEST_F(QpTest, CreateWithInvalidSrq) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_srq* srq = ibv_.CreateSrq(setup.pd);
  ASSERT_THAT(srq, NotNull());
  HandleGarble garble(srq->handle);
  ibv_qp_init_attr attr =
      QpInitAttribute().GetAttribute(setup.cq, setup.cq, IBV_QPT_RC);
  attr.srq = srq;
  EXPECT_THAT(ibv_.CreateQp(setup.pd, attr), IsNull());
}

TEST_F(QpTest, UnknownType) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp_init_attr attr = setup.basic_attr;
  attr.qp_type = static_cast<ibv_qp_type>(0xf0);
  ibv_qp* qp = ibv_.CreateQp(setup.pd, attr);
  EXPECT_THAT(qp, IsNull());
}

TEST_F(QpTest, CreateUd) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp_init_attr attr = setup.basic_attr;
  attr.qp_type = IBV_QPT_UD;
  EXPECT_THAT(ibv_.CreateQp(setup.pd, attr), NotNull());
}

TEST_F(QpTest, ZeroSendWr) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  setup.basic_attr.cap.max_send_wr = 0;
  EXPECT_THAT(ibv_.CreateQp(setup.pd, setup.basic_attr), NotNull());
}

TEST_F(QpTest, QueueRefs) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.cq);
  ASSERT_THAT(qp, NotNull());
  // Rejected because QP holds ref.
  EXPECT_EQ(ibv_destroy_cq(setup.cq), EBUSY);
}

TEST_F(QpTest, MaxQpWr) {
  const ibv_device_attr& device_attr = Introspection().device_attr();
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());

  ibv_qp_init_attr attr = setup.basic_attr;
  attr.cap = kBasicQpCap;
  attr.cap.max_send_wr = device_attr.max_qp_wr;
  EXPECT_THAT(ibv_.CreateQp(setup.pd, attr), NotNull());
  attr.cap.max_send_wr = device_attr.max_qp_wr + 1;
  ibv_qp* qp = ibv_.CreateQp(setup.pd, attr);
  if (qp != nullptr) {
    LOG(WARNING) << "Nonstandard behavior: Creating a QP with max send WR "
                    "greater than device limit is successful.";
  }

  attr.cap = kBasicQpCap;
  attr.cap.max_recv_wr = device_attr.max_qp_wr;
  EXPECT_THAT(ibv_.CreateQp(setup.pd, attr), NotNull());
  attr.cap.max_recv_wr = device_attr.max_qp_wr + 1;
  qp = ibv_.CreateQp(setup.pd, attr);
  if (qp != nullptr) {
    LOG(WARNING)
        << "Nonstandard behavior: Creating a QP with max recv WR greater "
           "than device limit is successful.";
  }
}

TEST_F(QpTest, MaxSge) {
  const ibv_device_attr& device_attr = Introspection().device_attr();
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp_init_attr attr = setup.basic_attr;

  attr.cap = kBasicQpCap;
  attr.cap.max_send_sge = device_attr.max_sge;
  EXPECT_THAT(ibv_.CreateQp(setup.pd, attr), NotNull());
  attr.cap.max_send_sge = device_attr.max_sge + 1;
  ibv_qp* qp = ibv_.CreateQp(setup.pd, attr);
  if (qp != nullptr) {
    EXPECT_LE(attr.cap.max_send_sge, device_attr.max_sge);
  }

  attr.cap = kBasicQpCap;
  attr.cap.max_recv_sge = device_attr.max_sge;
  EXPECT_THAT(ibv_.CreateQp(setup.pd, attr), NotNull());
  attr.cap.max_recv_sge = device_attr.max_sge + 1;
  qp = ibv_.CreateQp(setup.pd, attr);
  if (qp != nullptr) {
    EXPECT_LE(attr.cap.max_recv_sge, device_attr.max_sge);
  }
}

// TODO(author1): Cross context objects.
// TODO(author1): List of WQEs which is too large.
// TODO(author1): Single WQEs which is too large.
// TODO(author1): List of WQEs which partially fits (make sure to check
// where it stops)

TEST_F(QpTest, Modify) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());

  // Reset -> Init.
  ASSERT_EQ(InitRcQP(qp), 0);
  // Init -> Ready to receive.
  ibv_qp_attr mod_rtr = QpAttribute().GetRcInitToRtrAttr(
      setup.port_attr.port, setup.port_attr.gid_index, setup.port_attr.gid,
      qp->qp_num);
  int mask_rtr = QpAttribute().GetRcInitToRtrMask();
  ASSERT_EQ(ibv_modify_qp(qp, &mod_rtr, mask_rtr), 0);
  // Ready to receive -> Ready to send.
  ibv_qp_attr mod_rts = QpAttribute().GetRcRtrToRtsAttr();
  int mask_rts = QpAttribute().GetRcRtrToRtsMask();
  EXPECT_EQ(ibv_modify_qp(qp, &mod_rts, mask_rts), 0);
}

TEST_F(QpTest, ModifyInvalidResetToRtrTransition) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());

  ibv_qp_attr mod_rtr = QpAttribute().GetRcInitToRtrAttr(
      setup.port_attr.port, setup.port_attr.gid_index, setup.port_attr.gid,
      qp->qp_num);
  int mask_rtr = QpAttribute().GetRcInitToRtrMask();
  EXPECT_NE(ibv_modify_qp(qp, &mod_rtr, mask_rtr), 0);
}

TEST_F(QpTest, ModifyInvalidInitToRtsTransition) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());
  // Reset -> Init.
  ASSERT_EQ(InitRcQP(qp), 0);
  // Init -> Ready to send.
  ibv_qp_attr mod_rts = QpAttribute().GetRcRtrToRtsAttr();
  int mask_rts = QpAttribute().GetRcRtrToRtsMask();
  EXPECT_NE(ibv_modify_qp(qp, &mod_rts, mask_rts), 0);
}

TEST_F(QpTest, ModifyInvalidRtsToRtrTransition) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());
  // Reset -> Init.
  ASSERT_EQ(InitRcQP(qp), 0);
  // Init -> Ready to receive.
  ibv_qp_attr mod_rtr = QpAttribute().GetRcInitToRtrAttr(
      setup.port_attr.port, setup.port_attr.gid_index, setup.port_attr.gid,
      qp->qp_num);
  int mask_rtr = QpAttribute().GetRcInitToRtrMask();
  ASSERT_EQ(ibv_modify_qp(qp, &mod_rtr, mask_rtr), 0);
  // Ready to receive -> Ready to send.
  ibv_qp_attr mod_rts = QpAttribute().GetRcRtrToRtsAttr();
  int mask_rts = QpAttribute().GetRcRtrToRtsMask();
  ASSERT_EQ(ibv_modify_qp(qp, &mod_rts, mask_rts), 0);
  // Ready to send -> Ready to receive.
  EXPECT_NE(ibv_modify_qp(qp, &mod_rtr, mask_rtr), 0);
}

TEST_F(QpTest, ModifyBadResetToInitStateTransition) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());

  ibv_qp_attr mod_init =
      QpAttribute().GetRcResetToInitAttr(setup.port_attr.port);
  int mod_init_mask = QpAttribute().GetRcResetToInitMask();
  for (auto mask : CreateMaskCombinations(mod_init_mask)) {
    EXPECT_NE(ibv_modify_qp(qp, &mod_init, mask), 0);
  }
  EXPECT_EQ(ibv_modify_qp(qp, &mod_init, mod_init_mask), 0);
}

TEST_F(QpTest, ModifyBadInitToRtrStateTransition) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());

  // Reset -> Init.
  ASSERT_EQ(InitRcQP(qp), 0);
  // Init -> Ready to receive
  ibv_qp_attr mod_rtr = QpAttribute().GetRcInitToRtrAttr(
      setup.port_attr.port, setup.port_attr.gid_index, setup.port_attr.gid,
      qp->qp_num);
  int mask_rtr = QpAttribute().GetRcInitToRtrMask();
  for (auto mask : CreateMaskCombinations(mask_rtr)) {
    EXPECT_NE(ibv_modify_qp(qp, &mod_rtr, mask), 0);
  }
  EXPECT_EQ(ibv_modify_qp(qp, &mod_rtr, mask_rtr), 0);
}

TEST_F(QpTest, ModifyBadRtrToRtsStateTransition) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());

  // Reset -> Init.
  ASSERT_EQ(InitRcQP(qp), 0);
  // Init -> Ready to receive.
  ibv_qp_attr mod_rtr = QpAttribute().GetRcInitToRtrAttr(
      setup.port_attr.port, setup.port_attr.gid_index, setup.port_attr.gid,
      qp->qp_num);
  int mask_rtr = QpAttribute().GetRcInitToRtrMask();
  ASSERT_EQ(ibv_modify_qp(qp, &mod_rtr, mask_rtr), 0);

  // Ready to receive -> Ready to send.
  ibv_qp_attr mod_rts = QpAttribute().GetRcRtrToRtsAttr();
  int mask_rts = QpAttribute().GetRcRtrToRtsMask();
  for (auto mask : CreateMaskCombinations(mask_rts)) {
    int result = ibv_modify_qp(qp, &mod_rts, mask);
    EXPECT_NE(result, 0);
  }
  EXPECT_EQ(ibv_modify_qp(qp, &mod_rts, mask_rts), 0);
}

TEST_F(QpTest, ModifyBadRtsToSqdStateTransition) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());
  // Reset -> Init.
  ASSERT_EQ(InitRcQP(qp), 0);
  // Init -> RTR
  ibv_qp_attr mod_rtr = QpAttribute().GetRcInitToRtrAttr(
      setup.port_attr.port, setup.port_attr.gid_index, setup.port_attr.gid,
      qp->qp_num);
  int mask_rtr = QpAttribute().GetRcInitToRtrMask();
  ASSERT_EQ(ibv_modify_qp(qp, &mod_rtr, mask_rtr), 0);
  // RTR -> RTS
  ibv_qp_attr mod_rts = QpAttribute().GetRcRtrToRtsAttr();
  int mask_rts = QpAttribute().GetRcRtrToRtsMask();
  ASSERT_EQ(ibv_modify_qp(qp, &mod_rts, mask_rts), 0);
  // RTS -> SQD with invalid mask
  ibv_qp_attr mod_sqd = QpAttribute().GetRcRtsToSqdAttr();
  int mask_sqd = QpAttribute().GetRcRtsToSqdMask();
  for (auto mask : CreateMaskCombinations(mask_rts)) {
    if (mask != mask_sqd) {
      EXPECT_NE(ibv_modify_qp(qp, &mod_sqd, mask), 0);
    }
  }
  // RTS -> SQD with valid mask
  EXPECT_EQ(ibv_modify_qp(qp, &mod_sqd, mask_sqd), 0);
}

TEST_F(QpTest, ModifyBadSqdToRtsStateTransition) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());
  // Reset -> Init.
  ASSERT_EQ(InitRcQP(qp), 0);
  // Init -> RTR
  ibv_qp_attr mod_rtr = QpAttribute().GetRcInitToRtrAttr(
      setup.port_attr.port, setup.port_attr.gid_index, setup.port_attr.gid,
      qp->qp_num);
  int mask_rtr = QpAttribute().GetRcInitToRtrMask();
  ASSERT_EQ(ibv_modify_qp(qp, &mod_rtr, mask_rtr), 0);
  // RTR -> RTS
  ibv_qp_attr mod_rts = QpAttribute().GetRcRtrToRtsAttr();
  int mask_rts = QpAttribute().GetRcRtrToRtsMask();
  ASSERT_EQ(ibv_modify_qp(qp, &mod_rts, mask_rts), 0);
  // RTS -> SQD
  ibv_qp_attr mod_sqd = QpAttribute().GetRcRtsToSqdAttr();
  int mask_sqd = QpAttribute().GetRcRtsToSqdMask();
  EXPECT_EQ(ibv_modify_qp(qp, &mod_sqd, mask_sqd), 0);
  // SQD -> RTS with invalid mask
  for (auto mask : CreateMaskCombinations(mask_rts)) {
    if (mask != mask_sqd) {
      EXPECT_NE(ibv_modify_qp(qp, &mod_rts, mask), 0);
    }
  }
  // SQD -> RTS with valid mask
  EXPECT_EQ(ibv_modify_qp(qp, &mod_rts, mask_sqd), 0);
}

TEST_F(QpTest, UdModify) {
  if (!Introspection().SupportsUdQp()) {
    GTEST_SKIP();
  }
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp_init_attr attr = setup.basic_attr;
  attr.qp_type = IBV_QPT_UD;
  ibv_qp* qp = ibv_.CreateQp(setup.pd, attr);
  ASSERT_THAT(qp, NotNull());

  // Reset -> Init.
  ASSERT_EQ(InitUdQP(qp, /*qkey=*/17), 0);
  // Init -> Ready to receive.
  ibv_qp_attr mod_rtr = {};
  mod_rtr.qp_state = IBV_QPS_RTR;
  constexpr int kRtrMask = IBV_QP_STATE;
  ASSERT_EQ(ibv_modify_qp(qp, &mod_rtr, kRtrMask), 0);
  // Ready to receive -> Ready to send.
  ibv_qp_attr mod_rts = {};
  mod_rts.qp_state = IBV_QPS_RTS;
  mod_rts.sq_psn = 1225;
  constexpr int kRtsMask = IBV_QP_STATE | IBV_QP_SQ_PSN;
  ASSERT_EQ(ibv_modify_qp(qp, &mod_rts, kRtsMask), 0);
}

TEST_F(QpTest, DeallocPdWithOutstandingQp) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());
  EXPECT_EQ(ibv_.DeallocPd(setup.pd), EBUSY);
}

TEST_F(QpTest, ModifyInvalidQp) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());
  HandleGarble garble(qp->handle);
  ibv_qp_attr mod_init =
      QpAttribute().GetRcResetToInitAttr(setup.port_attr.port);
  int mod_init_mask = QpAttribute().GetRcResetToInitMask();
  EXPECT_THAT(ibv_modify_qp(qp, &mod_init, mod_init_mask),
              AnyOf(ENOENT, EINVAL));
}

TEST_F(QpTest, QueryInvalidQp) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());
  // Reset -> Init.
  ASSERT_EQ(InitRcQP(qp), 0);
  ASSERT_EQ(verbs_util::GetQpState(qp), IBV_QPS_INIT);
  // Query on invalid qp.
  ibv_qp_attr attr;
  ibv_qp_init_attr init_attr;
  HandleGarble garble(qp->handle);
  EXPECT_THAT(ibv_query_qp(qp, &attr, IBV_QP_STATE, &init_attr),
              AnyOf(ENOENT, EINVAL));
}

TEST_F(QpTest, DestroyInvalidQp) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());
  HandleGarble garble(qp->handle);
  EXPECT_THAT(ibv_destroy_qp(qp), AnyOf(ENOENT, EINVAL));
}

// This is a series of test to test whether a QP correctly responds to a posted
// WR when QP is in different state. The expectation is set based on the IBTA
// spec, but some low level drivers will deviate from the spec for better
// performance. See specific testcases for explanation.
class QpStateTest : public RdmaVerbsFixture {
 protected:
  struct BasicSetup {
    ibv_context* context = nullptr;
    PortAttribute port_attr;
    ibv_cq* cq = nullptr;
    ibv_pd* pd = nullptr;
    ibv_qp* local_qp = nullptr;
    ibv_qp* remote_qp = nullptr;
    RdmaMemBlock buffer;
    ibv_mr* mr = nullptr;
  };

  absl::StatusOr<BasicSetup> CreateBasicSetup() {
    BasicSetup setup;
    ASSIGN_OR_RETURN(setup.context, ibv_.OpenDevice());
    setup.cq = ibv_.CreateCq(setup.context);
    if (!setup.cq) {
      return absl::InternalError("Failed to create cq.");
    }
    setup.port_attr = ibv_.GetPortAttribute(setup.context);
    setup.pd = ibv_.AllocPd(setup.context);
    if (!setup.pd) {
      return absl::InternalError("Failed to allocate pd.");
    }
    setup.local_qp = ibv_.CreateQp(setup.pd, setup.cq);
    if (!setup.local_qp) {
      return absl::InternalError("Failed to create qp.");
    }
    setup.remote_qp = ibv_.CreateQp(setup.pd, setup.cq);
    if (!setup.remote_qp) {
      return absl::InternalError("Failed to create remote qp.");
    }
    setup.buffer = ibv_.AllocBuffer(/*pages=*/1);
    setup.mr = ibv_.RegMr(setup.pd, setup.buffer);
    if (!setup.mr) {
      return absl::InternalError("Failed to register mr.");
    }
    return setup;
  }
};

// IBTA specs says we posting to a QP in RESET, INIT or RTR state should be
// be returned with immediate error (see IBTA v1, chapter 10.8.2, c10-96).
// But a vast majority of the low-level driver deviate from this rule for better
// performance. Hence here we anticipate the post is successful and make
// assertions in the event it is.
TEST_F(QpStateTest, PostSendReset) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ASSERT_EQ(verbs_util::GetQpState(setup.local_qp), IBV_QPS_RESET);
  ibv_sge sge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, setup.buffer.data(), setup.mr->rkey);
  ibv_send_wr* bad_wr = nullptr;
  // This deviates from IBTA spec, see comment above the test.
  if (ibv_post_send(setup.local_qp, &read, &bad_wr))
    return;
  // Modify the QP to RTS before we poll for completion.
  ASSERT_OK(ibv_.SetUpLoopbackRcQps(setup.remote_qp, setup.local_qp,
                                    setup.port_attr));
  EXPECT_EQ(verbs_util::GetQpState(setup.local_qp), IBV_QPS_RTS);
  EXPECT_EQ(verbs_util::GetQpState(setup.remote_qp), IBV_QPS_RTS);
  if (Introspection().SilentlyDropSendWrWhenResetInitRtr()) {
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(setup.cq));
    return;
  }
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(setup.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
}

TEST_F(QpStateTest, PostRecvReset) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ASSERT_EQ(verbs_util::GetQpState(setup.remote_qp), IBV_QPS_RESET);
  ibv_sge sge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_recv_wr recv = verbs_util::CreateRecvWr(/*wr_id=*/0, &sge, /*num_sge=*/1);
  ibv_recv_wr* bad_wr = nullptr;
  if (ibv_post_recv(setup.remote_qp, &recv, &bad_wr))
    return;
  // Modify local_qp to RTS state and post a send request.
  ASSERT_OK(ibv_.ModifyRcQpResetToRts(
      setup.local_qp, setup.port_attr, setup.port_attr.gid,
      setup.remote_qp->qp_num, QpAttribute().set_timeout(absl::Seconds(1))));
  ibv_sge lsge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(setup.local_qp, send);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(setup.cq));
  EXPECT_EQ(completion.status, IBV_WC_RETRY_EXC_ERR);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.qp_num, setup.local_qp->qp_num);
  EXPECT_TRUE(verbs_util::ExpectNoCompletion(setup.cq));
}

TEST_F(QpStateTest, PostSendInit) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ASSERT_EQ(ibv_.ModifyRcQpResetToInit(setup.local_qp, setup.port_attr.port),
            0);
  ASSERT_EQ(verbs_util::GetQpState(setup.local_qp), IBV_QPS_INIT);
  ibv_sge sge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, setup.buffer.data(), setup.mr->rkey);
  ibv_send_wr* bad_wr = nullptr;
  // This deviates from IBTA spec, see comment at QpStateTest.PostSendReset
  // for details.
  if (ibv_post_send(setup.local_qp, &read, &bad_wr))
    return;
  ASSERT_OK(ibv_.ModifyLoopbackRcQpResetToRts(setup.remote_qp, setup.local_qp,
                                              setup.port_attr));
  // Modify the QP to RTS before we poll for completion.
  ASSERT_EQ(
      ibv_.ModifyRcQpInitToRtr(setup.local_qp, setup.port_attr,
                               setup.port_attr.gid, setup.remote_qp->qp_num),
      0);
  ASSERT_EQ(ibv_.ModifyRcQpRtrToRts(setup.local_qp), 0);
  EXPECT_EQ(verbs_util::GetQpState(setup.local_qp), IBV_QPS_RTS);
  EXPECT_EQ(verbs_util::GetQpState(setup.remote_qp), IBV_QPS_RTS);
  if (Introspection().SilentlyDropSendWrWhenResetInitRtr()) {
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(setup.cq));
    return;
  }
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(setup.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
}

// We are allowed to post to recv queue in INIT or RTR state, but the WR will
// not get processed.
TEST_F(QpStateTest, PostRecvInit) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ASSERT_EQ(ibv_.ModifyRcQpResetToInit(setup.remote_qp, setup.port_attr.port),
            0);
  ASSERT_EQ(verbs_util::GetQpState(setup.remote_qp), IBV_QPS_INIT);
  ibv_sge sge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_recv_wr recv = verbs_util::CreateRecvWr(/*wr_id=*/0, &sge, /*num_sge=*/1);
  ibv_recv_wr* bad_wr = nullptr;
  ASSERT_EQ(ibv_post_recv(setup.remote_qp, &recv, &bad_wr), 0);
  ASSERT_OK(ibv_.ModifyRcQpResetToRts(
      setup.local_qp, setup.port_attr, setup.port_attr.gid,
      setup.remote_qp->qp_num, QpAttribute().set_timeout(absl::Seconds(1))));
  ibv_sge lsge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(setup.local_qp, send);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(setup.cq));
  EXPECT_EQ(completion.status, IBV_WC_RETRY_EXC_ERR);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.qp_num, setup.local_qp->qp_num);
  EXPECT_TRUE(verbs_util::ExpectNoCompletion(setup.cq));
}

TEST_F(QpStateTest, PostSendRtr) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ASSERT_EQ(ibv_.ModifyRcQpResetToInit(setup.local_qp, setup.port_attr.port),
            0);
  ASSERT_EQ(
      ibv_.ModifyRcQpInitToRtr(setup.local_qp, setup.port_attr,
                               setup.port_attr.gid, setup.remote_qp->qp_num),
      0);
  ASSERT_EQ(verbs_util::GetQpState(setup.local_qp), IBV_QPS_RTR);
  ibv_sge sge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_send_wr read = verbs_util::CreateReadWr(
      /*wr_id=*/1, &sge, /*num_sge=*/1, setup.buffer.data(), setup.mr->rkey);
  ibv_send_wr* bad_wr = nullptr;
  // This deviates from IBTA spec, see comment at QpStateTest.PostSendReset
  // for details.
  if (ibv_post_send(setup.local_qp, &read, &bad_wr))
    return;
  ASSERT_OK(ibv_.ModifyLoopbackRcQpResetToRts(setup.remote_qp, setup.local_qp,
                                              setup.port_attr));
  ASSERT_EQ(ibv_.ModifyRcQpRtrToRts(setup.local_qp), 0);
  EXPECT_EQ(verbs_util::GetQpState(setup.local_qp), IBV_QPS_RTS);
  EXPECT_EQ(verbs_util::GetQpState(setup.remote_qp), IBV_QPS_RTS);
  if (Introspection().SilentlyDropSendWrWhenResetInitRtr()) {
    EXPECT_TRUE(verbs_util::ExpectNoCompletion(setup.cq));
    return;
  }
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(setup.cq));
  EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
  EXPECT_EQ(completion.wr_id, 1);
  EXPECT_EQ(completion.opcode, IBV_WC_RDMA_READ);
}

TEST_F(QpStateTest, PostRecvRtr) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ASSERT_EQ(ibv_.ModifyRcQpResetToInit(setup.remote_qp, setup.port_attr.port),
            0);
  ASSERT_EQ(
      ibv_.ModifyRcQpInitToRtr(setup.remote_qp, setup.port_attr,
                               setup.port_attr.gid, setup.local_qp->qp_num),
      0);
  ASSERT_EQ(verbs_util::GetQpState(setup.remote_qp), IBV_QPS_RTR);
  ibv_sge sge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_recv_wr recv = verbs_util::CreateRecvWr(/*wr_id=*/0, &sge, /*num_sge=*/1);
  ibv_recv_wr* bad_wr = nullptr;
  EXPECT_EQ(ibv_post_recv(setup.remote_qp, &recv, &bad_wr), 0);
  ASSERT_EQ(ibv_.ModifyRcQpRtrToRts(setup.remote_qp), 0);
  ASSERT_OK(ibv_.ModifyRcQpResetToRts(setup.local_qp, setup.port_attr,
                                      setup.port_attr.gid,
                                      setup.remote_qp->qp_num));
  ibv_sge lsge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/1, &lsge, /*num_sge=*/1);
  verbs_util::PostSend(setup.local_qp, send);
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(setup.cq));
    if (completion.wr_id == 1) {
      EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
      EXPECT_EQ(completion.wr_id, 1);
      EXPECT_EQ(completion.qp_num, setup.local_qp->qp_num);
      EXPECT_EQ(completion.opcode, IBV_WC_SEND);
      EXPECT_EQ(verbs_util::GetQpState(setup.remote_qp), IBV_QPS_RTS);
    } else {
      EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
      EXPECT_EQ(completion.wr_id, 0);
      EXPECT_EQ(completion.qp_num, setup.remote_qp->qp_num);
      EXPECT_EQ(completion.opcode, IBV_WC_RECV);
      EXPECT_EQ(verbs_util::GetQpState(setup.remote_qp), IBV_QPS_RTS);
    }
  }
}

TEST_F(QpStateTest, PostSendErr) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ASSERT_OK(ibv_.SetUpLoopbackRcQps(setup.local_qp, setup.remote_qp,
                                    setup.port_attr));
  ASSERT_OK(ibv_.ModifyQpToError(setup.local_qp));
  ASSERT_EQ(verbs_util::GetQpState(setup.local_qp), IBV_QPS_ERR);
  ASSERT_OK_AND_ASSIGN(
      ibv_wc_status status,
      verbs_util::ExecuteRdmaRead(setup.local_qp, setup.buffer.span(), setup.mr,
                                  setup.buffer.data(), setup.mr->rkey));
  EXPECT_EQ(status, IBV_WC_WR_FLUSH_ERR);
}

TEST_F(QpStateTest, PostRecvErr) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ASSERT_OK(ibv_.SetUpLoopbackRcQps(setup.local_qp, setup.remote_qp,
                                    setup.port_attr));
  ASSERT_OK(ibv_.ModifyQpToError(setup.remote_qp));
  ASSERT_EQ(verbs_util::GetQpState(setup.remote_qp), IBV_QPS_ERR);
  ibv_sge rsge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/2, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(setup.remote_qp, recv);
  ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                       verbs_util::WaitForCompletion(setup.cq));
  EXPECT_EQ(completion.status, IBV_WC_WR_FLUSH_ERR);
}

TEST_F(QpStateTest, RtsSendToRtr) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ASSERT_EQ(ibv_.ModifyRcQpResetToInit(setup.local_qp, setup.port_attr.port),
            0);
  ASSERT_EQ(
      ibv_.ModifyRcQpInitToRtr(setup.local_qp, setup.port_attr,
                               setup.port_attr.gid, setup.remote_qp->qp_num),
      0);
  ASSERT_EQ(ibv_.ModifyRcQpRtrToRts(setup.local_qp), 0);
  ASSERT_EQ(verbs_util::GetQpState(setup.local_qp), IBV_QPS_RTS);
  ASSERT_EQ(ibv_.ModifyRcQpResetToInit(setup.remote_qp, setup.port_attr.port),
            0);
  ASSERT_EQ(
      ibv_.ModifyRcQpInitToRtr(setup.remote_qp, setup.port_attr,
                               setup.port_attr.gid, setup.local_qp->qp_num),
      0);
  ASSERT_EQ(verbs_util::GetQpState(setup.remote_qp), IBV_QPS_RTR);

  ibv_sge rsge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_recv_wr recv =
      verbs_util::CreateRecvWr(/*wr_id=*/1, &rsge, /*num_sge=*/1);
  verbs_util::PostRecv(setup.remote_qp, recv);
  ibv_sge ssge = verbs_util::CreateSge(setup.buffer.span(), setup.mr);
  ibv_send_wr send =
      verbs_util::CreateSendWr(/*wr_id=*/0, &ssge, /*num_sge=*/1);
  verbs_util::PostSend(setup.local_qp, send);

  for (int i = 0; i < 2; ++i) {
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(setup.cq));
    ASSERT_EQ(completion.status, IBV_WC_SUCCESS);
    if (completion.qp_num == setup.remote_qp->qp_num) {
      EXPECT_EQ(completion.opcode, IBV_WC_RECV);
      EXPECT_EQ(completion.wr_id, 1);
    } else {
      EXPECT_EQ(completion.opcode, IBV_WC_SEND);
      EXPECT_EQ(completion.wr_id, 0);
    }
  }
}

TEST_F(QpStateTest, ModRtsToError) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ASSERT_OK(ibv_.SetUpLoopbackRcQps(setup.local_qp, setup.remote_qp,
                                    setup.port_attr));
  ibv_qp_attr attr;
  attr.qp_state = IBV_QPS_ERR;
  ibv_qp_attr_mask attr_mask = IBV_QP_STATE;
  ASSERT_EQ(ibv_modify_qp(setup.local_qp, &attr, attr_mask), 0);
  ASSERT_EQ(verbs_util::GetQpState(setup.local_qp), IBV_QPS_ERR);
}

// This is a set of test that test the response (in particular error code) of
// the QP when posting.
class QpPostTest : public RdmaVerbsFixture {
 protected:
  struct BasicSetup {
    ibv_context* context = nullptr;
    PortAttribute port_attr;
    ibv_cq* cq = nullptr;
    ibv_pd* pd = nullptr;
    ibv_qp* qp = nullptr;
    ibv_qp* remote_qp = nullptr;
    RdmaMemBlock buffer;
    ibv_mr* mr = nullptr;
  };

  absl::StatusOr<BasicSetup> CreateBasicSetup() {
    BasicSetup setup;
    ASSIGN_OR_RETURN(setup.context, ibv_.OpenDevice());
    setup.cq = ibv_.CreateCq(setup.context);
    if (!setup.cq) {
      return absl::InternalError("Failed to create cq.");
    }
    setup.port_attr = ibv_.GetPortAttribute(setup.context);
    setup.pd = ibv_.AllocPd(setup.context);
    if (!setup.pd) {
      return absl::InternalError("Failed to allocate pd.");
    }
    setup.qp = ibv_.CreateQp(setup.pd, setup.cq, IBV_QPT_RC);
    if (!setup.qp) {
      return absl::InternalError("Failed to create qp.");
    }
    setup.remote_qp = ibv_.CreateQp(setup.pd, setup.cq);
    if (!setup.remote_qp) {
      return absl::InternalError("Failed to create remote qp.");
    }
    RETURN_IF_ERROR(
        ibv_.SetUpLoopbackRcQps(setup.qp, setup.remote_qp, setup.port_attr));
    setup.buffer = ibv_.AllocBuffer(/*pages=*/1);
    setup.mr = ibv_.RegMr(setup.pd, setup.buffer);
    if (!setup.mr) {
      return absl::InternalError("Failed to register mr.");
    }
    return setup;
  }

  static ibv_send_wr DummySend() {
    ibv_send_wr send;
    send.wr_id = 1;
    send.next = nullptr;
    send.sg_list = nullptr;
    send.num_sge = 0;
    send.opcode = IBV_WR_SEND;
    send.send_flags = IBV_SEND_SIGNALED;
    return send;
  }
};

TEST_F(QpPostTest, OverflowSendWr) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  // Issue enough commands to overflow the queue.
  ibv_send_wr wqe = DummySend();
  int max_send_wr = verbs_util::GetQpCap(setup.qp).max_send_wr;
  std::vector<ibv_send_wr> wqes(max_send_wr + 1, wqe);
  for (unsigned int i = 0; i < wqes.size() - 1; ++i) {
    wqes[i].wr_id = wqe.wr_id + i;
    wqes[i].next = &wqes[i + 1];
  }
  ibv_send_wr* bad_wr = nullptr;
  EXPECT_EQ(ibv_post_send(setup.qp, wqes.data(), &bad_wr), ENOMEM);
  EXPECT_EQ(bad_wr, &wqes[max_send_wr]);
}

class ResponderQpStateTest : public LoopbackFixture,
                             public testing::WithParamInterface<
                                 std::tuple<ibv_qp_state, ibv_wr_opcode>> {
 protected:
  void BringClientQpToState(Client& client, ibv_gid remote_gid,
                            uint32_t remote_qpn, ibv_qp_state target_state) {
    ASSERT_EQ(verbs_util::GetQpState(client.qp), IBV_QPS_RESET);
    if (target_state == IBV_QPS_RESET) {
      return;
    }
    ASSERT_EQ(ibv_.ModifyRcQpResetToInit(client.qp, client.port_attr.port), 0);
    if (target_state == IBV_QPS_INIT) {
      return;
    }
    ASSERT_EQ(ibv_.ModifyRcQpInitToRtr(client.qp, client.port_attr, remote_gid,
                                       remote_qpn),
              0);
    if (target_state == IBV_QPS_RTR) {
      return;
    }
    ASSERT_EQ(ibv_.ModifyRcQpRtrToRts(client.qp), 0);
  }

  void BringClientQpToRts(Client& client, ibv_gid remote_gid,
                          uint32_t remote_qpn) {
    ibv_qp_state current_state = verbs_util::GetQpState(client.qp);
    if (current_state == IBV_QPS_RESET) {
      ASSERT_EQ(ibv_.ModifyRcQpResetToInit(client.qp, client.port_attr.port),
                0);
      current_state = IBV_QPS_INIT;
    }
    if (current_state == IBV_QPS_INIT) {
      ASSERT_EQ(ibv_.ModifyRcQpInitToRtr(client.qp, client.port_attr,
                                         remote_gid, remote_qpn),
                0);
      current_state = IBV_QPS_RTR;
    }
    if (current_state == IBV_QPS_RTR) {
      ASSERT_EQ(ibv_.ModifyRcQpRtrToRts(client.qp), 0);
    }
  }
};

TEST_P(ResponderQpStateTest, ResponderNotReadyToReceive) {
  ASSERT_OK_AND_ASSIGN(Client requestor, CreateClient());
  ASSERT_OK_AND_ASSIGN(Client responder, CreateClient());
  ibv_qp_state responder_state = std::get<0>(GetParam());
  ibv_wr_opcode opcode = std::get<1>(GetParam());

  ASSERT_OK(ibv_.ModifyRcQpResetToRts(requestor.qp, requestor.port_attr,
                                      responder.port_attr.gid,
                                      responder.qp->qp_num,
                                      QpAttribute()
                                          .set_timeout(absl::Seconds(1))
                                          .set_retry_cnt(5)
                                          .set_rnr_retry(5)));
  ASSERT_NO_FATAL_FAILURE(
      BringClientQpToState(responder, requestor.port_attr.gid,
                           requestor.qp->qp_num, responder_state));
  ASSERT_EQ(verbs_util::GetQpState(requestor.qp), IBV_QPS_RTS);
  ASSERT_EQ(verbs_util::GetQpState(responder.qp), responder_state);

  ibv_sge sge = verbs_util::CreateSge(requestor.buffer.span(), requestor.mr);
  ibv_send_wr wr =
      verbs_util::CreateRdmaWr(opcode, /*wr_id=*/0, &sge, /*num_sge=*/1,
                               responder.buffer.data(), responder.mr->rkey);
  verbs_util::PostSend(requestor.qp, wr);
  EXPECT_EQ(verbs_util::GetQpState(requestor.qp), IBV_QPS_RTS);
  EXPECT_EQ(verbs_util::GetQpState(responder.qp), responder_state);

  if (!Introspection().BuffersMessagesWhenNotReadyToReceive()) {
    LOG(INFO) << "Provider does not buffer incoming messages. Expecting "
                 "unsuccessful completions.";
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(requestor.cq));
    EXPECT_EQ(completion.status, IBV_WC_RETRY_EXC_ERR);
    EXPECT_EQ(completion.wr_id, wr.wr_id);
    EXPECT_EQ(completion.qp_num, requestor.qp->qp_num);
  }

  ASSERT_NO_FATAL_FAILURE(BringClientQpToRts(responder, requestor.port_attr.gid,
                                             requestor.qp->qp_num));

  if (Introspection().BuffersMessagesWhenNotReadyToReceive()) {
    LOG(INFO) << "Provider buffers incoming messages. Expecting "
                 "successful completions.";
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(requestor.cq));
    EXPECT_EQ(completion.status, IBV_WC_SUCCESS);
    EXPECT_EQ(completion.wr_id, wr.wr_id);
    EXPECT_EQ(completion.qp_num, requestor.qp->qp_num);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ResponderQpStateTest, ResponderQpStateTest,
    testing::Combine(testing::Values(IBV_QPS_RESET, IBV_QPS_INIT),
                     testing::Values(IBV_WR_RDMA_READ, IBV_WR_RDMA_WRITE)),
    [](const testing::TestParamInfo<ResponderQpStateTest::ParamType>& param) {
      std::string state = [param]() {
        switch (std::get<0>(param.param)) {
          case IBV_QPS_RESET:
            return "Reset";
          case IBV_QPS_INIT:
            return "Init";
          default:
            return "Unknown";
        }
      }();
      std::string opcode = [param]() {
        switch (std::get<1>(param.param)) {
          case IBV_WR_RDMA_READ:
            return "Read";
          case IBV_WR_RDMA_WRITE:
            return "Write";
          default:
            return "Unknown";
        }
      }();
      return absl::StrCat("Service", opcode, "In", state, "State");
    });

// TODO(author1): Test larger MTU than the device allows

struct QpStateTransition {
  ibv_qp_state src_state;
  ibv_qp_state dst_state;
  bool valid;
};

class ModifyQpState : public QpTest,
                      public testing::WithParamInterface<QpStateTransition> {
 public:
  int ErrRcQP(ibv_qp* qp) const {
    // Modify QP State to ERR state
    ibv_qp_attr attr;
    attr.qp_state = IBV_QPS_ERR;
    ibv_qp_attr_mask attr_mask = IBV_QP_STATE;
    return ibv_modify_qp(qp, &attr, attr_mask);
  }
  absl::Status SetQpState(ibv_qp* qp, BasicSetup setup, ibv_qp_state qp_state) {
    // Move QP state from Reset to qp_state
    QpAttribute qp_attr = QpAttribute().set_timeout(absl::Seconds(1));
    int result_code;
    switch (qp_state) {
      case IBV_QPS_RESET:
        return absl::OkStatus();
      // Reset -> Init
      case IBV_QPS_INIT:
        result_code = InitRcQP(qp);
        if (result_code) {
          return absl::InternalError(absl::StrFormat(
              "Modified QP from RESET to INIT failed (%d).", result_code));
        }
        return absl::OkStatus();
      // Reset -> RTR
      case IBV_QPS_RTR:
        return (ibv_.ModifyRcQpResetToRtr(
            qp, setup.port_attr, setup.port_attr.gid, qp->qp_num, qp_attr));
      // Reset -> RTS
      case IBV_QPS_RTS:
        return (ibv_.ModifyRcQpResetToRts(
            qp, setup.port_attr, setup.port_attr.gid, qp->qp_num, qp_attr));
      // Reset -> SQD
      case IBV_QPS_SQD:
        return (ibv_.ModifyRcQpResetToSqd(
            qp, setup.port_attr, setup.port_attr.gid, qp->qp_num, qp_attr));
      // Reset -> ERR
      case IBV_QPS_ERR:
        result_code = InitRcQP(qp);
        if (result_code) {
          return absl::InternalError(absl::StrFormat(
              "Modified QP from RESET to INIT failed (%d).", result_code));
        }
        result_code = ErrRcQP(qp);
        if (result_code) {
          return absl::InternalError(absl::StrFormat(
              "Modified QP from INIT to ERR failed (%d).", result_code));
        }
        return absl::OkStatus();
    }
  }
};

TEST_P(ModifyQpState, ModifyQpStateTest) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ibv_qp* qp = ibv_.CreateQp(setup.pd, setup.basic_attr);
  ASSERT_THAT(qp, NotNull());
  const QpStateTransition& kTestCase = GetParam();
  // Move QP to src_state
  ASSERT_OK(SetQpState(qp, setup, kTestCase.src_state));
  ASSERT_EQ(verbs_util::GetQpState(qp), kTestCase.src_state);
  ibv_qp_attr attr;
  attr.qp_state = kTestCase.dst_state;
  ibv_qp_attr_mask attr_mask = IBV_QP_STATE;
  // Execute valid transitions and expect success
  if (kTestCase.valid) {
    ASSERT_EQ(ibv_modify_qp(qp, &attr, attr_mask), 0);
    ASSERT_EQ(verbs_util::GetQpState(qp), kTestCase.dst_state);
  }
  // Execute Invalid transitions and expect failure
  else {
    ASSERT_NE(ibv_modify_qp(qp, &attr, attr_mask), 0);
  }
}

std::vector<QpStateTransition> GenerateModifyQpStateParameters() {
  // Generate valid and invalid QP transitions
  std::vector<QpStateTransition> params;
  const int no_of_states = 7;
  // 0 is for invalid
  // 1 is for valid
  // 2 is for exclude
  // Reset to Init, Init to RTR and RTR to RTS are covered in QpTest.Modify
  // SQE is not supported in RC QP
  int state_transition_map[no_of_states][no_of_states] = {
      /*         RESET INIT RTR RTS SQD SQE ERR*/
      /* RESET */ {1, 2, 0, 0, 0, 0, 0},
      /* INIT */ {1, 1, 2, 0, 0, 0, 1},
      /* RTR */ {1, 0, 0, 2, 0, 0, 1},
      /* RTS */ {1, 0, 0, 1, 1, 0, 1},
      /* SQD */ {1, 0, 0, 1, 1, 0, 1},
      /* SQE */ {2, 2, 2, 2, 2, 2, 2},
      /* ERR */ {1, 0, 0, 0, 0, 0, 1}};
  for (int i = 0; i < no_of_states; i++) {
    for (int j = 0; j < no_of_states; j++) {
      if (state_transition_map[i][j] == 2) {
        continue;
      }
      QpStateTransition param{.src_state = static_cast<ibv_qp_state>(i),
                              .dst_state = static_cast<ibv_qp_state>(j),
                              .valid = state_transition_map[i][j]};
      params.push_back(param);
    }
  }
  return params;
}

std::string qp_state_enum_str(ibv_qp_state qp_state) {
  switch (qp_state) {
    case IBV_QPS_RESET:
      return "Reset";
    case IBV_QPS_INIT:
      return "Init";
    case IBV_QPS_RTR:
      return "RTR";
    case IBV_QPS_RTS:
      return "RTS";
    case IBV_QPS_SQD:
      return "SQD";
    case IBV_QPS_SQE:
      return "SQE";
    case IBV_QPS_ERR:
      return "ERR";
    default:
      return "Unknown";
  }
}
INSTANTIATE_TEST_SUITE_P(
    QpStateTest, ModifyQpState,
    testing::ValuesIn(GenerateModifyQpStateParameters()),
    [](const testing::TestParamInfo<ModifyQpState::ParamType>& info) {
      std::string src_state = qp_state_enum_str(info.param.src_state);
      std::string dst_state = qp_state_enum_str(info.param.dst_state);
      std::string valid = info.param.valid ? "Valid" : "Invalid";
      return absl::StrCat("Modify", src_state, "To", dst_state, valid);
    });
}  // namespace rdma_unit_test
