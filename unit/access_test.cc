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

#include <cstdint>
#include <string>
#include <utility>

#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "infiniband/verbs.h"
#include "public/introspection.h"
#include "public/rdma_memblock.h"
#include "public/status_matchers.h"
#include "public/verbs_helper_suite.h"
#include "public/verbs_util.h"
#include "unit/rdma_verbs_fixture.h"

namespace rdma_unit_test {
namespace {

using ::testing::NotNull;
using ::testing::Pair;

// The type of Ibverbs memory object from which an RKEY is derived.
enum class RKeyMemmoryType {
  kMemoryRegion,
  kMemoryWindowType1,
  kMemoryWindowType2
};

class AccessTestFixture : public RdmaVerbsFixture {
 protected:
  struct BasicSetup {
    ibv_context* context;
    PortAttribute port_attr;
    ibv_pd* pd;
    RdmaMemBlock src_buffer;
    RdmaMemBlock dst_buffer;
    RdmaMemBlock dst_buffer_2;
    ibv_cq* src_cq;
    ibv_cq* dst_cq;
    ibv_qp* src_qp;
    ibv_qp* dst_qp;
  };

  constexpr static int kMrAccessAll =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_MW_BIND | IBV_ACCESS_REMOTE_ATOMIC |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  constexpr static int kRemoteAccessAll = IBV_ACCESS_REMOTE_READ |
                                          IBV_ACCESS_REMOTE_WRITE |
                                          IBV_ACCESS_REMOTE_ATOMIC;

  absl::StatusOr<BasicSetup> CreateBasicSetup() {
    BasicSetup setup;
    setup.src_buffer = ibv_.AllocBuffer(/*pages=*/2);
    setup.dst_buffer = ibv_.AllocBuffer(/*pages=*/2);
    ASSIGN_OR_RETURN(setup.context, ibv_.OpenDevice());
    setup.port_attr = ibv_.GetPortAttribute(setup.context);
    setup.pd = ibv_.AllocPd(setup.context);
    if (!setup.pd) {
      return absl::InternalError("Failed to allcoate pd.");
    }
    setup.src_cq = ibv_.CreateCq(setup.context);
    if (!setup.src_cq) {
      return absl::InternalError("Failed to create source qp.");
    }
    setup.dst_cq = ibv_.CreateCq(setup.context);
    if (!setup.dst_cq) {
      return absl::InternalError("Failed to create destination qp.");
    }
    setup.src_qp = ibv_.CreateQp(setup.pd, setup.src_cq);
    if (!setup.src_qp) {
      return absl::InternalError("Failed to create source qp.");
    }
    setup.dst_qp = ibv_.CreateQp(setup.pd, setup.dst_cq);
    if (!setup.dst_qp) {
      return absl::InternalError("Failed to create destination qp.");
    }
    RETURN_IF_ERROR(
        ibv_.SetUpLoopbackRcQps(setup.src_qp, setup.dst_qp, setup.port_attr));
    return setup;
  }

  // Generate a R_KEY for a `memory_type` on a buffer specified by `memblock`.
  // The memory type can be memory region, type 1 or type 2 memory window.
  // `access` can be a combination of IBV_ACCESS_REMOTE_READ,
  // IBV_ACCESS_REMOTE_WRITE and IBV_ACCESS_REMOTE_ATOMIC.
  absl::StatusOr<uint32_t> CreateRKeyForBuffer(ibv_pd* pd,
                                               const RdmaMemBlock& memblock,
                                               ibv_qp* qp, int access,
                                               RKeyMemmoryType memory_type) {
    // Always enable local write on an MR since MR with remote write/atomics
    // will also requrie local write. This should not have any impact on the
    // test since R_KEY access does not care about local write access.
    const int mr_access = memory_type == RKeyMemmoryType::kMemoryRegion
                              ? access | IBV_ACCESS_LOCAL_WRITE
                              : kMrAccessAll;
    ibv_mr* mr = ibv_.RegMr(pd, memblock, mr_access);
    if (mr == nullptr) {
      return absl::InternalError("Failed to register MR.");
    }
    if (memory_type == RKeyMemmoryType::kMemoryRegion) {
      return mr->rkey;
    }
    ibv_mw_type mw_type = memory_type == RKeyMemmoryType::kMemoryWindowType1
                              ? IBV_MW_TYPE_1
                              : IBV_MW_TYPE_2;
    ibv_mw* mw = ibv_.AllocMw(pd, mw_type);
    if (mw == nullptr) {
      return absl::InternalError("Failed to allocate MW.");
    }
    ibv_wc_status status = IBV_WC_SUCCESS;
    switch (mw_type) {
      case IBV_MW_TYPE_1: {
        ASSIGN_OR_RETURN(status, verbs_util::ExecuteType1MwBind(
                                     qp, mw, memblock.span(), mr, access));
        break;
      }
      case IBV_MW_TYPE_2: {
        static uint32_t type2_rkey = 1024;
        ASSIGN_OR_RETURN(
            status, verbs_util::ExecuteType2MwBind(qp, mw, memblock.span(),
                                                   ++type2_rkey, mr, access));
        if (status == IBV_WC_SUCCESS) {
          uint32_t ukey_mask = 0xff;
          mw->rkey = (mw->rkey & ~ukey_mask) | (type2_rkey & ukey_mask);
        }
        break;
      }
      default:
        LOG(FATAL) << "Unknown param";  // Crash OK
    }
    if (status != IBV_WC_SUCCESS) {
      return absl::InternalError(
          absl::StrCat("Cannot bind mw (", status, ")."));
    }
    return mw->rkey;
  }
};

class MessagingAccessTest : public AccessTestFixture {
 protected:
  void ExecuteTest(BasicSetup setup, int src_mr_access, int dst_mr_access,
                   ibv_wc_status src_status_expected,
                   ibv_wc_status dst_status_expected) {
    ibv_mr* src_mr = ibv_.RegMr(setup.pd, setup.src_buffer, src_mr_access);
    ASSERT_THAT(src_mr, NotNull());
    ibv_mr* dst_mr = ibv_.RegMr(setup.pd, setup.dst_buffer, dst_mr_access);
    ASSERT_THAT(dst_mr, NotNull());
    ibv_qp* src_qp = ibv_.CreateQp(setup.pd, setup.src_cq);
    ASSERT_THAT(src_qp, NotNull());
    ibv_qp* dst_qp = ibv_.CreateQp(setup.pd, setup.dst_cq);
    ASSERT_THAT(dst_qp, NotNull());
    ASSERT_THAT(ibv_.SetUpLoopbackRcQps(src_qp, dst_qp, setup.port_attr),
                IsOk());
    EXPECT_THAT(
        verbs_util::ExecuteSendRecv(src_qp, dst_qp, setup.src_buffer.span(),
                                    src_mr, setup.dst_buffer.span(), dst_mr),
        IsOkAndHolds(Pair(src_status_expected, dst_status_expected)));
  }

  void ExecuteTwoSgeTest(BasicSetup setup, int src_mr_access, int dst_mr_access,
                         int dst_mr_access_2, ibv_wc_status src_status_expected,
                         ibv_wc_status dst_status_expected) {
    // Allocate memory to second  buffer on remote.
    setup.dst_buffer_2 = ibv_.AllocBuffer(/*pages=*/2);
    ibv_mr* src_mr = ibv_.RegMr(setup.pd, setup.src_buffer, src_mr_access);
    ASSERT_THAT(src_mr, NotNull());
    ibv_mr* dst_mr = ibv_.RegMr(setup.pd, setup.dst_buffer, dst_mr_access);
    ASSERT_THAT(dst_mr, NotNull());
    ibv_mr* dst_mr_2 =
        ibv_.RegMr(setup.pd, setup.dst_buffer_2, dst_mr_access_2);
    ASSERT_THAT(dst_mr_2, NotNull());
    ibv_qp* src_qp = ibv_.CreateQp(setup.pd, setup.src_cq);
    ASSERT_THAT(src_qp, NotNull());
    ibv_qp* dst_qp = ibv_.CreateQp(setup.pd, setup.dst_cq);
    ASSERT_THAT(dst_qp, NotNull());
    ASSERT_THAT(ibv_.SetUpLoopbackRcQps(src_qp, dst_qp, setup.port_attr),
                IsOk());
    EXPECT_THAT(
        verbs_util::TwoSgeSendRecv(src_qp, dst_qp, setup.src_buffer.span(),
                                   src_mr, setup.dst_buffer.span(), dst_mr,
                                   setup.dst_buffer_2.span(), dst_mr_2),
        IsOkAndHolds(Pair(src_status_expected, dst_status_expected)));
  }
};

TEST_F(MessagingAccessTest, AllAccess) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ExecuteTest(setup, kMrAccessAll, kMrAccessAll, IBV_WC_SUCCESS,
              IBV_WC_SUCCESS);
}

TEST_F(MessagingAccessTest, MissingSrcLocalWrite) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ExecuteTest(setup, IBV_ACCESS_MW_BIND | IBV_ACCESS_REMOTE_READ, kMrAccessAll,
              IBV_WC_SUCCESS, IBV_WC_SUCCESS);
}

TEST_F(MessagingAccessTest, MissingDstLocalWrite) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ExecuteTest(setup, kMrAccessAll, IBV_ACCESS_MW_BIND | IBV_ACCESS_REMOTE_READ,
              IBV_WC_REM_OP_ERR, IBV_WC_LOC_PROT_ERR);
}

TEST_F(MessagingAccessTest, MissingDstRemoteWrite) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ExecuteTest(setup, kMrAccessAll, kMrAccessAll & ~IBV_ACCESS_REMOTE_WRITE,
              IBV_WC_SUCCESS, IBV_WC_SUCCESS);
}

TEST_F(MessagingAccessTest, MinimalAccess) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  // Local read is always enabled on the MR by default.
  ExecuteTest(setup, 0, IBV_ACCESS_LOCAL_WRITE, IBV_WC_SUCCESS, IBV_WC_SUCCESS);
}

TEST_F(MessagingAccessTest, TwoSgeValidAccess) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ExecuteTwoSgeTest(setup, kMrAccessAll, kMrAccessAll, kMrAccessAll,
                    IBV_WC_SUCCESS, IBV_WC_SUCCESS);
}

TEST_F(MessagingAccessTest, TwoSgeMixedAccess) {
  ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
  ExecuteTwoSgeTest(setup, kMrAccessAll, kMrAccessAll,
                    IBV_ACCESS_MW_BIND | IBV_ACCESS_REMOTE_READ, IBV_WC_REM_OP_ERR,
                    IBV_WC_LOC_PROT_ERR);
}

// The testsuite define a parameterized tests to test RDMA READ/WRITE access
// to a memory buffer given by an RKEY.
class RdmaAccessTest : public AccessTestFixture,
                       public testing::WithParamInterface<
                           std::tuple<RKeyMemmoryType, ibv_wr_opcode>> {
 protected:
  void SetUp() override {
    AccessTestFixture::SetUp();
    if (std::get<0>(GetParam()) == RKeyMemmoryType::kMemoryWindowType1 &&
        !Introspection().SupportsType1()) {
      GTEST_SKIP() << "NIC does not support type 1 MW.";
    }
    if (std::get<0>(GetParam()) == RKeyMemmoryType::kMemoryWindowType2 &&
        !Introspection().SupportsType2()) {
      GTEST_SKIP() << "NIC does not support type 2 MW.";
    }
  }

  void ExecuteTest(int src_mr_access, int dst_remote_access,
                   ibv_wc_status expected_status) {
    ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
    RKeyMemmoryType memory_type = std::get<0>(GetParam());
    ibv_wr_opcode opcode = std::get<1>(GetParam());
    ibv_mr* src_mr = ibv_.RegMr(setup.pd, setup.src_buffer, src_mr_access);
    ASSERT_THAT(src_mr, NotNull());
    ASSERT_OK_AND_ASSIGN(
        uint32_t rkey,
        CreateRKeyForBuffer(setup.pd, setup.dst_buffer, setup.dst_qp,
                            dst_remote_access, memory_type));
    ibv_sge sge = verbs_util::CreateSge(setup.src_buffer.span(), src_mr);
    ibv_send_wr wr =
        verbs_util::CreateRdmaWr(opcode, /*wr_id=*/1, &sge, /*num_sge=*/1,
                                 setup.dst_buffer.data(), rkey);
    verbs_util::PostSend(setup.src_qp, wr);
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(setup.src_cq));
    EXPECT_EQ(completion.status, expected_status);
    EXPECT_EQ(completion.wr_id, wr.wr_id);
    EXPECT_EQ(completion.qp_num, setup.src_qp->qp_num);
  }

  void ExecuteTwoDstBufferTest(int src_mr_access, int dst_remote_access,
                               int dst_remote_access_2,
                               ibv_wc_status expected_status_1,
                               ibv_wc_status expected_status_2) {
    ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
    setup.dst_buffer_2 = ibv_.AllocBuffer(/*pages=*/2);
    RKeyMemmoryType memory_type = std::get<0>(GetParam());
    ibv_wr_opcode opcode = std::get<1>(GetParam());
    ibv_mr* src_mr = ibv_.RegMr(setup.pd, setup.src_buffer, src_mr_access);
    ASSERT_THAT(src_mr, NotNull());
    ASSERT_OK_AND_ASSIGN(
        uint32_t rkey,
        CreateRKeyForBuffer(setup.pd, setup.dst_buffer, setup.dst_qp,
                            dst_remote_access, memory_type));
    ASSERT_OK_AND_ASSIGN(
        uint32_t rkey_2,
        CreateRKeyForBuffer(setup.pd, setup.dst_buffer_2, setup.dst_qp,
                            dst_remote_access_2, memory_type));
    ibv_sge sge = verbs_util::CreateSge(setup.src_buffer.span(), src_mr);
    ibv_send_wr wr =
        verbs_util::CreateRdmaWr(opcode, /*wr_id=*/1, &sge, /*num_sge=*/1,
                                 setup.dst_buffer.data(), rkey);
    verbs_util::PostSend(setup.src_qp, wr);
    LOG(INFO) << "Operation done on first buffer, waiting for WC...";
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(setup.src_cq));
    EXPECT_EQ(completion.status, expected_status_1);
    EXPECT_EQ(completion.wr_id, wr.wr_id);
    EXPECT_EQ(completion.qp_num, setup.src_qp->qp_num);
    wr = verbs_util::CreateRdmaWr(opcode, /*wr_id=*/2, &sge, /*num_sge=*/1,
                                  setup.dst_buffer_2.data(), rkey_2);
    verbs_util::PostSend(setup.src_qp, wr);
    LOG(INFO) << "Operation done on second buffer, waiting for WC...";
    ASSERT_OK_AND_ASSIGN(completion,
                         verbs_util::WaitForCompletion(setup.src_cq));
    EXPECT_EQ(completion.status, expected_status_2);
    EXPECT_EQ(completion.wr_id, wr.wr_id);
    EXPECT_EQ(completion.qp_num, setup.src_qp->qp_num);
  }
};

TEST_P(RdmaAccessTest, AllAccess) {
  ExecuteTest(/*src_mr_access=*/kMrAccessAll,
              /*dst_remote_access=*/kRemoteAccessAll,
              /*expected_status=*/IBV_WC_SUCCESS);
}

TEST_P(RdmaAccessTest, MissingSrcLocalWrite) {
  ibv_wr_opcode opcode = std::get<1>(GetParam());
  ExecuteTest(/*src_mr_access=*/IBV_ACCESS_MW_BIND | IBV_ACCESS_REMOTE_READ,
              /*dst_remote_access=*/kRemoteAccessAll,
              /*expected_status=*/opcode == IBV_WR_RDMA_READ
                  ? IBV_WC_LOC_PROT_ERR
                  : IBV_WC_SUCCESS);
}

TEST_P(RdmaAccessTest, MissingDstRemoteWrite) {
  ibv_wr_opcode opcode = std::get<1>(GetParam());
  ExecuteTest(/*src_mr_access=*/kMrAccessAll,
              /*dst_remote_access=*/kRemoteAccessAll & ~IBV_ACCESS_REMOTE_WRITE,
              /*expected_status=*/opcode == IBV_WR_RDMA_WRITE
                  ? IBV_WC_REM_ACCESS_ERR
                  : IBV_WC_SUCCESS);
}

TEST_P(RdmaAccessTest, MissingDstRemoteRead) {
  ibv_wr_opcode opcode = std::get<1>(GetParam());
  ExecuteTest(/*src_mr_access=*/kMrAccessAll,
              /*dst_remote_access=*/kRemoteAccessAll & ~IBV_ACCESS_REMOTE_WRITE,
              /*expected_status=*/opcode == IBV_WR_RDMA_WRITE
                  ? IBV_WC_REM_ACCESS_ERR
                  : IBV_WC_SUCCESS);
}

TEST_P(RdmaAccessTest, MissingDstRemoteAtomic) {
  ExecuteTest(
      /*src_mr_access=*/kMrAccessAll,
      /*dst_remote_access=*/kRemoteAccessAll & ~IBV_ACCESS_REMOTE_ATOMIC,
      /*expected_status=*/IBV_WC_SUCCESS);
}

TEST_P(RdmaAccessTest, MinimalAccess) {
  ibv_wr_opcode opcode = std::get<1>(GetParam());
  ExecuteTest(opcode == IBV_WR_RDMA_READ ? IBV_ACCESS_LOCAL_WRITE : 0,
              opcode == IBV_WR_RDMA_READ
                  ? IBV_ACCESS_REMOTE_READ
                  : IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE,
              IBV_WC_SUCCESS);
}

TEST_P(RdmaAccessTest, TwoDstBufferValidAccess) {
  ExecuteTwoDstBufferTest(kMrAccessAll,
                          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
                          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
                          IBV_WC_SUCCESS, IBV_WC_SUCCESS);
}

TEST_P(RdmaAccessTest, TwoDstBufferMixedAccess) {
  constexpr static int kRemoteAccessMr1 =
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
  ExecuteTwoDstBufferTest(kMrAccessAll, kRemoteAccessMr1,
                          IBV_ACCESS_LOCAL_WRITE, IBV_WC_SUCCESS,
                          IBV_WC_REM_ACCESS_ERR);
}

INSTANTIATE_TEST_SUITE_P(
    Rdma, RdmaAccessTest,
    testing::Combine(testing::Values(RKeyMemmoryType::kMemoryRegion,
                                     RKeyMemmoryType::kMemoryWindowType1,
                                     RKeyMemmoryType::kMemoryWindowType2),
                     testing::Values(IBV_WR_RDMA_READ, IBV_WR_RDMA_WRITE)),
    [](const testing::TestParamInfo<RdmaAccessTest::ParamType>& info) {
      std::string memory_type = "";
      switch (std::get<0>(info.param)) {
        case RKeyMemmoryType::kMemoryRegion: {
          memory_type = "MemoryRegion";
          break;
        }
        case RKeyMemmoryType::kMemoryWindowType1: {
          memory_type = "MemoryWindowType1";
          break;
        }
        case RKeyMemmoryType::kMemoryWindowType2: {
          memory_type = "MemoryWindowType2";
          break;
        }
      }
      ibv_wr_opcode opcode = std::get<1>(info.param);
      std::string optype = "";
      switch (opcode) {
        case IBV_WR_RDMA_READ: {
          optype = "RdmaRead";
          break;
        }
        case IBV_WR_RDMA_WRITE: {
          optype = "RdmaWrite";
          break;
        }
        default: {
          optype = "Unsupported";
        }
      }
      return memory_type + optype;
    });

class AtomicAccessTest : public AccessTestFixture,
                         public testing::WithParamInterface<
                             std::tuple<RKeyMemmoryType, ibv_wr_opcode>> {
 protected:
  static constexpr uint64_t kCompareAdd = 0xADD;
  static constexpr uint64_t kSwap = 0xBEE;

  void SetUp() override {
    AccessTestFixture::SetUp();
    if (std::get<0>(GetParam()) == RKeyMemmoryType::kMemoryWindowType1 &&
        !Introspection().SupportsType1()) {
      GTEST_SKIP() << "NIC does not support type 1 MW.";
    }
    if (std::get<0>(GetParam()) == RKeyMemmoryType::kMemoryWindowType2 &&
        !Introspection().SupportsType2()) {
      GTEST_SKIP() << "NIC does not support type 2 MW.";
    }
  }

  void ExecuteTest(int src_mr_access, int dst_remote_access,
                   ibv_wc_status expected_status) {
    ASSERT_OK_AND_ASSIGN(BasicSetup setup, CreateBasicSetup());
    RKeyMemmoryType memory_type = std::get<0>(GetParam());
    ibv_wr_opcode opcode = std::get<1>(GetParam());
    ibv_mr* src_mr = ibv_.RegMr(setup.pd, setup.src_buffer, src_mr_access);
    ASSERT_THAT(src_mr, NotNull());
    ASSERT_OK_AND_ASSIGN(
        uint32_t rkey,
        CreateRKeyForBuffer(setup.pd, setup.dst_buffer, setup.dst_qp,
                            dst_remote_access, memory_type));
    ibv_sge sge = verbs_util::CreateAtomicSge(setup.src_buffer.data(), src_mr);
    ibv_send_wr wr = verbs_util::CreateAtomicWr(
        opcode, /*wr_id=*/1, &sge, /*num_sge=*/1, setup.dst_buffer.data(), rkey,
        kCompareAdd, kSwap);
    verbs_util::PostSend(setup.src_qp, wr);
    if (expected_status == IBV_WC_REM_ACCESS_ERR) {
      expected_status = Introspection().GeneratesRetryExcOnConnTimeout()
                            ? IBV_WC_RETRY_EXC_ERR
                            : expected_status;
    }
    ASSERT_OK_AND_ASSIGN(ibv_wc completion,
                         verbs_util::WaitForCompletion(setup.src_cq));
    EXPECT_EQ(completion.status, expected_status);
    EXPECT_EQ(completion.wr_id, wr.wr_id);
    EXPECT_EQ(completion.qp_num, setup.src_qp->qp_num);
  }
};

TEST_P(AtomicAccessTest, AllAccess) {
  ExecuteTest(/*src_mr_access=*/kMrAccessAll,
              /*dst_remote_access=*/kRemoteAccessAll,
              /*expected_status=*/IBV_WC_SUCCESS);
}

TEST_P(AtomicAccessTest, MissingSrcLocalWrite) {
  ExecuteTest(/*src_mr_access=*/IBV_ACCESS_MW_BIND | IBV_ACCESS_REMOTE_READ,
              /*dst_remote_access=*/kRemoteAccessAll,
              /*expected_status=*/IBV_WC_LOC_PROT_ERR);
}

TEST_P(AtomicAccessTest, MissingDstRemoteWrite) {
  ExecuteTest(/*src_mr_access=*/kMrAccessAll,
              /*dst_remote_access=*/kRemoteAccessAll & ~IBV_ACCESS_REMOTE_WRITE,
              /*expected_status=*/IBV_WC_SUCCESS);
}

TEST_P(AtomicAccessTest, MissingDstRemoteRead) {
  ExecuteTest(/*src_mr_access=*/kMrAccessAll,
              /*dst_remote_access=*/kRemoteAccessAll & ~IBV_ACCESS_REMOTE_READ,
              /*expected_status=*/IBV_WC_SUCCESS);
}

TEST_P(AtomicAccessTest, MissingDstRemoteAtomic) {
  ExecuteTest(
      /*src_mr_access=*/kMrAccessAll,
      /*dst_remote_access=*/kRemoteAccessAll & ~IBV_ACCESS_REMOTE_ATOMIC,
      /*expected_status=*/IBV_WC_REM_ACCESS_ERR);
}

INSTANTIATE_TEST_SUITE_P(
    Atomic, AtomicAccessTest,
    testing::Combine(testing::Values(RKeyMemmoryType::kMemoryRegion,
                                     RKeyMemmoryType::kMemoryWindowType1,
                                     RKeyMemmoryType::kMemoryWindowType2),
                     testing::Values(IBV_WR_ATOMIC_FETCH_AND_ADD,
                                     IBV_WR_ATOMIC_CMP_AND_SWP)),
    [](const testing::TestParamInfo<RdmaAccessTest::ParamType>& info) {
      std::string memory_type = "";
      switch (std::get<0>(info.param)) {
        case RKeyMemmoryType::kMemoryRegion: {
          memory_type = "MemoryRegion";
          break;
        }
        case RKeyMemmoryType::kMemoryWindowType1: {
          memory_type = "MemoryWindowType1";
          break;
        }
        case RKeyMemmoryType::kMemoryWindowType2: {
          memory_type = "MemoryWindowType2";
          break;
        }
      }
      ibv_wr_opcode opcode = std::get<1>(info.param);
      std::string optype = "";
      switch (opcode) {
        case IBV_WR_ATOMIC_FETCH_AND_ADD: {
          optype = "FetchAndAdd";
          break;
        }
        case IBV_WR_ATOMIC_CMP_AND_SWP: {
          optype = "CompareAndSwap";
          break;
        }
        default: {
          optype = "Unsupported";
        }
      }
      return memory_type + optype;
    });

}  // namespace
}  // namespace rdma_unit_test
