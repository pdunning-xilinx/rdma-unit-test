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
#include <netinet/in.h>

#include <cstdlib>
#include <thread>  // NOLINT
#include <vector>

#include "absl/status/statusor.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "infiniband/verbs.h"
#include "public/introspection.h"
#include "public/rdma_memblock.h"
#include "public/status_matchers.h"
#include "public/verbs_helper_suite.h"
#include "unit/rdma_verbs_fixture.h"

// ntoh for 64-bit value is not available in netinet/in.h
#define ntohll(x)  \
  ((1 == ntohl(1)) \
       ? (x)       \
       : ((uint64_t)ntohl((x)&0xFFFFFFFF) << 32) | ntohl((x) >> 32))

namespace rdma_unit_test {

using ::testing::NotNull;

class DeviceTest : public RdmaVerbsFixture {};

TEST_F(DeviceTest, GetDeviceList) {
  int num_devices = 0;
  ibv_device** devices = ibv_get_device_list(&num_devices);
  ASSERT_THAT(devices, NotNull());
  ibv_free_device_list(devices);

  devices = ibv_get_device_list(nullptr);
  ASSERT_THAT(devices, NotNull());
  ibv_free_device_list(devices);
}

TEST_F(DeviceTest, Open) {
  int num_devices = 0;
  ibv_device** devices = ibv_get_device_list(&num_devices);
  ASSERT_THAT(devices, NotNull());
  ASSERT_GE(num_devices, 1);

  for (int i = 0; i < num_devices; ++i) {
    ibv_device* device = devices[i];
    ASSERT_THAT(device, NotNull());
    LOG(INFO) << "Found device " << device->name << ".";
    ibv_context* context = ibv_open_device(device);
    ASSERT_THAT(context, NotNull());
    EXPECT_EQ(context->device, device);
    ASSERT_EQ(ibv_close_device(context), 0);
  }
  ibv_free_device_list(devices);
}

TEST_F(DeviceTest, GetDeviceGuid) {
  int num_devices = 0;
  ibv_device** devices = ibv_get_device_list(&num_devices);
  ASSERT_THAT(devices, NotNull());
  ASSERT_GE(num_devices, 1);

  for (int i = 0; i < num_devices; ++i) {
    ibv_device* device = devices[i];
    ASSERT_THAT(device, NotNull());
    uint64_t guid = (uint64_t)ntohll(ibv_get_device_guid(device));
    LOG(INFO) << "Device:" << device->name << " GUID:" << std::hex << guid;
    ASSERT_NE(guid, 0);
  }
  ibv_free_device_list(devices);
}

TEST_F(DeviceTest, OpenMany) {
  for (int i = 0; i < 100; ++i) {
    auto context = ibv_.OpenDevice();
    ASSERT_OK(context.status());
  }
}

TEST_F(DeviceTest, OpenInAnotherThread) {
  std::thread another_thread([this]() {
    auto context = ibv_.OpenDevice();
    EXPECT_OK(context.status());
  });
  another_thread.join();
}

TEST_F(DeviceTest, OpenInManyThreads) {
  std::vector<std::thread> threads;
  for (int i = 0; i < 100; i++) {
    threads.push_back(std::thread([this]() {
      auto context = ibv_.OpenDevice();
      EXPECT_OK(context.status());
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(DeviceTest, QueryDevice) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context, ibv_.OpenDevice());
  ibv_device_attr dev_attr = {};
  ASSERT_EQ(ibv_query_device(context, &dev_attr), 0);
  LOG(INFO) << "Device capabilities = " << std::hex
            << dev_attr.device_cap_flags;
}

TEST_F(DeviceTest, ContextTomfoolery) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context1, ibv_.OpenDevice());
  ASSERT_OK_AND_ASSIGN(ibv_context * context2, ibv_.OpenDevice());
  auto* pd = ibv_alloc_pd(context1);
  ASSERT_THAT(pd, NotNull());
  // Try to delete with the other context.
  pd->context = context2;
  ASSERT_EQ(ENOENT, ibv_dealloc_pd(pd));
  pd->context = context1;
  ASSERT_EQ(ibv_dealloc_pd(pd), 0);
}

// The fixture is used to test validity of resource limit on an ibv_device as
// stated in ibv_device_attr.
class DeviceLimitTest : public DeviceTest {
 protected:
  // In most NICs, the actual elements you can create in a resource might be
  // deviating from the indicated limit in ibv_device_attr. This constant
  // specifies the tolerable limits.
  static constexpr int kErrorMax = 200;
  // Bytes of memory to be allocated
  uint64_t big_mr_size = 0x400000;
};

TEST_F(DeviceLimitTest, MaxAh) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context, ibv_.OpenDevice());
  ibv_pd* pd = ibv_.AllocPd(context);
  ASSERT_THAT(pd, NotNull());
  PortAttribute port_attr = ibv_.GetPortAttribute(context);
  int max_ah = Introspection().device_attr().max_ah;
  int actual_max = 0;
  for (int i = 0; i < max_ah + kErrorMax + 10; ++i) {
    if (ibv_.CreateLoopbackAh(pd, port_attr) != nullptr) {
      ++actual_max;
    } else {
      break;
    }
  }
  LOG(INFO) << "max_ah = " << max_ah;
  LOG(INFO) << "max_ah (actual) = " << actual_max;
  ASSERT_GE(actual_max, 0);
  EXPECT_LE(std::abs(max_ah - actual_max), kErrorMax);
}

TEST_F(DeviceLimitTest, MaxCqMixed) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context, ibv_.OpenDevice());
  int max_cq = Introspection().device_attr().max_cq;
  int actual_max = 0;
  for (int i = 0; i < max_cq + kErrorMax + 10; ++i) {
    if (i & 1 || !Introspection().SupportsExtendedCqs()) {
      if (ibv_.CreateCq(context) != nullptr) {
        ++actual_max;
      } else {
        break;
      }
    } else {
      if (ibv_.CreateCqEx(context) != nullptr) {
        ++actual_max;
      } else {
        break;
      }
    }
  }
  LOG(INFO) << "max_cq = " << max_cq;
  LOG(INFO) << "max_cq (actual) = " << actual_max;
  EXPECT_LE(std::abs(max_cq - actual_max), kErrorMax);
}

TEST_F(DeviceLimitTest, MaxMr) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context, ibv_.OpenDevice());
  RdmaMemBlock buffer = ibv_.AllocBuffer(/*pages=*/1);
  ibv_pd* pd = ibv_.AllocPd(context);
  ASSERT_THAT(pd, NotNull());
  int max_mr = Introspection().device_attr().max_mr;
  int actual_max = 0;
  for (int i = 0; i < max_mr + kErrorMax + 10; ++i) {
    if (ibv_.RegMr(pd, buffer) != nullptr) {
      ++actual_max;
    } else {
      break;
    }
  }
  LOG(INFO) << "max_mr = " << max_mr;
  LOG(INFO) << "max_mr (actual) = " << actual_max;
  EXPECT_LE(std::abs(max_mr - actual_max), kErrorMax);
}

TEST_F(DeviceLimitTest, MaxMw) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context, ibv_.OpenDevice());
  ibv_pd* pd = ibv_.AllocPd(context);
  ASSERT_THAT(pd, NotNull());
  int max_mw = Introspection().device_attr().max_mw;
  int actual_max = 0;
  for (int i = 0; i < max_mw + kErrorMax + 10; ++i) {
    if (ibv_.AllocMw(pd, i & 1 ? IBV_MW_TYPE_1 : IBV_MW_TYPE_2) != nullptr) {
      ++actual_max;
    } else {
      break;
    }
  }
  LOG(INFO) << "max_mw = " << max_mw;
  LOG(INFO) << "max_mw (actual) = " << actual_max;
  EXPECT_LE(std::abs(max_mw - actual_max), kErrorMax);
}

TEST_F(DeviceLimitTest, MaxPd) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context, ibv_.OpenDevice());
  int max_pd = Introspection().device_attr().max_pd;
  int actual_max = 0;
  for (int i = 0; i < max_pd + kErrorMax + 10; ++i) {
    if (ibv_.AllocPd(context) != nullptr) {
      ++actual_max;
    } else {
      break;
    }
  }
  LOG(INFO) << "max_pd = " << max_pd;
  LOG(INFO) << "max_pd (actual) = " << actual_max;
  EXPECT_LE(std::abs(max_pd - actual_max), kErrorMax);
}

TEST_F(DeviceLimitTest, MaxQp) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context, ibv_.OpenDevice());
  ibv_cq* cq = ibv_.CreateCq(context);
  ASSERT_THAT(cq, NotNull());
  ibv_pd* pd = ibv_.AllocPd(context);
  ASSERT_THAT(pd, NotNull());
  int max_qp = Introspection().device_attr().max_qp;
  int actual_max = 0;
  for (int i = 0; i < 2 * max_qp; ++i) {
    if (ibv_.CreateQp(pd, cq) != nullptr) {
      ++actual_max;
    } else {
      break;
    }
  }
  LOG(INFO) << "max_qp = " << max_qp;
  LOG(INFO) << "max_qp (actual) = " << actual_max;
  EXPECT_LE(std::abs(max_qp - actual_max), kErrorMax);
}

TEST_F(DeviceLimitTest, MaxSrq) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context, ibv_.OpenDevice());
  ibv_pd* pd = ibv_.AllocPd(context);
  ASSERT_THAT(pd, NotNull());
  int max_srq = Introspection().device_attr().max_srq;
  int actual_max = 0;
  for (int i = 0; i < max_srq + kErrorMax + 10; ++i) {
    if (ibv_.CreateSrq(pd) != nullptr) {
      ++actual_max;
    } else {
      break;
    }
  }
  LOG(INFO) << "max_srq = " << max_srq;
  LOG(INFO) << "max_srq (actual) = " << actual_max;
  EXPECT_LE(std::abs(max_srq - actual_max), kErrorMax);
}

TEST_F(DeviceLimitTest, BigMrSize) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context, ibv_.OpenDevice());
  ibv_pd* pd = ibv_.AllocPd(context);
  ASSERT_THAT(pd, NotNull());
  RdmaMemBlock buffer = ibv_.AllocAlignedBufferByBytes(big_mr_size);
  ibv_mr* mr = ibv_.RegMr(pd, buffer);
  ASSERT_THAT(mr, NotNull());
}

TEST_F(DeviceLimitTest, MultiMrVaryMrSize) {
  ASSERT_OK_AND_ASSIGN(ibv_context * context, ibv_.OpenDevice());
  ibv_pd* pd = ibv_.AllocPd(context);
  ASSERT_THAT(pd, NotNull());
  uint32_t multi_mr = 10;
  uint32_t i;
  for (i = 1; i <= multi_mr; ++i) {
    ASSERT_LE((i * kPageSize), big_mr_size);
    RdmaMemBlock buffer = ibv_.AllocBuffer(/*page_size*/ i);
    ibv_mr* mr = ibv_.RegMr(pd, buffer);
    ASSERT_THAT(mr, NotNull());
  }
}
// TODO(author1): Create Max

}  // namespace rdma_unit_test
