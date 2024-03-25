/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_RDMA_UNIT_TEST_PUBLIC_VERBS_UTIL_H_
#define THIRD_PARTY_RDMA_UNIT_TEST_PUBLIC_VERBS_UTIL_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "infiniband/verbs.h"


namespace rdma_unit_test {
namespace verbs_util {


//////////////////////////////////////////////////////////////////////////////
//                          Constants
//////////////////////////////////////////////////////////////////////////////

// Default Wr capacity for send queue, receive queue, completion queue and
// shared receive queue.
constexpr uint32_t kDefaultMaxWr = 200;
// Default Sge capacity for single Wr in send queue, receive queue, compleetion
// queue and shared receive queue.
constexpr uint32_t kDefaultMaxSge = 1;
// Default timeout waiting for completion.
constexpr absl::Duration kDefaultCompletionTimeout = absl::Seconds(2);
// Default timeout waiting for completion on a known qp error.
constexpr absl::Duration kDefaultErrorCompletionTimeout = absl::Seconds(2);
// Definition for IPv4 Loopback Address.
constexpr std::string_view kIpV4LoopbackAddress{"127.0.0.1"};
// Definition for IPv6 Loopback Address.
constexpr std::string_view kIpV6LoopbackAddress{"::1"};

//////////////////////////////////////////////////////////////////////////////
//                          Helper Functions
//////////////////////////////////////////////////////////////////////////////

// Converts an ibv_mtu to int.
int VerbsMtuToInt(ibv_mtu mtu);

int GetIpAddressType(const ibv_gid& gid);

// Converts an uint64_t mtu to a ibv_mtu object.
ibv_mtu ToVerbsMtu(uint64_t mtu);


// Enumerates the names of all the devices available for the host.
absl::StatusOr<std::vector<std::string>> EnumerateDeviceNames();

// Create a defaulted ibv_srq_attr.
ibv_srq_attr DefaultSrqAttr();

// Returns the defaulted ibv_srq_attr value.
ibv_srq_attr DefaultSrqAttr();

///////////////////////////////////////////////////////////////////////////////
//                           Verbs Utilities
// Useful helper functions to eliminate the tediousness of filling repetitive
// attributes and flags. Simplifies the verb API.
///////////////////////////////////////////////////////////////////////////////

// Returns the state of the QP.
ibv_qp_state GetQpState(ibv_qp* qp);

// Returns the capacity of the QP.
ibv_qp_cap GetQpCap(ibv_qp* qp);

// Creates a Scatter Gather
ibv_sge CreateSge(absl::Span<uint8_t> buffer, ibv_mr* mr);

// Creates an SGE for atomic operation. Addr must be 8-byte aligned.
ibv_sge CreateAtomicSge(void* addr, ibv_mr* mr);

// Creates a MW bind info struct with local buffer information.
ibv_mw_bind_info CreateMwBindInfo(absl::Span<uint8_t> buffer, ibv_mr* mr,
                                  int access = IBV_ACCESS_REMOTE_READ |
                                               IBV_ACCESS_REMOTE_WRITE |
                                               IBV_ACCESS_REMOTE_ATOMIC);

// Creates a WR for type 1 MW bind.
ibv_mw_bind CreateType1MwBindWr(uint64_t wr_id, absl::Span<uint8_t> buffer,
                                ibv_mr* mr,
                                int access = IBV_ACCESS_REMOTE_READ |
                                             IBV_ACCESS_REMOTE_WRITE |
                                             IBV_ACCESS_REMOTE_ATOMIC);

// Creates a WR for type 2 MW bind.
ibv_send_wr CreateType2BindWr(uint64_t wr_id, ibv_mw* mw,
                              const absl::Span<uint8_t> buffer, uint32_t rkey,
                              ibv_mr* mr,
                              int access = IBV_ACCESS_REMOTE_READ |
                                           IBV_ACCESS_REMOTE_WRITE |
                                           IBV_ACCESS_REMOTE_ATOMIC);

// Creates a WR for local invalidate.
ibv_send_wr CreateLocalInvalidateWr(uint64_t wr_id, uint32_t rkey);

// Creates a WR for send.
ibv_send_wr CreateSendWr(uint64_t wr_id, ibv_sge* sge, int num_sge);

// Creates a WR for send with invalidate.
ibv_send_wr CreateSendWithInvalidateWr(uint64_t wr_id, uint32_t rkey);

// Creates  a WR for recv.
ibv_recv_wr CreateRecvWr(uint64_t wr_id, ibv_sge* sge, int num_sge);

// Creates a RDMA work request.
ibv_send_wr CreateRdmaWr(ibv_wr_opcode opcode, uint64_t wr_id, ibv_sge* sge,
                         int num_sge, void* remote_addr, uint32_t rkey);

// Creates a WR for RDMA read.
ibv_send_wr CreateReadWr(uint64_t wr_id, ibv_sge* sge, int num_sge,
                         void* remote_buffer, uint32_t rkey);

// Creates a WR for RDMA write.
ibv_send_wr CreateWriteWr(uint64_t wr_id, ibv_sge* sge, int num_sge,
                          void* remote_buffer, uint32_t rkey);

// Create an Atomic work request.
ibv_send_wr CreateAtomicWr(ibv_wr_opcode opcode, uint64_t wr_id, ibv_sge* sge,
                           int num_sge, void* remote_buffer, uint32_t rkey,
                           uint64_t compare_add, uint64_t swap = 0);

// Create a WR for fetch and add.
ibv_send_wr CreateFetchAddWr(uint64_t wr_id, ibv_sge* sge, int num_sge,
                             void* remote_buffer, uint32_t rkey,
                             uint64_t compare_add);

// Creates a WR for compare and swap.
ibv_send_wr CreateCompSwapWr(uint64_t wr_id, ibv_sge* sge, int num_sge,
                             void* remote_buffer, uint32_t rkey,
                             uint64_t compare_add, uint64_t swap);

// Posts a type 1 MW bind WR to a queue pair.
void PostType1Bind(ibv_qp* qp, ibv_mw* mw, const ibv_mw_bind& bind_args);

// Posts a WR to send queue.
void PostSend(ibv_qp* qp, const ibv_send_wr& wr);

// Posts a WR to recv queue.
void PostRecv(ibv_qp* qp, const ibv_recv_wr& wr);

// Posts a WR to shared receive queue.
void PostSrqRecv(ibv_srq* srq, const ibv_recv_wr& wr);

// Polls for and returns a completion.
absl::StatusOr<ibv_wc> WaitForCompletion(
    ibv_cq* cq, absl::Duration timeout = kDefaultCompletionTimeout);

absl::Status WaitForPollingExtendedCompletion(
    ibv_cq_ex* cq, absl::Duration timeout = kDefaultCompletionTimeout);

absl::Status WaitForNextExtendedCompletion(
    ibv_cq_ex* cq, absl::Duration timeout = kDefaultCompletionTimeout);

bool CheckExtendedCompletionHasCapability(ibv_context* context,
                                          uint64_t wc_flag);

bool ExpectNoCompletion(
    ibv_cq* cq, absl::Duration timeout = kDefaultErrorCompletionTimeout);

bool ExpectNoExtendedCompletion(
    ibv_cq_ex* cq, absl::Duration timeout = kDefaultErrorCompletionTimeout);

void PrintCompletion(const ibv_wc& completion);

// Wait for an async event on a ibv_context.
absl::StatusOr<ibv_async_event> WaitForAsyncEvent(
    ibv_context* context, absl::Duration timeout = absl::Seconds(100));

// Executes a type 1 MW Bind operation synchronously.
absl::StatusOr<ibv_wc_status> ExecuteType1MwBind(
    ibv_qp* qp, ibv_mw* mw, absl::Span<uint8_t> buffer, ibv_mr* mr,
    int access = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
                 IBV_ACCESS_REMOTE_ATOMIC);

// Executes a type 2 MW Bind operation synchronnously.
absl::StatusOr<ibv_wc_status> ExecuteType2MwBind(
    ibv_qp* qp, ibv_mw* mw, absl::Span<uint8_t> buffer, uint32_t rkey,
    ibv_mr* mr,
    int access = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
                 IBV_ACCESS_REMOTE_ATOMIC);

// Executes a RDMA operation synchronnously.
absl::StatusOr<ibv_wc_status> ExecuteRdma(ibv_wr_opcode opcode, ibv_qp* qp,
                                          absl::Span<uint8_t> local_buffer,
                                          ibv_mr* local_mr, void* remote_buffer,
                                          uint32_t rkey);

// Executes a RDMA read operation synchronously.
absl::StatusOr<ibv_wc_status> ExecuteRdmaRead(ibv_qp* qp,
                                              absl::Span<uint8_t> local_buffer,
                                              ibv_mr* local_mr,
                                              void* remote_buffer,
                                              uint32_t rkey);

// Executes a RDMA write operation synchronously.
absl::StatusOr<ibv_wc_status> ExecuteRdmaWrite(ibv_qp* qp,
                                               absl::Span<uint8_t> local_buffer,
                                               ibv_mr* local_mr,
                                               void* remote_buffer,
                                               uint32_t rkey);

// Execute an atomic operation synchronnously.
absl::StatusOr<ibv_wc_status> ExecuteAtomic(
    ibv_wr_opcode opcode, ibv_qp* qp, void* local_buffer, ibv_mr* local_mr,
    void* remote_buffer, uint32_t rkey, uint64_t comp_add, uint64_t swap = 0);

// Executes a fetch and add operation synchronously.
absl::StatusOr<ibv_wc_status> ExecuteFetchAndAdd(ibv_qp* qp, void* local_buffer,
                                                 ibv_mr* local_mr,
                                                 void* remote_buffer,
                                                 uint32_t rkey,
                                                 uint64_t comp_add);

// Executes a compare and swap operation synchronously.
absl::StatusOr<ibv_wc_status> ExecuteCompareAndSwap(
    ibv_qp* qp, void* local_buffer, ibv_mr* local_mr, void* remote_buffer,
    uint32_t rkey, uint64_t comp_add, uint64_t swap);

// Execute a local invalidate op and return completion status.
absl::StatusOr<ibv_wc_status> ExecuteLocalInvalidate(ibv_qp* qp, uint32_t rkey);

// The return pair consists of first the send side completion status then the
// recv side completion status.
absl::StatusOr<std::pair<ibv_wc_status, ibv_wc_status>> ExecuteSendRecv(
    ibv_qp* src_qp, ibv_qp* dst_qp, absl::Span<uint8_t> src_buffer,
    ibv_mr* src_mr, absl::Span<uint8_t> dst_buffer, ibv_mr* dst_mr);

absl::StatusOr<std::pair<ibv_wc_status, ibv_wc_status>> TwoSgeSendRecv(
    ibv_qp* src_qp, ibv_qp* dst_qp, absl::Span<uint8_t> src_buffer,
    ibv_mr* src_mr, absl::Span<uint8_t> dst_buffer_1, ibv_mr* dst_mr_1,
    absl::Span<uint8_t> dst_buffer_2, ibv_mr* dst_mr_2);

// Opens device. If device_name is not empty, tries to open the first device
// with the name; else, tries to open devices[0]. Returns the context. This
// function, OpenUntrackedDevice(), is mainly used as an internal util function.
// VerbsAllocator::OpenDevice() is preferred for most end user calls.
absl::StatusOr<ibv_context*> OpenUntrackedDevice(const std::string device_name);

}  // namespace verbs_util
}  // namespace rdma_unit_test

#endif  // THIRD_PARTY_RDMA_UNIT_TEST_PUBLIC_VERBS_UTIL_H_
