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

#ifndef THIRD_PARTY_RDMA_UNIT_TEST_TRAFFIC_RDMA_STRESS_FIXTURE_H_
#define THIRD_PARTY_RDMA_UNIT_TEST_TRAFFIC_RDMA_STRESS_FIXTURE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "infiniband/verbs.h"
#include "internal/verbs_attribute.h"
#include "public/basic_fixture.h"
#include "public/verbs_helper_suite.h"
#include "traffic/client.h"
#include "traffic/latency_measurement.h"
#include "traffic/transport_validation.h"

namespace rdma_unit_test {

// Parent fixture for all RDMA datapath tests. Provides functions to set up
// queue pairs acrosss multiple clients, execute ops, poll completions and
// verify statistics.
class RdmaStressFixture : public BasicFixture {
 public:
  static constexpr uint64_t kAtomicWordSize = 8;

  RdmaStressFixture();
  ~RdmaStressFixture() override = default;

  // Creates a one way connection from local qp to remote qp. A separate call to
  // this function is required if you need to setup connection in the reverse
  // direction.
  absl::Status SetUpRcClientsQPs(Client* local, uint32_t local_qp_id,
                                 Client* remote, uint32_t remote_qp_id,
                                 QpAttribute qp_attr = QpAttribute());

  // Creates number of qps_per_client RC qps for each client and connects pairs
  // across the two clients.
  void CreateSetUpRcQps(Client& initiator, Client& target,
                        uint16_t qps_per_client,
                        QpAttribute qp_attr = QpAttribute());

  // Creates number of qps_per_client RC qps for each client and connects pairs
  // across the two clients.
  void CreateSetUpMrcQps(Client& initiator, Client& target,
                        uint16_t qps_per_client,
                        QpAttribute qp_attr = QpAttribute());

  // Indicates how address handles should be assigned to queue pairs.
  enum class AddressHandleMapping {
    kIndependent,  // Each src:dst pair has its own address handle.
    kShared,  // A single address handle is shared amongst all src:dst pairs.
  };

  // Creates `qps_per_client` UD qps at both the initiator and the target
  // client, and allocates a separate AddressHandle to post ops from one
  // initiator QP to one target QP, with one-to-one mapping.
  void CreateSetUpOneToOneUdQps(Client& initiator, Client& target,
                                uint16_t qps_per_client);

  // Creates `qps_per_client` UD qps at both the initiator and the target
  // client, and allocates AddressHandles to send from any initiator QP to any
  // target QP using independent or shared AHs based on `ah_mapping`.
  void CreateSetUpMultiplexedUdQps(Client& initiator, Client& target,
                                   uint16_t initiator_qps, uint16_t target_qps,
                                   AddressHandleMapping ah_mapping);

  // Halt execution of ops by:
  // 1. Dumps the pending ops.
  // 2. Check whether all async events have completed.
  void HaltExecution(Client& client);

  // Best effort attempt to poll async event queue and log results to stderr for
  // all contexts.
  absl::Status PollAndAckAsyncEvents();

  // Best effort attempt to poll async event queue and log results to stderr for
  // a specific context. Polls the event queue once in non-blocking mode (the
  // fixture's constructor sets up the non-blocking mode) and acknowledges the
  // events polled. Returns OkStatus if no events polled, returns Internal error
  // if an event is polled and acknowledged, returns error if it fails on poll
  // or event calls.
  absl::Status PollAndAckAsyncEvents(ibv_context* context);

  // Configures latencies measurement parameters for the given RDMA operations.
  void ConfigureLatencyMeasurements(OpTypes op_type);

  // Collects latencies statistics for a a given client.
  void CollectClientLatencyStats(const Client& client);

  // Makes sure that the latency measurements in each stats set are within
  // a certain percentage of one another.
  void CheckLatencies();

  // Limits the number of total ops issued in a test based on op_size.
  static int LimitNumOps(int op_size, int num_ops);

  // Returns true if the test checks whether retransmission happens based on
  // number of queue pairs.
  bool AllowRetxCheck(int num_qp);

  // Return the context of index. Used for supporting running multiple
  // functions in one test. Return nullptr if index is not valid.
  ibv_context* context(uint index = 0) const {
    if (index >= contexts_.size()) return nullptr;
    return contexts_.at(index);
  }

  // Return the port_attr of index. Used for supporting running multiple
  // functions in one test. Return empty PortAttribute is index is not valid.
  PortAttribute port_attr(uint index = 0) const {
    if (index >= port_attrs_.size()) return PortAttribute();
    return port_attrs_.at(index);
  }

 protected:
  std::vector<ibv_context*> contexts_ = {};
  std::vector<PortAttribute> port_attrs_ = {};
  std::unique_ptr<TransportValidation> validation_ = nullptr;
  std::unique_ptr<LatencyMeasurement> latency_measure_;
  VerbsHelperSuite ibv_;
};

}  // namespace rdma_unit_test

#endif  // THIRD_PARTY_RDMA_UNIT_TEST_TRAFFIC_RDMA_STRESS_FIXTURE_H_
