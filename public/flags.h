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

#ifndef THIRD_PARTY_RDMA_UNIT_TEST_PUBLIC_FLAGS_H_
#define THIRD_PARTY_RDMA_UNIT_TEST_PUBLIC_FLAGS_H_

#include <cstdint>
#include <string>

#include "absl/flags/declare.h"

ABSL_DECLARE_FLAG(bool, ipv4_only);
ABSL_DECLARE_FLAG(std::string, device_name);
ABSL_DECLARE_FLAG(uint64_t, completion_wait_multiplier);
ABSL_DECLARE_FLAG(uint32_t, port_num);
ABSL_DECLARE_FLAG(int, gid_index);
ABSL_DECLARE_FLAG(bool, peer_server);
ABSL_DECLARE_FLAG(bool, peer_client);
ABSL_DECLARE_FLAG(uint16_t, peer_port);
ABSL_DECLARE_FLAG(std::string, server_ip);

#endif  // THIRD_PARTY_RDMA_UNIT_TEST_PUBLIC_FLAGS_H_
