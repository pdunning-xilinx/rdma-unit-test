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

#ifndef THIRD_PARTY_RDMA_UNIT_TEST_INTERNAL_INTROSPECTION_IONIC_H_
#define THIRD_PARTY_RDMA_UNIT_TEST_INTERNAL_INTROSPECTION_IONIC_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "infiniband/verbs.h"
#include "internal/introspection_registrar.h"
#include "public/introspection.h"

namespace rdma_unit_test {

// Concrete class to override specific behaviour for Ionic NIC.
class IntrospectionIonic : public NicIntrospection {
 public:
  // Register IONIC NIC with the Introspection Registrar.
  static void Register() {
    IntrospectionRegistrar::GetInstance().Register(
        "ionic", [](const std::string& name, const ibv_device_attr& attr) {
          return new IntrospectionIonic(name, attr);
        });
    IntrospectionRegistrar::GetInstance().Register(
        "roce[1dd8:", [](const std::string& name, const ibv_device_attr& attr) {
          return new IntrospectionIonic(name, attr);
        });
  }

 protected:
  const absl::flat_hash_map<TestcaseKey, std::string>& GetDeviations()
      const override {
    static const absl::flat_hash_map<TestcaseKey, std::string> deviations{};
    return deviations;
  }

 private:
  IntrospectionIonic() = delete;
  ~IntrospectionIonic() = default;
  explicit IntrospectionIonic(const std::string& name,
                              const ibv_device_attr& attr)
      : NicIntrospection(name, attr) {}
};

}  // namespace rdma_unit_test

#endif  // THIRD_PARTY_RDMA_UNIT_TEST_INTERNAL_INTROSPECTION_IONIC_H_
