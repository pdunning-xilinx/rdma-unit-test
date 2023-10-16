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

#ifndef THIRD_PARTY_RDMA_UNIT_TEST_INTERNAL_INTROSPECTION_XRNIC_H_
#define THIRD_PARTY_RDMA_UNIT_TEST_INTERNAL_INTROSPECTION_XRNIC_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "infiniband/verbs.h"
#include "internal/introspection_registrar.h"
#include "public/introspection.h"

namespace rdma_unit_test {

// Concrete class to override specific behaviour for XRNIC.
class IntrospectionXRnic : public NicIntrospection {
 public:
  // Register XRNIC with the Introspection Registrar.
  static void Register() {
    IntrospectionRegistrar::GetInstance().Register(
        "xrnic", [](const std::string& name, const ibv_device_attr& attr) {
	  return new IntrospectionXRnic(name, attr);
	});
  }

  bool SupportsIpV6() const override { return false; }

  bool SupportsUdQp() const override { return false; }

 protected:
  const absl::flat_hash_map<TestcaseKey, std::string>& GetDeviations()
      const override {
    static const absl::flat_hash_map<TestcaseKey, std::string> deviations{
      /* XRMOD-365 Model does not report errors to host */
      {{"MessagingAccessTest", "MissingDstLocalWrite"},
       "Model does not report errors to host currently"},
      {{"MessagingAccessTest", "TwoSgeMixedAccess"},
       "Model does not report errors to host currently"},
      {{"RdmaAccessTest", "MissingSrcLocalWrite"},
       "Model does not report errors to host currently"},
      {{"RdmaAccessTest", "MissingDstRemoteWrite"},
       "Model does not report errors to host currently"},
      {{"RdmaAccessTest", "MissingDstRemoteRead"},
       "Model does not report errors to host currently"},
      {{"RdmaAccessTest", "TwoDstBufferMixedAccess"},
       "Model does not report errors to host currently"},
      /* XRMOD-268 Model wont report errors to host */
      {{"PdRcLoopbackMrTest", "SendMrOtherPdLocal"}, ""},
      {{"PdRcLoopbackMrTest", "SendMrOtherPdRemote"}, ""},
      {{"PdRcLoopbackMrTest", "BasicReadMrOtherPdLocal"}, ""},
      {{"PdRcLoopbackMrTest", "SendMrOtherPdRemote"}, ""},
      {{"PdRcLoopbackMrTest", "BasicReadMrOtherPdRemote"}, ""},
      {{"PdRcLoopbackMrTest", "BasicWriteMrOtherPdLocal"}, ""},
      {{"PdRcLoopbackMrTest", "BasicWriteMrOtherPdRemote"}, ""},
      {{"PdRcLoopbackMrTest", "BasicFetchAddMrOtherPdLocal"}, ""},
      {{"PdRcLoopbackMrTest", "BasicFetchAddMrOtherPdRemote"}, ""},
      {{"PdRcLoopbackMrTest", "BasicCompSwapMrOtherPdLocal"}, ""},
      {{"PdRcLoopbackMrTest", "BasicCompSwapMrOtherPdRemote"}, ""},
      {{"BufferTest", "ReadMrExceedFront"}, ""},
      {{"BufferTest", "ReadMrExceedRear"}, ""},
      {{"BufferTest", "SendZeroByteOutsideMr"}, ""},
      {{"BufferTest", "BasicReadZeroByteOutsideMr"}, ""},
      {{"BufferTest", "BasicWriteZeroByteOutsideMr"}, ""},
      {{"BufferTest", "BasicReadZeroByteOutsideZeroByteMr"}, ""},
      {{"BufferTest", "BasicWriteZeroByteOutsideZeroByteMr"}, ""},
      {{"BufferTest", "ZeroByteReadInvalidRKey"}, ""},
      {{"BufferTest", "ZeroByteWriteInvalidRKey"}, ""},
      {{"CompChannelTest", "AcknowledgeWithoutOutstanding"}, ""},    /* Timeout error */
      {{"CompChannelTest", "AcknowledgeTooMany"}, ""},    /* Timeout error */

      /* XRMOD-352 Model does not have Solicited Support */
      {{"CompChannelTest", "RecvSolicitedNofityAny"}, ""},
      {{"CompChannelTest", "RecvSolicitedNofitySolicited"}, ""},
      {{"CompChannelTest", "RecvUnsolicitedNofityAny"}, ""},
      {{"CompChannelTest", "RecvUnsolicitedNofitySolicited"}, ""},
      {{"CompChannelTest", "DowngradeRequest"}, ""},
      {{"CompChannelTest", "UpgradeRequest"}, ""},

      /* XRFW-70 FW does not have SRQ support */
      {{"SrqPdTest", "CreateSrq"}, ""},
      {{"SrqPdTest", "SrqRecvMrSrqMatch"}, ""},
      {{"SrqPdTest", "SrqRecvMrSrqMismatch"}, ""},

      /* XRDRIV-253 Provider does not have AH support */
      {{"PdUdLoopbackTest", "SendAhOnOtherPd"}, ""},

      /* XRDRIV-270 Provider does not have Atomic support */
      {{"CompChannelTest", "Atomic"}, ""},    /* Timeout error */

      /* XRMOD-361 zero byte Send message */
      {{"BufferTest", "SendZeroByte"}, ""},
      {{"BufferTest", "SendZeroByteFromZeroByteMr"}, ""},
      {{"BufferTest", "BasicWriteZeroByte"}, ""},
      {{"BufferTest", "BasicWriteZeroByteToZeroByteMr"}, ""},

      /* XRMOD-371 NPT out of range */
      {{"BufferTest", "BasicReadZeroByteFromZeroByteMr"}, ""},

      /* XRMOD-371 Assert from pcieproxy */
      {{"BufferTest", "BasicReadZeroByte"}, ""},
 
      /* Hardware returns true when requesting notification on a CQ without a
       * Completion Channel. XRNIC-TODO: Explore more */
      {{"CompChannelTest", "RequestNoificationOnCqWithoutCompChannel"}, ""},

      /* Delete MR MCDI timeout */
      {{"MRLoopbackTest", "OutstandingWrite"}, ""},
      {{"MRLoopbackTest", "OutstandingRead"}, ""},

      /* XRDRIV-1198 Model does not support error reporting */
      {{"LoopbackRcQpTest", "BadWriteAddrLocal"}, ""}, /* Timeout while waiting for a completion */
      {{"LoopbackRcQpTest", "BadWriteAddrRemote"}, ""}, /* Timeout while waiting for a completion */

      /* XRDRIV-1199 Model does not support error reporting */
      {{"LoopbackRcQpTest", "BadReadAddrLocal"}, ""}, /* Timeout while waiting for a completion */
      {{"LoopbackRcQpTest", "BadReadAddrRemote"}, ""}, /* Timeout while waiting for a completion */

      /* XRDRIV-1139 Model does not support error reporting */
      {{"LoopbackRcQpTest", "SendBufferExceedMr"}, ""},  /* Timeout while waiting for a completion */
      {{"LoopbackRcQpTest", "BadSendAddr"}, ""},         /* Timeout while waiting for a completion */

      /* RNR is not supported */
      {{"RnrRecoverTest", "RnrRecoverTests"}, ""},

      /* XRDRIV-1192 Model does not support error reporting */
      /* Timeout while waiting for a completion */
      {{"LoopbackUdQpTest", "BadSendLkey"}, ""},
      {{"LoopbackUdQpTest", "BadRecvLkey"}, ""},
      {{"LoopbackUdQpTest", "BadSendAddr"}, ""},
      {{"LoopbackUdQpTest", "BadRecvAddr"}, ""},

      /* XRDRIV-1192 Inline data operation is not supported */
      {{"LoopbackUdQpTest", "SendInlineData"}, ""},

      /* IPv6 is not supported */
      {{"LoopbackUdQpTest", "SendFlowLabel"}, ""},
    };
    return deviations;
  }

 private:
  IntrospectionXRnic() = delete;
  ~IntrospectionXRnic() = default;
  explicit IntrospectionXRnic(const std::string& name, const ibv_device_attr& attr)
      : NicIntrospection(name, attr) {}
};

}  // namespace rdma_unit_test

#endif  // THIRD_PARTY_RDMA_UNIT_TEST_IMPL_INTROSPECTION_XRNIC_H_
