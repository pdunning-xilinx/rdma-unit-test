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

#include "public/verbs_util.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <resolv.h>
#include <sys/poll.h>
#include <sys/socket.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include <magic_enum.hpp>
#include "infiniband/verbs.h"
#include "public/flags.h"
#include "public/status_matchers.h"

namespace rdma_unit_test {
namespace verbs_util {

int VerbsMtuToInt(ibv_mtu mtu) {
  // The enum ibv_mtu use value 1 to 5 for IBV_MTU_256 to IBV_MTU_4096.
  return 128 << mtu;
}

// Determines whether the gid is a valid ipv4 or ipv6 ip address.
// Returns AF_INET if ipv4.
// Returns AF_INET6 if ipv6.
// Reterns -1 if an invalid ip address.
int GetIpAddressType(const ibv_gid& gid) {
  char ip_str[INET6_ADDRSTRLEN];
  const char* result = inet_ntop(
      AF_INET6, reinterpret_cast<const char*>(gid.raw), ip_str, sizeof(ip_str));
  if (result == nullptr) return -1;

  const in6_addr* addr6 = reinterpret_cast<const in6_addr*>(gid.raw);
  if (addr6->s6_addr32[0] != 0 || addr6->s6_addr32[1] != 0 ||
      addr6->s6_addr16[4] != 0 || addr6->s6_addr16[5] != 0xffff) {
    return AF_INET6;
  }
  return AF_INET;
}

void PrintIpHeader(void* ip_hdr, int ip_family) {
  uint8_t header_size;
  if (ip_family == AF_INET) {
    header_size = 20;
    std::cout << "IPv4 header:" << std::endl;
  } else {
    header_size = 40;
    std::cout << "IPv6 header:" << std::endl;
  }

  std::cout << std::hex << std::setfill('0');
  auto* ptr = reinterpret_cast<unsigned char*>(ip_hdr);
  for (uint8_t i = 0; i < header_size; i++, ptr++) {
    if (i && (i % 4 == 0)) {
      std::cout << std::endl;
    }
    std::cout << std::setw(2) << static_cast<unsigned>(*ptr) << " ";
  }
  std::cout << std::endl;
}

std::string GidToString(const ibv_gid& gid) {
  return absl::StrFormat("GID: %x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x",
                         gid.raw[0], gid.raw[1], gid.raw[2], gid.raw[3],
                         gid.raw[4], gid.raw[5], gid.raw[6], gid.raw[7],
                         gid.raw[8], gid.raw[9], gid.raw[10], gid.raw[11],
                         gid.raw[12], gid.raw[13], gid.raw[14], gid.raw[15]);
}

absl::StatusOr<std::vector<std::string>> EnumerateDeviceNames() {
  ibv_device** devices = nullptr;
  absl::Cleanup free_list = [&devices]() {
    if (devices) {
      ibv_free_device_list(devices);
    }
  };
  int num_devices = 0;
  devices = ibv_get_device_list(&num_devices);
  std::vector<std::string> device_names;
  if (num_devices <= 0 || !devices) {
    return device_names;
  }
  for (int i = 0; i < num_devices; ++i) {
    ibv_device* device = devices[i];
    VLOG(1) << "Found device " << device->name << ".";
    device_names.push_back(device->name);
  }
  return device_names;
}

ibv_srq_attr DefaultSrqAttr() {
  ibv_srq_attr attr;
  attr.max_wr = verbs_util::kDefaultMaxWr;
  attr.max_sge = verbs_util::kDefaultMaxSge;
  attr.srq_limit = 0;  // not used for infiniband.
  return attr;
}

ibv_qp_state GetQpState(ibv_qp* qp) {
  ibv_qp_attr attr;
  ibv_qp_init_attr init_attr;
  int result = ibv_query_qp(qp, &attr, IBV_QP_STATE, &init_attr);
  DCHECK_EQ(0, result);
  return attr.qp_state;
}

ibv_qp_cap GetQpCap(ibv_qp* qp) {
  ibv_qp_attr attr;
  ibv_qp_init_attr init_attr;
  int result = ibv_query_qp(qp, &attr, IBV_QP_CAP, &init_attr);
  DCHECK_EQ(0, result);
  return attr.cap;
}

ibv_wc_opcode WrToWcOpcode(ibv_wr_opcode opcode) {
  switch (opcode) {
    case IBV_WR_RDMA_WRITE:
    case IBV_WR_RDMA_WRITE_WITH_IMM:
      return IBV_WC_RDMA_WRITE;
    case IBV_WR_SEND:
    case IBV_WR_SEND_WITH_INV:
    case IBV_WR_SEND_WITH_IMM:
      return IBV_WC_SEND;
    case IBV_WR_RDMA_READ:
      return IBV_WC_RDMA_READ;
    case IBV_WR_ATOMIC_CMP_AND_SWP:
      return IBV_WC_COMP_SWAP;
    case IBV_WR_ATOMIC_FETCH_AND_ADD:
      return IBV_WC_FETCH_ADD;
    case IBV_WR_LOCAL_INV:
      return IBV_WC_LOCAL_INV;
    case IBV_WR_BIND_MW:
      return IBV_WC_BIND_MW;
    default:
      LOG(DFATAL) << "Unsupported opcode " << static_cast<int>(opcode);
      return static_cast<ibv_wc_opcode>(0xff);
  }
}

ibv_sge CreateSge(absl::Span<uint8_t> buffer, ibv_mr* mr) {
  ibv_sge sge;
  sge.addr = reinterpret_cast<uint64_t>(buffer.data());
  sge.length = buffer.length();
  sge.lkey = mr->lkey;
  return sge;
}

ibv_sge CreateAtomicSge(void* addr, ibv_mr* mr) {
  DCHECK_EQ(reinterpret_cast<uint64_t>(addr) % 8, 0ul)
      << "Address is not 8 byte aligned.";
  return CreateSge(absl::MakeSpan(reinterpret_cast<uint8_t*>(addr), 8), mr);
}

ibv_mw_bind_info CreateMwBindInfo(absl::Span<uint8_t> buffer, ibv_mr* mr,
                                  int access) {
  ibv_mw_bind_info info;
  info.addr = reinterpret_cast<uint64_t>(buffer.data());
  info.length = buffer.length();
  info.mr = mr;
  info.mw_access_flags = access;
  return info;
}

ibv_mw_bind CreateType1MwBindWr(uint64_t wr_id, absl::Span<uint8_t> buffer,
                                ibv_mr* mr, int access) {
  ibv_mw_bind bind;
  bind.wr_id = wr_id;
  bind.send_flags = IBV_SEND_SIGNALED;
  bind.bind_info = CreateMwBindInfo(buffer, mr, access);
  return bind;
}

ibv_send_wr CreateType2BindWr(uint64_t wr_id, ibv_mw* mw,
                              const absl::Span<uint8_t> buffer, uint32_t rkey,
                              ibv_mr* mr, int access) {
  ibv_send_wr bind;
  bind.wr_id = wr_id;
  bind.next = nullptr;
  bind.sg_list = nullptr;
  bind.num_sge = 0;
  bind.opcode = IBV_WR_BIND_MW;
  bind.send_flags = IBV_SEND_SIGNALED;
  bind.bind_mw.mw = mw;
  bind.bind_mw.rkey = rkey;
  bind.bind_mw.bind_info = CreateMwBindInfo(buffer, mr, access);
  return bind;
}

ibv_send_wr CreateLocalInvalidateWr(uint64_t wr_id, uint32_t rkey) {
  ibv_send_wr invalidate;
  invalidate.wr_id = wr_id;
  invalidate.next = nullptr;
  invalidate.sg_list = nullptr;
  invalidate.num_sge = 0;
  invalidate.opcode = IBV_WR_LOCAL_INV;
  invalidate.send_flags = IBV_SEND_SIGNALED;
  invalidate.invalidate_rkey = rkey;
  return invalidate;
}

ibv_send_wr CreateSendWr(uint64_t wr_id, ibv_sge* sge, int num_sge) {
  ibv_send_wr send;
  send.wr_id = wr_id;
  send.next = nullptr;
  send.sg_list = sge;
  send.num_sge = num_sge;
  send.opcode = IBV_WR_SEND;
  send.send_flags = IBV_SEND_SIGNALED;
  return send;
}

ibv_send_wr CreateSendWithInvalidateWr(uint64_t wr_id, uint32_t rkey) {
  ibv_send_wr inv;
  inv.wr_id = wr_id;
  inv.next = nullptr;
  inv.sg_list = nullptr;
  inv.num_sge = 0;
  inv.opcode = IBV_WR_SEND_WITH_INV;
  inv.invalidate_rkey = rkey;
  inv.send_flags = IBV_SEND_SIGNALED;
  return inv;
}

// Only opcode is set in this, in test case explicitly set the imm_data of WQE.
ibv_send_wr CreateSendWithImmWr(uint64_t wr_id, ibv_sge* sge, int num_sge) {
  ibv_send_wr send;
  send.wr_id = wr_id;
  send.next = nullptr;
  send.sg_list = sge;
  send.num_sge = num_sge;
  send.opcode = IBV_WR_SEND_WITH_IMM;
  send.send_flags = IBV_SEND_SIGNALED;
  return send;
}

ibv_recv_wr CreateRecvWr(uint64_t wr_id, ibv_sge* sge, int num_sge) {
  ibv_recv_wr recv;
  recv.wr_id = wr_id;
  recv.next = nullptr;
  recv.sg_list = sge;
  recv.num_sge = num_sge;
  return recv;
}

ibv_send_wr CreateRdmaWr(ibv_wr_opcode opcode, uint64_t wr_id, ibv_sge* sge,
                         int num_sge, void* remote_addr, uint32_t rkey) {
  DCHECK(opcode == IBV_WR_RDMA_READ || opcode == IBV_WR_RDMA_WRITE ||
         opcode == IBV_WR_RDMA_WRITE_WITH_IMM)
      << "Opcode " << static_cast<int>(opcode) << "is not RDMA.";
  return ibv_send_wr{
      .wr_id = wr_id,
      .next = nullptr,
      .sg_list = sge,
      .num_sge = num_sge,
      .opcode = opcode,
      .send_flags = IBV_SEND_SIGNALED,
      .wr{.rdma{.remote_addr = reinterpret_cast<uint64_t>(remote_addr),
                .rkey = rkey}}};
}

ibv_send_wr CreateReadWr(uint64_t wr_id, ibv_sge* sge, int num_sge,
                         void* remote_buffer, uint32_t rkey) {
  return CreateRdmaWr(IBV_WR_RDMA_READ, wr_id, sge, num_sge, remote_buffer,
                      rkey);
}

ibv_send_wr CreateWriteWr(uint64_t wr_id, ibv_sge* sge, int num_sge,
                          void* remote_buffer, uint32_t rkey) {
  return CreateRdmaWr(IBV_WR_RDMA_WRITE, wr_id, sge, num_sge, remote_buffer,
                      rkey);
}

ibv_send_wr CreateWriteWithImmWr(uint64_t wr_id, ibv_sge* sge, int num_sge,
                                 void* remote_buffer, uint32_t rkey) {
  return CreateRdmaWr(IBV_WR_RDMA_WRITE_WITH_IMM, wr_id, sge, num_sge,
                      remote_buffer, rkey);
}

ibv_send_wr CreateAtomicWr(ibv_wr_opcode opcode, uint64_t wr_id, ibv_sge* sge,
                           int num_sge, void* remote_buffer, uint32_t rkey,
                           uint64_t compare_add, uint64_t swap) {
  DCHECK(opcode == IBV_WR_ATOMIC_FETCH_AND_ADD ||
         opcode == IBV_WR_ATOMIC_CMP_AND_SWP)
      << "Opcode " << static_cast<int>(opcode) << " is not atomic.";
  return ibv_send_wr{
      .wr_id = wr_id,
      .next = nullptr,
      // TODO(author2): IBTA Spec table 100 atomics  does not carry scatter
      // gather list. The sg_list field is used to denote a local address to
      // write the return value.
      .sg_list = sge,
      .num_sge = num_sge,
      .opcode = opcode,
      .send_flags = IBV_SEND_SIGNALED,
      .wr{.atomic{
          .remote_addr = reinterpret_cast<uint64_t>(remote_buffer),
          .compare_add = compare_add,
          .swap = swap,
          .rkey = rkey,
      }}};
}

ibv_send_wr CreateFetchAddWr(uint64_t wr_id, ibv_sge* sge, int num_sge,
                             void* remote_buffer, uint32_t rkey,
                             uint64_t compare_add) {
  return CreateAtomicWr(IBV_WR_ATOMIC_FETCH_AND_ADD, wr_id, sge, num_sge,
                        remote_buffer, rkey, compare_add);
}

ibv_send_wr CreateCompSwapWr(uint64_t wr_id, ibv_sge* sge, int num_sge,
                             void* remote_buffer, uint32_t rkey,
                             uint64_t compare_add, uint64_t swap) {
  return CreateAtomicWr(IBV_WR_ATOMIC_CMP_AND_SWP, wr_id, sge, num_sge,
                        remote_buffer, rkey, compare_add, swap);
}

void PostType1Bind(ibv_qp* qp, ibv_mw* mw, const ibv_mw_bind& bind_args) {
  int result = ibv_bind_mw(qp, mw, const_cast<ibv_mw_bind*>(&bind_args));
  ASSERT_EQ(0, result);
}

void PostSend(ibv_qp* qp, const ibv_send_wr& wr) {
  ibv_send_wr* bad_wr = nullptr;
  int result = ibv_post_send(qp, const_cast<ibv_send_wr*>(&wr), &bad_wr);
  ASSERT_EQ(0, result);
}

void PostRecv(ibv_qp* qp, const ibv_recv_wr& wr) {
  ibv_recv_wr* bad_wr = nullptr;
  int result = ibv_post_recv(qp, const_cast<ibv_recv_wr*>(&wr), &bad_wr);
  ASSERT_EQ(0, result);
}

void PostSrqRecv(ibv_srq* srq, const ibv_recv_wr& wr) {
  ibv_recv_wr* bad_wr = nullptr;
  int result = ibv_post_srq_recv(srq, const_cast<ibv_recv_wr*>(&wr), &bad_wr);
  ASSERT_EQ(0, result);
}

bool is_client() { return absl::GetFlag(FLAGS_peer_client) ? true : false; }

bool is_server() { return absl::GetFlag(FLAGS_peer_server) ? true : false; }

bool peer_mode() { return (is_client() || is_server()) ? true : false; }

int connect_peer(struct ibv_qp* qp, int local_psn,
                 struct conn_attr& remote_host, uint8_t port, int sgid_index) {
  // Attributes for bringing QP from RESET to INIT state.
  const unsigned int qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                                       IBV_ACCESS_REMOTE_READ |
                                       IBV_ACCESS_REMOTE_ATOMIC;
  const uint16_t pkey_index = 0;

  // Attributes for bringing QP from INIT to RTR state.
  const uint8_t min_rnr_timer = 26;  // 81.92 milliseconds

  // Attributes for bringing QP from RTR to RTS state.
  const uint8_t timeout = 0;  // infinite timeout
  const uint8_t retry_cnt = 5;
  const uint8_t rnr_retry = 5;
  const uint8_t max_rd_atomic = 10;
  const uint8_t sl = 5;
  const uint8_t hop_limit = 127;
  int rc;

  // Initialize QP state
  struct ibv_qp_attr attr = {.qp_state = IBV_QPS_INIT,
                             .qp_access_flags = qp_access_flags,
                             .pkey_index = pkey_index,
                             .port_num = port};

  rc = ibv_modify_qp(qp, &attr, RESET_TO_INIT_MASK);
  if (rc) {
    LOG(ERROR) << "Failed to modify QP to INIT, rc:" << rc;
    return rc;
  }

  // Change QP state to Ready To Receive state
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_1024;
  attr.rq_psn = remote_host.psn;
  attr.dest_qp_num = remote_host.qpn;
  attr.ah_attr.dlid = remote_host.lid;
  attr.ah_attr.sl = sl;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.is_global = 0;
  attr.ah_attr.port_num = port;
  attr.max_dest_rd_atomic = max_rd_atomic;
  attr.min_rnr_timer = min_rnr_timer;

  if (remote_host.gid.global.interface_id) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.hop_limit = hop_limit;
    attr.ah_attr.grh.dgid = remote_host.gid;
    attr.ah_attr.grh.sgid_index = sgid_index;
  }

  rc = ibv_modify_qp(qp, &attr, INIT_TO_RTR_MASK);
  if (rc) {
    LOG(ERROR) << "Failed to modify QP to RTR, rc:" << rc;
    return rc;
  }

  // Change QP state to Ready To Send state
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = timeout;
  attr.retry_cnt = retry_cnt;
  attr.rnr_retry = rnr_retry;
  attr.sq_psn = local_psn;
  attr.max_rd_atomic = max_rd_atomic;
  rc = ibv_modify_qp(qp, &attr, RTR_TO_RTS_MASK);
  if (rc) LOG(ERROR) << "Failed to modify QP to RTS, rc:" << rc;
  return rc;
}

// Convert the received gid to the CPUs native byte order
static void wire_gid_to_gid(const char* wgid, union ibv_gid* gid) {
  uint32_t tmp_gid[4];
  char tmp[9];
  __be32 v32;
  int i;

  for (tmp[8] = 0, i = 0; i < 4; ++i) {
    memcpy(tmp, wgid + i * 8, 8);
    sscanf(tmp, "%x", &v32);
    tmp_gid[i] = be32toh(v32);
  }
  memcpy(gid, tmp_gid, sizeof(*gid));
}

// Convert the local gid to big endian byte order
static void gid_to_wire_gid(const union ibv_gid* gid, char wgid[]) {
  uint32_t tmp_gid[4];
  int i;

  memcpy(tmp_gid, gid, sizeof(tmp_gid));
  for (i = 0; i < 4; ++i) sprintf(&wgid[i * 8], "%08x", htobe32(tmp_gid[i]));
}

int RunServer(const struct conn_attr& local_host,
              struct conn_attr& remote_host) {
  // Message to share the connection attributes with Client
  char msg[sizeof "0000:000000:000000:00000000000000000000000000000000"];
  struct sockaddr_in address;
  int addrlen = sizeof(address);
  int backlog = 0;
  int new_socket;
  int server_fd;
  char gid[33];
  int opt = 1;
  int rc;

  // Creating socket file descriptor
  server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd == -1) {
    LOG(FATAL) << "Server: Socket creation failed";
  }

  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                 sizeof(opt))) {
    LOG(FATAL) << "Server: Setsockopt Failed";
  }

  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(absl::GetFlag(FLAGS_peer_port));

  if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
    shutdown(server_fd, SHUT_RDWR);
    LOG(FATAL) << "Server: Bind failed";
  }

  if (listen(server_fd, backlog) < 0) {
    shutdown(server_fd, SHUT_RDWR);
    LOG(FATAL) << "Server: Listen failed";
  }

  if ((new_socket = accept(server_fd, (struct sockaddr*)&address,
                           (socklen_t*)&addrlen)) < 0) {
    shutdown(server_fd, SHUT_RDWR);
    LOG(FATAL) << "Server: Accept failed";
  }

  if ((rc = read(new_socket, msg, sizeof msg)) != sizeof msg) {
    LOG(ERROR) << "Server: Couldn't read remote address";
    if (rc == -1) {
      LOG(ERROR) << "Server: Read failed, error:" << errno;
    } else {
      LOG(ERROR) << "Server: Read returned " << rc << " bytes";
    }
    goto out;
  }

  // storing received message to remote conn attributes
  sscanf(msg, "%hu:%x:%x:%s", &remote_host.lid, &remote_host.qpn,
         &remote_host.psn, gid);
  wire_gid_to_gid(gid, &remote_host.gid);

  // preparing message to send local conn attributes
  gid_to_wire_gid(&local_host.gid, gid);
  sprintf(msg, "%04x:%06x:%06x:%s", local_host.lid, local_host.qpn,
          local_host.psn, gid);

  if ((rc = write(new_socket, msg, sizeof msg)) != sizeof msg) {
    LOG(ERROR) << "Server: Couldn't send local address";
    if (rc == -1) {
      LOG(ERROR) << "Server: Write failed, error:" << errno;
    } else {
      LOG(ERROR) << "Server: Write returned " << rc << " bytes";
    }
    goto out;
  }
  rc = 0;

out:
  // closing the connected socket
  close(new_socket);
  // closing the listening socket
  shutdown(server_fd, SHUT_RDWR);
  return rc;
}

int RunClient(const struct conn_attr& local_host,
              struct conn_attr& remote_host) {
  // Message to share the connection attributes with Server
  char msg[sizeof "0000:000000:000000:00000000000000000000000000000000"];
  struct sockaddr_in serv_addr;
  int client_fd;
  char gid[33];
  int rc;

  // Code for Client Connection
  client_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (client_fd == -1) {
    LOG(FATAL) << "Client: Socket creation failed";
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(absl::GetFlag(FLAGS_peer_port));

  // Convert IPv4 address from Presentation (string) to Numeric format (binary)
  if (inet_pton(AF_INET, absl::GetFlag(FLAGS_server_ip).c_str(),
                &serv_addr.sin_addr) <= 0) {
    LOG(ERROR) << "Client: Invalid address/ Address not supported";
    goto out;
  }

  rc = connect(client_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
  if (rc) {
    LOG(ERROR) << "Client: Connect failed, rc:" << rc;
    goto out;
  }

  // preparing message to send local conn attributes
  gid_to_wire_gid(&local_host.gid, gid);
  sprintf(msg, "%04x:%06x:%06x:%s", local_host.lid, local_host.qpn,
          local_host.psn, gid);

  if ((rc = write(client_fd, msg, sizeof msg)) != sizeof msg) {
    LOG(ERROR) << "Client: Couldn't send local address";
    if (rc == -1) {
      LOG(ERROR) << "Client: Write failed, error:" << errno;
    } else {
      LOG(ERROR) << "Client: Write returned " << rc << " bytes";
    }
    goto out;
  }

  if ((rc = read(client_fd, msg, sizeof msg)) != sizeof msg) {
    LOG(ERROR) << "Couldn't read remote address";
    if (rc == -1) {
      LOG(ERROR) << "Client: Read failed, error:" << errno;
    } else {
      LOG(ERROR) << "Client: Read returned " << rc << " bytes";
    }
    goto out;
  }
  rc = 0;

  // storing received message to remote conn attributes
  sscanf(msg, "%hu:%x:%x:%s", &remote_host.lid, &remote_host.qpn,
         &remote_host.psn, gid);
  wire_gid_to_gid(gid, &remote_host.gid);

out:
  // closing the connected socket
  close(client_fd);
  return rc;
}

absl::Duration GetSlowDownTimeout(absl::Duration timeout, uint64_t multiplier) {
  if (!multiplier) {
    LOG(ERROR) << "completion_wait_multiplier should be a positive value";
    multiplier = 1;
  } else if (multiplier > 1) {
    LOG(INFO) << "Excepted timeout: " << timeout
              << ", multiplier: " << multiplier;
  }
  return timeout * multiplier;
}

absl::StatusOr<ibv_wc> WaitForCompletion(ibv_cq* cq, absl::Duration timeout) {
  ibv_wc result;
  absl::Time stop =
      absl::Now() +
      GetSlowDownTimeout(timeout,
                         absl::GetFlag(FLAGS_completion_wait_multiplier));
  int count = ibv_poll_cq(cq, 1, &result);
  while (count == 0 && absl::Now() < stop) {
    absl::SleepFor(absl::Milliseconds(10));
    count = ibv_poll_cq(cq, 1, &result);
  }
  if (count > 0) {
    return result;
  }
  return absl::DeadlineExceededError("Timeout while waiting for a completion.");
}

absl::Status WaitForPollingExtendedCompletion(ibv_cq_ex* cq,
                                              absl::Duration timeout) {
  ibv_poll_cq_attr poll_attr = {};
  int result = ibv_start_poll(cq, &poll_attr);
  absl::Time stop =
      absl::Now() +
      GetSlowDownTimeout(timeout,
                         absl::GetFlag(FLAGS_completion_wait_multiplier));
  while (result == ENOENT && absl::Now() < stop) {
    absl::SleepFor(absl::Milliseconds(10));
    result = ibv_start_poll(cq, &poll_attr);
  }
  if (result == 0) {
    return absl::OkStatus();
  }
  if (result != ENOENT) {
    return absl::InternalError("Failed to start polling completion.");
  }
  return absl::DeadlineExceededError("Timeout while waiting for a completion.");
}

absl::Status WaitForNextExtendedCompletion(ibv_cq_ex* cq,
                                           absl::Duration timeout) {
  int result = ibv_next_poll(cq);
  absl::Time stop =
      absl::Now() +
      GetSlowDownTimeout(timeout,
                         absl::GetFlag(FLAGS_completion_wait_multiplier));
  while (result == ENOENT && absl::Now() < stop) {
    absl::SleepFor(absl::Milliseconds(10));
    result = ibv_next_poll(cq);
  }
  if (result == 0) {
    return absl::OkStatus();
  }
  ibv_end_poll(cq);
  if (result != ENOENT) {
    return absl::InternalError("Failed to get next completion.");
  }
  return absl::DeadlineExceededError("Timeout while waiting for a completion.");
}

bool CheckExtendedCompletionHasCapability(ibv_context* context,
                                          uint64_t wc_flag) {
  ibv_cq_init_attr_ex cq_attr = {.cqe = 1, .wc_flags = wc_flag};
  ibv_cq_ex* cq = ibv_create_cq_ex(context, &cq_attr);
  if (cq != nullptr) {
    int result = ibv_destroy_cq(ibv_cq_ex_to_cq(cq));
    DCHECK_EQ(result, 0);
    return true;
  }
  return false;
}

bool ExpectNoCompletion(ibv_cq* cq, absl::Duration timeout) {
  return absl::IsDeadlineExceeded(WaitForCompletion(cq, timeout).status());
}

bool ExpectNoExtendedCompletion(ibv_cq_ex* cq, absl::Duration timeout) {
  return absl::IsDeadlineExceeded(
      WaitForPollingExtendedCompletion(cq, timeout));
}

void PrintCompletion(const ibv_wc& completion) {
  LOG(INFO) << "Completion: ";
  LOG(INFO) << "  status = " << magic_enum::enum_name(completion.status);
  LOG(INFO) << "  vendor_err = " << completion.vendor_err;
  LOG(INFO) << "  wr_id = " << completion.wr_id;
  if (completion.status == IBV_WC_SUCCESS) {
    // magic_enum::enum_name can't be used as the enum value goes beyond
    // MAGIC_ENUM_RANGE_MAX (128)
    LOG(INFO) << "  opcode = " << completion.opcode;
  }
  LOG(INFO) << "  qp_num = " << completion.qp_num;
}

absl::StatusOr<ibv_async_event> WaitForAsyncEvent(ibv_context* context,
                                                  absl::Duration timeout) {
  if (context->async_fd < 0) {
    return absl::FailedPreconditionError(
        absl::StrCat("Invalid context async_fd: ", context->async_fd));
  }

  int flags = fcntl(context->async_fd, F_GETFL);
  if (flags < 0) {
    return absl::InternalError(absl::StrCat(
        "Internal error while getting fd flags: ", strerror(errno)));
  }
  int result = fcntl(context->async_fd, F_SETFL, flags | O_NONBLOCK);
  if (result < 0) {
    return absl::InternalError(absl::StrCat(
        "Internal error while getting fd flags: ", strerror(errno)));
  }

  pollfd poll_fd{
      .fd = context->async_fd,
      .events = POLLIN,
      .revents = 0,
  };
  int poll_result = poll(&poll_fd, 1, absl::ToInt64Milliseconds(timeout));
  if (poll_result < 0) {
    return absl::InternalError(absl::StrCat("Poll error: ", strerror(errno)));
  } else if (poll_result == 0) {
    return absl::DeadlineExceededError("Poll timed out");
  }
  ibv_async_event event;
  int get_event_result = ibv_get_async_event(context, &event);
  if (get_event_result != 0) {
    return absl::InternalError(
        absl::StrFormat("Failed to get async event (%d).", get_event_result));
  }
  ibv_ack_async_event(&event);
  return event;
}

absl::StatusOr<ibv_wc_status> ExecuteType1MwBind(ibv_qp* qp, ibv_mw* mw,
                                                 absl::Span<uint8_t> buffer,
                                                 ibv_mr* mr, int access) {
  static uint32_t wr_id = 1;
  ibv_mw_bind bind = CreateType1MwBindWr(wr_id++, buffer, mr, access);
  PostType1Bind(qp, mw, bind);
  ibv_wc completion = WaitForCompletion(qp->send_cq).value();
  EXPECT_EQ(completion.wr_id, bind.wr_id);
  EXPECT_EQ(completion.qp_num, qp->qp_num);
  if (completion.status == IBV_WC_SUCCESS) {
    EXPECT_EQ(completion.opcode, IBV_WC_BIND_MW);
  }
  return completion.status;
}

absl::StatusOr<ibv_wc_status> ExecuteType2MwBind(ibv_qp* qp, ibv_mw* mw,
                                                 absl::Span<uint8_t> buffer,
                                                 uint32_t rkey, ibv_mr* mr,
                                                 int access) {
  static uint32_t wr_id = 1;
  ibv_send_wr bind = CreateType2BindWr(wr_id++, mw, buffer, rkey, mr, access);
  PostSend(qp, bind);
  ibv_wc completion = WaitForCompletion(qp->send_cq).value();
  EXPECT_EQ(completion.wr_id, bind.wr_id);
  EXPECT_EQ(completion.qp_num, qp->qp_num);
  if (completion.status == IBV_WC_SUCCESS) {
    EXPECT_EQ(IBV_WC_BIND_MW, completion.opcode);
  }
  return completion.status;
}

absl::StatusOr<ibv_wc_status> ExecuteRdma(ibv_wr_opcode opcode, ibv_qp* qp,
                                          absl::Span<uint8_t> local_buffer,
                                          ibv_mr* local_mr, void* remote_buffer,
                                          uint32_t rkey) {
  static uint32_t wr_id = 1;
  ibv_sge sge = CreateSge(local_buffer, local_mr);
  ibv_send_wr wr =
      CreateRdmaWr(opcode, wr_id++, &sge, /*num_sge=*/1, remote_buffer, rkey);
  PostSend(qp, wr);
  ASSIGN_OR_RETURN(ibv_wc completion, WaitForCompletion(qp->send_cq));
  EXPECT_EQ(completion.qp_num, qp->qp_num);
  EXPECT_EQ(completion.wr_id, wr.wr_id);
  if (completion.status == IBV_WC_SUCCESS) {
    EXPECT_EQ(completion.opcode, WrToWcOpcode(opcode));
  }
  return completion.status;
}

absl::StatusOr<ibv_wc_status> ExecuteRdmaRead(ibv_qp* qp,
                                              absl::Span<uint8_t> local_buffer,
                                              ibv_mr* local_mr,
                                              void* remote_buffer,
                                              uint32_t rkey) {
  return ExecuteRdma(IBV_WR_RDMA_READ, qp, local_buffer, local_mr,
                     remote_buffer, rkey);
}

absl::StatusOr<ibv_wc_status> ExecuteRdmaWrite(ibv_qp* qp,
                                               absl::Span<uint8_t> local_buffer,
                                               ibv_mr* local_mr,
                                               void* remote_buffer,
                                               uint32_t rkey) {
  return ExecuteRdma(IBV_WR_RDMA_WRITE, qp, local_buffer, local_mr,
                     remote_buffer, rkey);
}

absl::StatusOr<ibv_wc_status> ExecuteFetchAndAdd(ibv_qp* qp, void* local_buffer,
                                                 ibv_mr* local_mr,
                                                 void* remote_buffer,
                                                 uint32_t rkey,
                                                 uint64_t comp_add) {
  static uint32_t wr_id = 1;
  ibv_sge sge{
      .addr = reinterpret_cast<uint64_t>(local_buffer),
      .length = 8,
      .lkey = local_mr->lkey,
  };
  ibv_send_wr fetch_add = CreateFetchAddWr(wr_id++, &sge, /*num_sge=*/1,
                                           remote_buffer, rkey, comp_add);
  PostSend(qp, fetch_add);
  ASSIGN_OR_RETURN(ibv_wc completion, WaitForCompletion(qp->send_cq));
  EXPECT_EQ(completion.qp_num, qp->qp_num);
  EXPECT_EQ(completion.wr_id, fetch_add.wr_id);
  if (completion.status == IBV_WC_SUCCESS) {
    EXPECT_EQ(IBV_WC_FETCH_ADD, completion.opcode);
  }
  return completion.status;
}

absl::StatusOr<ibv_wc_status> ExecuteCompareAndSwap(
    ibv_qp* qp, void* local_buffer, ibv_mr* local_mr, void* remote_buffer,
    uint32_t rkey, uint64_t comp_add, uint64_t swap) {
  static uint32_t wr_id = 1;
  ibv_sge sge{.addr = reinterpret_cast<uint64_t>(local_buffer),
              .length = 8,
              .lkey = local_mr->lkey};
  ibv_send_wr comp_swap = CreateCompSwapWr(wr_id++, &sge, /*num_sge=*/1,
                                           remote_buffer, rkey, comp_add, swap);
  PostSend(qp, comp_swap);
  ASSIGN_OR_RETURN(ibv_wc completion, WaitForCompletion(qp->send_cq));
  EXPECT_EQ(completion.qp_num, qp->qp_num);
  EXPECT_EQ(completion.wr_id, comp_swap.wr_id);
  if (completion.status == IBV_WC_SUCCESS) {
    EXPECT_EQ(IBV_WC_COMP_SWAP, completion.opcode);
  }
  return completion.status;
}

absl::StatusOr<ibv_wc_status> ExecuteLocalInvalidate(ibv_qp* qp,
                                                     uint32_t rkey) {
  static uint32_t wr_id = 1;
  ibv_send_wr invalidate = CreateLocalInvalidateWr(wr_id++, rkey);
  PostSend(qp, invalidate);
  ASSIGN_OR_RETURN(ibv_wc completion, WaitForCompletion(qp->send_cq));
  EXPECT_EQ(completion.wr_id, invalidate.wr_id);
  EXPECT_EQ(completion.qp_num, qp->qp_num);
  if (completion.status == IBV_WC_SUCCESS) {
    EXPECT_EQ(completion.opcode, IBV_WC_LOCAL_INV);
  }
  return completion.status;
}

absl::StatusOr<std::pair<ibv_wc_status, ibv_wc_status>> ExecuteSendRecv(
    ibv_qp* src_qp, ibv_qp* dst_qp, absl::Span<uint8_t> src_buffer,
    ibv_mr* src_mr, absl::Span<uint8_t> dst_buffer, ibv_mr* dst_mr) {
  ibv_sge dst_sge = CreateSge(dst_buffer, dst_mr);
  ibv_recv_wr recv = CreateRecvWr(/*wr_id=*/0, &dst_sge, /*num_sge=*/1);
  PostRecv(dst_qp, recv);

  ibv_sge src_sge = CreateSge(src_buffer, src_mr);
  ibv_send_wr send = CreateSendWr(/*wr_id=*/1, &src_sge, /*num_sge=*/1);
  PostSend(src_qp, send);

  ASSIGN_OR_RETURN(ibv_wc src_completion, WaitForCompletion(src_qp->send_cq));
  if (src_completion.status == IBV_WC_SUCCESS) {
    EXPECT_EQ(IBV_WC_SEND, src_completion.opcode);
  }
  ASSIGN_OR_RETURN(ibv_wc dst_completion, WaitForCompletion(dst_qp->recv_cq));
  if (dst_completion.status == IBV_WC_SUCCESS) {
    EXPECT_EQ(IBV_WC_RECV, dst_completion.opcode);
  }
  return std::make_pair(src_completion.status, dst_completion.status);
}

absl::StatusOr<std::pair<ibv_wc_status, ibv_wc_status>> TwoSgeSendRecv(
    ibv_qp* src_qp, ibv_qp* dst_qp, absl::Span<uint8_t> src_buffer,
    ibv_mr* src_mr, absl::Span<uint8_t> dst_buffer_1, ibv_mr* dst_mr_1,
    absl::Span<uint8_t> dst_buffer_2, ibv_mr* dst_mr_2) {
  ibv_sge dst_sge[2] = {{CreateSge(dst_buffer_1, dst_mr_1)},
                        {CreateSge(dst_buffer_2, dst_mr_2)}};
  ibv_sge src_sge = CreateSge(src_buffer, src_mr);
  ibv_wc src_completion, dst_completion;
  int i;
  for (i = 0; i < 2; i++) {
    // The for loop creates 2 send and 2 receive work requests.
    // In order to maintain unique values of the work request id across 4
    // requests, using expressions i*2 and (i*2)+1, below.
    ibv_recv_wr recv =
        CreateRecvWr(/*wr_id=*/i * 2, &dst_sge[i], /*num_sge=*/1);
    PostRecv(dst_qp, recv);
    ibv_send_wr send =
        CreateSendWr(/*wr_id=*/(i * 2) + 1, &src_sge, /*num_sge=*/1);
    PostSend(src_qp, send);

    ASSIGN_OR_RETURN(src_completion, WaitForCompletion(src_qp->send_cq));
    ASSIGN_OR_RETURN(dst_completion, WaitForCompletion(dst_qp->recv_cq));
    if (src_completion.status != IBV_WC_SUCCESS ||
        dst_completion.status != IBV_WC_SUCCESS) {
      goto label;
    } else {
      EXPECT_EQ(IBV_WC_SEND, src_completion.opcode);
      EXPECT_EQ(IBV_WC_RECV, dst_completion.opcode);
    }
  }
label:
  return std::make_pair(src_completion.status, dst_completion.status);
}

absl::StatusOr<ibv_context*> OpenUntrackedDevice(
    const std::string device_name) {
  ibv_device** devices = nullptr;
  absl::Cleanup free_list = [&devices]() {
    if (devices) {
      ibv_free_device_list(devices);
    }
  };
  int num_devices = 0;
  devices = ibv_get_device_list(&num_devices);
  if (num_devices <= 0 || !devices) {
    return absl::InternalError("No devices found.");
  }

  ibv_device* device = nullptr;
  bool device_selected = false;
  if (device_name.empty()) {
    device = devices[0];
    LOG(INFO) << "Select devices[0] (" << device->name << ").";
    device_selected = true;
  }
  for (int i = 0; i < num_devices; ++i) {
    if (device_name == devices[i]->name) {
      LOG(INFO) << "Select device " << device_name << ".";
      device = devices[i];
      device_selected = true;
    }
  }

  if (!device_selected) {
    LOG(INFO) << "Available devices for --device_name flag";
    for (int i = 0; i < num_devices; i++) {
      LOG(INFO) << devices[i]->name;
    }
    return absl::InternalError("RDMA device " + device_name + " not found.");
  }
  if (!device) {
    return absl::InternalError("Selected device is nullptr.");
  }

  ibv_context* context = ibv_open_device(device);
  if (!context) {
    return absl::InternalError("Failed to open device.");
  }

  return context;
}

}  // namespace verbs_util
}  // namespace rdma_unit_test
