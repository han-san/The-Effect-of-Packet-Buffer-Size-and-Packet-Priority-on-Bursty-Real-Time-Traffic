#pragma once

#include <arpa/inet.h>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std::chrono_literals;
namespace chr = std::chrono;

// Make sure our monotonic clock has nanosecond resolution (both libc++ and
// libstdc++ should use clock_gettime if available).
static_assert(
    std::is_same_v<chr::steady_clock::period, std::ratio<1, 1'000'000'000>>);
static_assert(std::is_same_v<chr::steady_clock::rep, std::int64_t>);

// Defaulted to IEC values
struct ExperimentVariables {
  std::size_t burstCount = 1000;
  chr::steady_clock::duration burstPeriod = 208300ns;
  // In SV packets per client.
  std::size_t pi1SvPacketsPerBurst = 10;
  std::size_t pi2SvPacketsPerBurst = 10;
  // In bytes
  std::size_t switchBufferSize = 100000;
  chr::steady_clock::duration maxLatency = 156000ns;
  // In bytes. Includes both Ethernet header and, in our case, UDP header.
  std::size_t svPacketSize = 160;
  // IEC example value is actually 1530, but our MTU is 1500.
  std::size_t ipPacketSize = 1500;
  std::size_t pi1IpPacketsPerBurst = 1;
  std::size_t pi2IpPacketsPerBurst = 0;
};

// Evil global variable.
ExperimentVariables experimentVariables{};

// How many sync messages should be sent.
constexpr std::size_t syncMessageCount = 500;

// in bytes
constexpr int udpHeaderSize = 8;
constexpr int ipHeaderSize = 20;
constexpr int ethernetHeaderSize = 18;

constexpr int mtu = 1500;

// If we don't receive anything for two whole burst period, then we're probably
// done. The minimum timeout is 1ms.
inline chr::milliseconds recvPollTimeout() {
  return std::max(1ms, chr::duration_cast<chr::milliseconds>(
                           2 * experimentVariables.burstPeriod));
}

constexpr const char *port = "27015";
enum class Device { pi1, pi2, laptop };
constexpr std::size_t clientCount = 2;
constexpr const char *laptopHighPrioIp = "192.168.1.2";
constexpr const char *pi1HighPrioIp = "192.168.1.10";
constexpr const char *pi2HighPrioIp = "192.168.1.4";
constexpr const char *laptopLowPrioIp = "192.168.2.2";
constexpr const char *pi1LowPrioIp = "192.168.2.10";
constexpr const char *pi2LowPrioIp = "192.168.2.4";

inline void loadConfigFromFile(const std::string &fileName) {
  // We read our experiment configuration from the provided space separated
  // values file.
  auto file = std::ifstream(fileName);
  if (!file.is_open()) {
    throw std::logic_error(
        "Error opening config file (did you give the correct path?).");
  }

  std::string key;
  std::string value;
  file >> key;
  file >> value;
  experimentVariables.burstCount = stoull(value);
  file >> key;
  file >> value;
  experimentVariables.burstPeriod = chr::nanoseconds{stoll(value)};
  file >> key;
  file >> value;
  experimentVariables.pi1SvPacketsPerBurst = stoull(value);
  file >> key;
  file >> value;
  experimentVariables.pi2SvPacketsPerBurst = stoull(value);
  file >> key;
  file >> value;
  experimentVariables.pi1IpPacketsPerBurst = stoull(value);
  file >> key;
  file >> value;
  experimentVariables.pi2IpPacketsPerBurst = stoull(value);
  file >> key;
  file >> value;
  experimentVariables.switchBufferSize = stoull(value);
  file >> key;
  file >> value;
  experimentVariables.maxLatency = chr::nanoseconds{stoll(value)};
  file >> key;
  file >> value;
  experimentVariables.svPacketSize = stoull(value);
  file >> key;
  file >> value;
  experimentVariables.ipPacketSize = stoull(value);

  std::cout << std::format("Loaded config from file {}:\n\n", fileName);
  std::cout << std::format("Burst count: {}\n", experimentVariables.burstCount);
  std::cout << std::format("Burst period(ns): {}\n",
                           experimentVariables.burstPeriod);
  std::cout << std::format("Pi1 SV packets per burst: {}\n",
                           experimentVariables.pi1SvPacketsPerBurst);
  std::cout << std::format("Pi2 SV packets per burst: {}\n",
                           experimentVariables.pi2SvPacketsPerBurst);
  std::cout << std::format("Pi1 IP packets per burst: {}\n",
                           experimentVariables.pi1IpPacketsPerBurst);
  std::cout << std::format("Pi2 IP packets per burst: {}\n",
                           experimentVariables.pi2IpPacketsPerBurst);
  std::cout << std::format("Switch buffer size(B): {}\n",
                           experimentVariables.switchBufferSize);
  std::cout << std::format("Max latency(ns): {}\n",
                           experimentVariables.maxLatency);
  std::cout << std::format("SV packet size(B): {}\n",
                           experimentVariables.svPacketSize);
  std::cout << std::format("IP packet size(B): {}\n",
                           experimentVariables.ipPacketSize);
}

class AddrInfo {
public:
  AddrInfo(const AddrInfo &) = delete;
  AddrInfo(AddrInfo &&) = delete;
  AddrInfo &operator=(const AddrInfo &) = delete;
  AddrInfo &operator=(AddrInfo &&) = delete;
  ~AddrInfo() noexcept {
    freeaddrinfo(addrInfo_);
    addrInfo_ = nullptr;
  }

  static AddrInfo getOtherAddrInfo(const std::string &otherIp) noexcept(false) {
    struct addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;

    return AddrInfo(otherIp.c_str(), port, &hints);
  }

  static AddrInfo getOwnAddrInfo() noexcept(false) {
    struct addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    return AddrInfo(nullptr, port, &hints);
  }

  enum class Priority { High, Low };

  static AddrInfo from(Device target, Priority priority = Priority::High) {
    switch (target) {
    case Device::laptop: {
      const auto &ip =
          priority == Priority::High ? laptopHighPrioIp : laptopLowPrioIp;
      return AddrInfo::getOtherAddrInfo(ip);
    }
    case Device::pi1: {
      const auto &ip =
          priority == Priority::High ? pi1HighPrioIp : pi1LowPrioIp;
      return AddrInfo::getOtherAddrInfo(ip);
    }
    case Device::pi2: {
      const auto &ip =
          priority == Priority::High ? pi2HighPrioIp : pi2LowPrioIp;
      return AddrInfo::getOtherAddrInfo(ip);
    }
    }

    throw std::logic_error("Unknown device");
  }

  const addrinfo &get() const {
    if (addrInfo_ == nullptr) {
      throw std::logic_error("Tried to get a null addrinfo");
    }
    return *addrInfo_;
  }

private:
  AddrInfo(const char *name, const char *service, const addrinfo *hints) {
    addrinfo *servinfo = nullptr;
    int status = getaddrinfo(name, service, hints, &servinfo);
    if (status != 0) {
      throw std::logic_error(std::format("Error getting address info: {}\n",
                                         gai_strerror(status)));
    }

    addrInfo_ = servinfo;
  }

  addrinfo *addrInfo_ = nullptr;
};

inline std::string to_string(const sockaddr_storage &addr) {
  char buf[INET_ADDRSTRLEN]{};
  const in_addr *sin_addr =
      &(reinterpret_cast<const sockaddr_in *>(&addr)->sin_addr);
  inet_ntop(addr.ss_family, sin_addr, buf,
            static_cast<unsigned int>(std::size(buf)));
  return std::string(buf);
}

struct DecodedPacket {
  enum class Type { Sv, Ip };
  Type type{};
  std::uint32_t sequenceNumber{};
  chr::steady_clock::duration timeSent{};
  sockaddr_storage fromAddr{};
  chr::steady_clock::time_point timeReceived{};
};

class UndecodedPacket {
public:
  UndecodedPacket(std::vector<std::byte> buf, const sockaddr_storage &fromAddr,
                  const chr::steady_clock::time_point &timeReceived,
                  const ssize_t packetSize)
      : buf_{std::move(buf)}, size_{packetSize}, fromAddr_{fromAddr},
        timeReceived_{timeReceived} {}

  UndecodedPacket() = default;

  static std::size_t svPacketSizeWithoutHeaders() {
    return experimentVariables.svPacketSize - udpHeaderSize - ipHeaderSize -
           ethernetHeaderSize;
  };

  static std::size_t ipPacketSizeWithoutHeaders() {
    return experimentVariables.ipPacketSize - udpHeaderSize - ipHeaderSize -
           ethernetHeaderSize;
  };

  DecodedPacket decode() const {
    std::uint8_t currentOffset = 0;
    DecodedPacket packet{};

    if (size_ < 0) {
      throw std::logic_error("Packet size is < 0, which shouldn't be possible");
    }

    auto size = static_cast<std::size_t>(size_);
    if (size == svPacketSizeWithoutHeaders()) {
      packet.type = DecodedPacket::Type::Sv;
    } else if (size == ipPacketSizeWithoutHeaders()) {
      packet.type = DecodedPacket::Type::Ip;
    } else {
      throw std::logic_error(
          "The packet size is neither equal to an SV packet or an IP packet");
    }

    if (currentOffset + sizeof(packet.sequenceNumber) > buf_.size()) {
      throw std::logic_error("buffer overflow");
    }
    std::memcpy(&packet.sequenceNumber, buf_.data() + currentOffset,
                sizeof(packet.sequenceNumber));
    currentOffset += sizeof(packet.sequenceNumber);
    packet.sequenceNumber = ntohl(packet.sequenceNumber);

    std::int64_t rawTimeSent{};

    if (currentOffset + sizeof(rawTimeSent) > buf_.size()) {
      throw std::logic_error("buffer overflow");
    }
    std::memcpy(&rawTimeSent, buf_.data() + currentOffset, sizeof(rawTimeSent));
    rawTimeSent = static_cast<std::int64_t>(
        be64toh(static_cast<std::uint64_t>(rawTimeSent)));

    packet.timeSent = chr::steady_clock::duration{rawTimeSent};

    packet.timeReceived = timeReceived_;
    packet.fromAddr = fromAddr_;

    return packet;
  }

  std::string address() const noexcept { return to_string(fromAddr_); }
  chr::steady_clock::time_point receivedTimepoint() const noexcept {
    return timeReceived_;
  }

private:
  std::vector<std::byte> buf_{};
  ssize_t size_{};
  sockaddr_storage fromAddr_{};
  chr::steady_clock::time_point timeReceived_{};
};

class Socket {
public:
  ~Socket() noexcept {
    close(fd_);
    fd_ = -1;
  }

  Socket(const Socket &) = delete;
  Socket(Socket &&) = delete;
  Socket &operator=(const Socket &) = delete;
  Socket &operator=(Socket &&) = delete;

  static Socket createClient(const std::string &serverIp) {
    std::size_t expectedRecvPackageCount = 1;
    return Socket(AddrInfo::getOtherAddrInfo(serverIp),
                  expectedRecvPackageCount);
  }
  static Socket createServer() {

    auto expectedSvPackets = experimentVariables.burstCount *
                             (experimentVariables.pi1SvPacketsPerBurst +
                              experimentVariables.pi2SvPacketsPerBurst);
    auto expectedIpPackets = experimentVariables.burstCount *
                             (experimentVariables.pi1IpPacketsPerBurst +
                              experimentVariables.pi2IpPacketsPerBurst);
    auto expectedRecvPackageCount = expectedIpPackets + expectedSvPackets;
    return Socket(AddrInfo::getOwnAddrInfo(),
                  std::max(expectedRecvPackageCount, syncMessageCount));
  }

  enum class Blocking { no, yes };

  std::optional<UndecodedPacket> recvPacket(Blocking shouldBlock) {
    auto &buf = buffers.at(currentBufIndex);
    struct sockaddr_storage fromAddr {};
    socklen_t socklen = sizeof(fromAddr);

    if (shouldBlock == Blocking::no) {
      pollfd poller{.fd = fd_, .events = POLLIN, .revents = 0};
      int ret = poll(&poller, 1, static_cast<int>(recvPollTimeout().count()));
      if (ret < 1) {
        if (ret == 0) {
          // Timeout
          return std::nullopt;
        }

        throw std::logic_error("Error during polling");
      }
    }

    ssize_t bytes = recvfrom(fd_, buf.data(), buf.size(), 0,
                             reinterpret_cast<sockaddr *>(&fromAddr), &socklen);
    auto now = chr::steady_clock::now();

    if (bytes != static_cast<ssize_t>(
                     UndecodedPacket::svPacketSizeWithoutHeaders()) &&
        bytes != static_cast<ssize_t>(
                     UndecodedPacket::ipPacketSizeWithoutHeaders())) {
      throw std::logic_error("Error receiving packet");
    }

    ++currentBufIndex;

    return std::optional<UndecodedPacket>(std::in_place, std::move(buf),
                                          fromAddr, now, bytes);
  }

  void resetSequenceNumberAndBuffers() {
    svSequenceNumber = 0;
    ipSequenceNumber = 0;
    currentBufIndex = 0;
    for (auto &buf : buffers) {
      buf = std::vector<std::byte>(
          std::max(UndecodedPacket::svPacketSizeWithoutHeaders(),
                   UndecodedPacket::ipPacketSizeWithoutHeaders()));
    }
  }

  void sendSvPacket(const AddrInfo &recipient) {
    auto sequenceNumber = svSequenceNumber;
    ++svSequenceNumber;

    sendPacket(recipient, sequenceNumber,
               UndecodedPacket::svPacketSizeWithoutHeaders());
  }

  void sendIpPacket(const AddrInfo &recipient) {
    auto sequenceNumber = ipSequenceNumber;
    ++ipSequenceNumber;
    sendPacket(recipient, sequenceNumber,
               UndecodedPacket::ipPacketSizeWithoutHeaders());
  }

private:
  explicit Socket(const AddrInfo &addrInfo, std::size_t expectedRecvPacketCount)
      : fd_(socket(addrInfo.get().ai_family, addrInfo.get().ai_socktype,
                   addrInfo.get().ai_protocol)),
        buffers(expectedRecvPacketCount) {

    if (fd_ == -1) {
      throw std::logic_error("Couldn't open socket");
    }

    int res = bind(fd_, addrInfo.get().ai_addr, addrInfo.get().ai_addrlen);
    if (res != 0) {
      throw std::logic_error("Error binding socket");
    }

    for (auto &buf : buffers) {
      buf = std::vector<std::byte>(
          std::max(UndecodedPacket::svPacketSizeWithoutHeaders(),
                   UndecodedPacket::ipPacketSizeWithoutHeaders()));
    }
  }

  void sendPacket(const AddrInfo &recipient, std::uint32_t sequenceNumber,
                  std::size_t packetSize) {
    auto networkNumber = htonl(sequenceNumber);
    std::memcpy(sendBuffer.data(), &networkNumber, sizeof(networkNumber));
    auto now = chr::steady_clock::now().time_since_epoch().count();
    static_assert(std::is_same_v<decltype(now), std::int64_t>);
    std::uint8_t currentOffset = sizeof(networkNumber);

    auto timeInNetworkOrder = htobe64(static_cast<std::uint64_t>(now));
    std::memcpy(sendBuffer.data() + currentOffset, &timeInNetworkOrder,
                sizeof(timeInNetworkOrder));

    ssize_t sent = sendto(fd_, sendBuffer.data(), packetSize, 0,
                          recipient.get().ai_addr, recipient.get().ai_addrlen);

    if (sent != static_cast<ssize_t>(packetSize)) {
      throw std::logic_error("Error sending packet");
    }
  }

  int fd_ = -1;

  std::vector<std::byte> sendBuffer{
      std::max(UndecodedPacket::svPacketSizeWithoutHeaders(),
               UndecodedPacket::ipPacketSizeWithoutHeaders())};
  std::size_t currentBufIndex = 0;
  std::vector<std::vector<std::byte>> buffers{};

  std::uint32_t svSequenceNumber = 0;
  std::uint32_t ipSequenceNumber = 0;
};
