#include "common.hpp"

#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: server [configname].ssv [client ID (1 or 2)]\n";
    return EXIT_FAILURE;
  }

  auto clientId = std::stoi(argv[2]);
  if (clientId != 1 && clientId != 2) {
    throw std::invalid_argument("The client ID must be 1 or 2");
  }

  loadConfigFromFile(argv[1]);

  auto svPacketsPerBurst = clientId == 1
                               ? experimentVariables.pi1SvPacketsPerBurst
                               : experimentVariables.pi2SvPacketsPerBurst;
  auto ipPacketsPerBurst = clientId == 1
                               ? experimentVariables.pi1IpPacketsPerBurst
                               : experimentVariables.pi2IpPacketsPerBurst;

  auto laptopHighPrioAddrInfo = AddrInfo::from(Device::laptop);
  auto laptopLowPrioAddrInfo =
      AddrInfo::from(Device::laptop, AddrInfo::Priority::Low);
  auto socket = Socket::createServer();

  // Try to figure out the delay with an empty buffer.
  for (std::size_t i = 0; i < syncMessageCount; ++i) {
    std::this_thread::sleep_for(1ms);
    socket.sendSvPacket(laptopHighPrioAddrInfo);
  }

  socket.resetSequenceNumberAndBuffers();

  // Try to figure out the delay with an empty buffer.
  for (std::size_t i = 0; i < syncMessageCount; ++i) {
    std::this_thread::sleep_for(1ms);
    socket.sendSvPacket(laptopLowPrioAddrInfo);
  }

  socket.resetSequenceNumberAndBuffers();

  std::cout << "waiting for go message..." << '\n';
  std::flush(std::cout);
  socket.recvPacket(Socket::Blocking::yes);
  auto periodStart = chr::steady_clock::now();
  for (std::size_t i = 0; i < experimentVariables.burstCount; ++i) {
    for (std::size_t j = 0; j < svPacketsPerBurst; ++j) {
      socket.sendSvPacket(laptopHighPrioAddrInfo);
    }
    for (std::size_t j = 0; j < ipPacketsPerBurst; ++j) {
      socket.sendIpPacket(laptopLowPrioAddrInfo);
    }

    std::this_thread::sleep_until(periodStart +
                                  ((i + 1) * experimentVariables.burstPeriod));
  }

  // Wait for a while until we print to avoid extra traffic when connected via
  // SSH.
  std::this_thread::sleep_for(2s);
  std::cout << "sent packets\n";
}
