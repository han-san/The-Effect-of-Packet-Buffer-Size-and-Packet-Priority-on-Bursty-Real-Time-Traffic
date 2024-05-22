#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <ranges>
#include <sstream>
#include <thread>
#include <vector>

// CSV data format (times are monotonic to the respective device) (things like
// which burst a packet is part of can be calculated from the sequence number
// and the size of each burst):
//
// [Sequence number, burst index, ip of sending device, time sent (ns), time
// received (ns)]
//
// The packets are listed in the order they are received.

std::string create_csv(const std::vector<DecodedPacket> &packets) {
  std::stringstream csvString;
  csvString << std::format(
      "Sequence number,burst index,ip of sending device,time sent "
      "(ns),time received (ns)\n");

  for (const auto &packet : packets) {
    const auto deviceIp = to_string(packet.fromAddr);
    const auto isFromPi1 =
        deviceIp == pi1HighPrioIp || deviceIp == pi1LowPrioIp;
    const auto svPacketsPerBurst =
        isFromPi1 ? experimentVariables.pi1SvPacketsPerBurst
                  : experimentVariables.pi2SvPacketsPerBurst;
    const auto ipPacketsPerBurst =
        isFromPi1 ? experimentVariables.pi1IpPacketsPerBurst
                  : experimentVariables.pi2IpPacketsPerBurst;
    const auto packetsPerBurst = packet.type == DecodedPacket::Type::Sv
                                     ? svPacketsPerBurst
                                     : ipPacketsPerBurst;
    if (packetsPerBurst == 0) {
      throw std::logic_error("The packetsPerBurst variables somehow became 0 "
                             "when a positive value was expected");
    }
    const auto burstIndex = packet.sequenceNumber / packetsPerBurst;

    csvString << std::format("{},{},\"{}\",{},{}\n", packet.sequenceNumber,
                             burstIndex, deviceIp, packet.timeSent.count(),
                             packet.timeReceived.time_since_epoch().count());
  }

  return csvString.str();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: server [iec|...].ssv\n";
    return EXIT_FAILURE;
  }

  loadConfigFromFile(argv[1]);

  auto totalPacketsPerBurst = experimentVariables.pi1IpPacketsPerBurst +
                              experimentVariables.pi2IpPacketsPerBurst +
                              experimentVariables.pi1SvPacketsPerBurst +
                              experimentVariables.pi2SvPacketsPerBurst;
  auto totalPackets = totalPacketsPerBurst * experimentVariables.burstCount;

  auto sock = Socket::createServer();
  auto pi1AddrInfo = AddrInfo::from(Device::pi1);
  auto pi2AddrInfo = AddrInfo::from(Device::pi2);

  std::vector<UndecodedPacket> packets;
  packets.reserve(totalPackets);

  std::map<std::string, std::vector<DecodedPacket>> piSyncMessages{
      {pi1HighPrioIp, {}},
      {pi2HighPrioIp, {}},
      {pi1LowPrioIp, {}},
      {pi2LowPrioIp, {}}};

  for (auto &[k, v] : piSyncMessages) {
    v.reserve(syncMessageCount);
  }

  for (std::size_t i = 0; i < clientCount; ++i) {
    std::cout << "Syncing with pi...\n";
    std::flush(std::cout);

    for (std::size_t j = 0; j < syncMessageCount; ++j) {
      const auto maybePacket = sock.recvPacket(Socket::Blocking::yes);
      const auto &packet = maybePacket.value().decode();
      try {
        piSyncMessages.at(to_string(packet.fromAddr)).push_back(packet);
      } catch (const std::out_of_range &err) {
        throw std::out_of_range(std::format(
            "Received packet from unexpected IP [{}]. (Thrown exception: {})",
            to_string(packet.fromAddr), err.what()));
      }
    }
    sock.resetSequenceNumberAndBuffers();
    for (std::size_t j = 0; j < syncMessageCount; ++j) {
      const auto maybePacket = sock.recvPacket(Socket::Blocking::yes);
      const auto &packet = maybePacket.value().decode();
      try {
        piSyncMessages.at(to_string(packet.fromAddr)).push_back(packet);
      } catch (const std::out_of_range &err) {
        throw std::out_of_range(std::format(
            "Received packet from unexpected IP [{}]. (Thrown exception: {})",
            to_string(packet.fromAddr), err.what()));
      }
    }

    auto fullSends = std::accumulate(
        piSyncMessages.cbegin(), piSyncMessages.cend(), std::size_t{0},
        [](auto acc, const auto &kvp) {
          return acc + (kvp.second.size() == syncMessageCount ? 1 : 0);
        });
    if (fullSends != (i + 1) * clientCount) {
      for (const auto &[k, v] : piSyncMessages) {
        std::cerr << std::format("{}: {}/{}\n", k, v.size(), syncMessageCount);
      }
      throw std::logic_error("Error syncing with Pi");
    }

    sock.resetSequenceNumberAndBuffers();
  }

  std::cout << "Done!\n";

  // Really make sure both clients have time to get ready.
  std::this_thread::sleep_for(1s);

  // TODO: Make start signal multicast?
  std::cout << "sending start signal\n";
  std::flush(std::cout);
  sock.sendSvPacket(pi1AddrInfo);
  sock.sendSvPacket(pi2AddrInfo);

  // Receive all packets without any kind of processing to avoid delays.
  for (std::size_t i = 0; i < totalPackets; ++i) {
    // recvPacket will return a nullopt if it times out .
    const auto maybePacket = sock.recvPacket(Socket::Blocking::no);
    if (!maybePacket) {
      break;
    }
    packets.emplace_back(*maybePacket);
  }

  std::cout << "Finished receiving packets.\n";

  std::vector<DecodedPacket> decodedPackets;
  decodedPackets.reserve(packets.size());

  std::ranges::transform(packets, std::back_inserter(decodedPackets),
                         [](const auto &packet) { return packet.decode(); });

  const auto &pi1HighPrioSyncMessages = piSyncMessages.at(pi1HighPrioIp);
  const auto &pi2HighPrioSyncMessages = piSyncMessages.at(pi2HighPrioIp);
  const auto &pi1LowPrioSyncMessages = piSyncMessages.at(pi1LowPrioIp);
  const auto &pi2LowPrioSyncMessages = piSyncMessages.at(pi2LowPrioIp);

  const auto pi1HighPrioMinDiff = std::ranges::min(std::views::transform(
      pi1HighPrioSyncMessages, [](const DecodedPacket &packet) {
        return packet.timeReceived.time_since_epoch() - packet.timeSent;
      }));

  const auto pi2HighPrioMinDiff = std::ranges::min(std::views::transform(
      pi2HighPrioSyncMessages, [](const DecodedPacket &packet) {
        return packet.timeReceived.time_since_epoch() - packet.timeSent;
      }));

  const auto pi1LowPrioMinDiff = std::ranges::min(std::views::transform(
      pi1LowPrioSyncMessages, [](const DecodedPacket &packet) {
        return packet.timeReceived.time_since_epoch() - packet.timeSent;
      }));

  const auto pi2LowPrioMinDiff = std::ranges::min(std::views::transform(
      pi2LowPrioSyncMessages, [](const DecodedPacket &packet) {
        return packet.timeReceived.time_since_epoch() - packet.timeSent;
      }));

  // Create a csv from the stats gathered.
  {
    auto csvString = create_csv(decodedPackets);
    auto now = chr::utc_clock::now();

    auto metadata = std::format(
        "{}\nburst_count={}\nburst_period={}\npi1_sv_packets_per_burst={}\npi1_"
        "ip_packets_per_burst={}\npi2_sv_packets_per_burst={}\npi2_ip_packets_"
        "per_burst={}\nsv_packet_size={}\nip_packet_size={}\nswitch_buffer_"
        "size={}\np1_high_prio_min_diff={}\np2_high_prio_min_diff={}\np1_low_"
        "prio_min_diff={}\np2_low_prio_min_diff={}\nmax_latency={}",
        now, experimentVariables.burstCount, experimentVariables.burstPeriod,
        experimentVariables.pi1SvPacketsPerBurst,
        experimentVariables.pi1IpPacketsPerBurst,
        experimentVariables.pi2SvPacketsPerBurst,
        experimentVariables.pi2IpPacketsPerBurst,
        experimentVariables.svPacketSize, experimentVariables.ipPacketSize,
        experimentVariables.switchBufferSize, pi1HighPrioMinDiff,
        pi2HighPrioMinDiff, pi1LowPrioMinDiff, pi2LowPrioMinDiff,
        experimentVariables.maxLatency);

    auto statsFile = std::ofstream(std::format("{}.csv", now), std::ios::out);
    statsFile << csvString;
    auto metadataFile =
        std::ofstream(std::format("{}.meta", now), std::ios::out);
    metadataFile << metadata;
  }
}

