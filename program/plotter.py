#! /usr/bin/env python3

import matplotlib.pyplot as plt
import scienceplots
import os
import pandas as pd
import sys
import math

import matplotlib as mpl

plt.style.use(['science', 'ieee'])
# Tight layout
plt.rcParams["figure.autolayout"]  = True

pi1HighPrioIp = '192.168.1.10'
pi1LowPrioIp = '192.168.2.10'
pi2HighPrioIp = '192.168.1.4'
pi2LowPrioIp = '192.168.2.4'

def getPiDataFrame(dataFrame, piIp):
  return dataFrame[dataFrame['ip'].str.contains(piIp)]

def getBurst(dataFrame, burstIndex):
  return dataFrame[dataFrame['burst'] == burstIndex]

def piBurstTimeToSendAll(dataFrame, piIp, burstIndex):
  pi = getPiDataFrame(dataFrame, piIp)
  burst = getBurst(pi, burstIndex)
  if len(burst) == 0:
    return math.nan
  sentIndex = 3
  first = burst.iloc[0, sentIndex]
  last = burst.iloc[len(burst) - 1, sentIndex]
  return last - first

def piBurstTimeToRecvAll(dataFrame, piIp, burstIndex):
  pi = getPiDataFrame(dataFrame, piIp)
  burst = getBurst(pi, burstIndex)
  recv = burst['recv']
  if len(recv) == 0:
    return math.nan
  first = recv.iloc[0]
  last = recv.iloc[len(burst) - 1]
  return last - first

def burstTimeToRecvAll(dataFrame, burstIndex):
  burst = getBurst(dataFrame, burstIndex)
  recv = burst['recv']
  if len(recv) == 0:
    return math.nan
  first = recv.iloc[0]
  last = recv.iloc[-1]
  return last - first

def startRecvTime(dataFrame):
  if len(dataFrame['recv']) == 0:
    return math.nan
  return dataFrame['recv'].iat[0]

def startSendTime(dataFrame):
  if len(dataFrame['sent']) == 0:
    return math.nan
  return dataFrame['sent'].iat[0]

def packetsLostDuringBurst(df, piIp, burstIndex):
  burst = getBurst(df, burstIndex)
  pi = getPiDataFrame(burst, piIp)

  sequenceNumbers = pi.drop(columns=['ip', 'sent', 'recv', 'burst']).sort_values(by='seq')

  diff = sequenceNumbers.diff(periods=1)
  if len(diff) == 0:
    return math.nan
  sum = (diff - 1).sum().iat[0]
  return sum

# FIXME: Maybe not have order received as x, but instead sequence number?
def plotTimeBetweenReceivedPackets(df):
  print("Plotting time between received packets... ", end='', flush=True)
  packets = timeBetweenReceivedPackets(df)
  packets.plot(xlabel='Order received', ylabel = 'Time between this and last packet (ns)')
  print("Done!")

def timeBetweenReceivedPackets(df):
  recvPackets = df.drop(columns = ['seq', 'ip', 'sent', 'burst'])
  return recvPackets.diff(periods=1)

# Assumes df only contains entries for one device.
def timeBetweenSentPackets(df):
  sentPackets = df.drop(columns = ['seq', 'ip', 'recv', 'burst'])
  return sentPackets.diff(periods=1)

def plotTimeBetweenSentPackets(df):
  print("Plotting time between sent packets... ", end='', flush=True)
  pi1 = getPiDataFrame(df, pi1HighPrioIp)
  pi2 = getPiDataFrame(df, pi2HighPrioIp)

  pi1time = timeBetweenSentPackets(pi1)
  pi2time = timeBetweenSentPackets(pi2)
  
  pi1time.plot(xlabel='Order received', ylabel = 'Time between this and last packet (ns)')
  pi2time.plot(xlabel='Order received', ylabel = 'Time between this and last packet (ns)')
  print("Done!")

def getCumulativeReceiveTimes(df, piIp):
  pi = getPiDataFrame(df, piIp)
  piStartRecvTime = startRecvTime(pi)
  piReceiveTimes = pi.drop(columns = ['burst', 'ip', 'sent'])
  piCumulativeReceiveTimes = piReceiveTimes - [0, piStartRecvTime]
  return piCumulativeReceiveTimes

def getCumulativeSendTimes(df, piIp):
  pi = getPiDataFrame(df, piIp)
  piStartSendTime = startSendTime(pi)
  piSendTimes = pi.drop(columns = ['burst', 'ip', 'recv'])
  piCumulativeSendTimes = piSendTimes - [0, piStartSendTime]
  return piCumulativeSendTimes

def plotCumulativeSendTimes(df):
  print("Plotting cumulative send times... ", end='', flush=True)
  pi1CumulativeSendTimes = getCumulativeSendTimes(df, pi1HighPrioIp)
  pi2CumulativeSendTimes = getCumulativeSendTimes(df, pi2HighPrioIp)

  merged = pi1CumulativeSendTimes.merge(pi2CumulativeSendTimes, on='seq', how='outer', suffixes=("pi1", "pi2"))
  df = pd.DataFrame(data=merged)
  df.plot(xlabel='Sequence number', ylabel='Time since first packet sent from device',  x='seq')
  print("Done!")

def plotPacketsLost(df):
  print("Plotting packets lost... ", end='', flush=True)
  burstCount = getBurstCount(df)

  pi1bursts = [packetsLostDuringBurst(df, pi1HighPrioIp, n) for n in range(burstCount)]
  pi2bursts = [packetsLostDuringBurst(df, pi2HighPrioIp, n) for n in range(burstCount)]

  data = { 'pi1': pi1bursts, 'pi2': pi2bursts}
  df = pd.DataFrame(index=[n for n in range(1, burstCount + 1)], data=data)
  df.plot(xlabel='Burst index', ylabel='Packets lost')
  print("Done!")

def getBurstCount(df):
  return df['burst'].iloc[-1] + 1

class Measurement:
  # burstCount can be supplied if the measurements are only interesting up to a certain point. Like if the buffer becomes full on burst 200, you might not be interested in bursts 400-1000.
  def __init__(self, fileName: str, burstCount: int | None = None):
    self.df = pd.read_csv(fileName, header=0, names=['seq', 'burst', 'ip', 'sent', 'recv'])
    metaFileStr: list[str]

    with open(fileName.replace('.csv', '.meta')) as metaFile:
      metaFileStr = list(map(lambda s: s.strip(), list(metaFile)))

    variables = metaFileStr[1:]
    print(variables)
    # We want to be able to limit the burst count from the script, in case the measurements totally overshoot the amount and become unreadable.
    self.burstCount = burstCount if burstCount is not None else self.df['burst'].iloc[-1] + 1
    # Only include bursts up to burstCount.
    self.df = self.df[self.df['burst'] < self.burstCount]
    self.burstPeriodSec = int(variables[1].split('=')[1][:-2]) / 10**9
    self.pi1SvPacketsPerBurst = int(variables[2].split('=')[1])
    self.pi1IpPacketsPerBurst = int(variables[3].split('=')[1])
    self.pi2SvPacketsPerBurst = int(variables[4].split('=')[1])
    self.pi2IpPacketsPerBurst = int(variables[5].split('=')[1])
    self.svPacketSize = int(variables[6].split('=')[1])
    self.ipPacketSize = int(variables[7].split('=')[1])
    self.switchBufferSize = int(variables[8].split('=')[1])
    self.pi1HighPrioDelaySec = int(variables[9].split('=')[1][:-2]) / 10**9
    self.pi2HighPrioDelaySec = int(variables[10].split('=')[1][:-2]) / 10**9
    self.pi1LowPrioDelaySec = int(variables[11].split('=')[1][:-2]) / 10**9
    self.pi2LowPrioDelaySec = int(variables[12].split('=')[1][:-2]) / 10**9
    self.maxLatencySec = int(variables[13].split('=')[1][:-2]) / 10**9
    self.ingressSpeedBps = 100_000_000 # FIXME read from filename.
    self.egressSpeedBps = 100_000_000 # FIXME read from filename.

    pi1burstSizeInBytes = self.pi1SvPacketsPerBurst * self.svPacketSize + self.pi1IpPacketsPerBurst * self.ipPacketSize
    pi2burstSizeInBytes = self.pi2SvPacketsPerBurst * self.svPacketSize + self.pi2IpPacketsPerBurst * self.ipPacketSize
    maxBurstSizeInBytes = max(pi1burstSizeInBytes, pi2burstSizeInBytes)

    self.theoreticalBurstDurationSec = maxBurstSizeInBytes * 8 / self.ingressSpeedBps
    # FIXME: Not this simple?
    self.theoreticalFullLossLatencySec = self.maxLatencySec + (self.burstPeriodSec - self.theoreticalBurstDurationSec)

    print("Initializing a bunch of data frames, this might take a while... ", end="", flush=True)
    self.piDfs = {pi1HighPrioIp: getPiDataFrame(self.df, pi1HighPrioIp), pi2HighPrioIp: getPiDataFrame(self.df, pi2HighPrioIp), pi1LowPrioIp: getPiDataFrame(self.df, pi1LowPrioIp), pi2LowPrioIp: getPiDataFrame(self.df, pi2LowPrioIp)}
    self.burstDfs = [ getBurst(self.df, n) for n in range(self.burstCount) ]
    self.piBursts = {pi1HighPrioIp: [ getPiDataFrame(burstDf, pi1HighPrioIp) for burstDf in self.burstDfs ],  pi2HighPrioIp: [ getPiDataFrame(burstDf, pi2HighPrioIp) for burstDf in self.burstDfs ], pi1LowPrioIp: [ getPiDataFrame(burstDf, pi1LowPrioIp) for burstDf in self.burstDfs ],  pi2LowPrioIp: [ getPiDataFrame(burstDf, pi2LowPrioIp) for burstDf in self.burstDfs ] }
    print("Done!")

  df: pd.DataFrame
  piDfs: dict[str, pd.DataFrame] = {}
  burstDfs: list[pd.DataFrame] = []
  burstCount: int

  def getBurstCount(self):
    return self.burstCount

  def getPiDataFrame(self, piIp: str):
    return self.piDfs[piIp]

  def getBurst(self, burstIndex: int):
    return self.burstDfs[burstIndex]

  def getPiBurst(self, piIp: str, burstIndex: int):
    return self.piBursts[piIp][burstIndex]

  def getCumulativeReceiveTimes(self, piIp):
    pi = self.getPiDataFrame(piIp)
    piStartRecvTime = startRecvTime(pi)
    piReceiveTimes = pi.drop(columns = ['burst', 'ip', 'sent'])
    piCumulativeReceiveTimes = piReceiveTimes - [0, piStartRecvTime]
    return piCumulativeReceiveTimes

  def plotCumulativeReceiveTimes(self):
    print("Plotting cumulative receive times... ", end='', flush=True)
    pi1CumulativeReceiveTimes = self.getCumulativeReceiveTimes(pi1HighPrioIp)
    pi2CumulativeReceiveTimes = self.getCumulativeReceiveTimes(pi2HighPrioIp)

    merged = pi1CumulativeReceiveTimes.merge(pi2CumulativeReceiveTimes, on='seq', how='outer', suffixes=("pi1", "pi2"))
    df = pd.DataFrame(data=merged)
    df.plot(xlabel='Sequence number', ylabel='Time since first packet received from device (ns)', x='seq')
    print("Done!")

  def piBurstTimeToRecvAll(self, piIp, burstIndex):
    piBurst = self.getPiBurst(piIp, burstIndex)
    recv = piBurst['recv']
    if len(recv) == 0:
      return math.nan
    first = recv.iloc[0]
    last = recv.iloc[len(recv) - 1]
    return last - first

  def piBurstTimeToSendAll(self, piIp, burstIndex):
    piBurst = self.getPiBurst(piIp, burstIndex)
    if len(piBurst) == 0:
      return math.nan
    sentIndex = 3
    first = piBurst.iloc[0, sentIndex]
    last = piBurst.iloc[len(piBurst) - 1, sentIndex]
    return last - first

  def burstTimeToRecvAll(self, burstIndex):
    burst = self.getBurst(burstIndex)
    recv = burst['recv']
    if len(recv) == 0:
      return math.nan
    first = recv.iloc[0]
    last = recv.iloc[-1]
    return last - first

  def plotBurstTotalSentAndReceiveTimes(self):
    print("Plotting burst total sent and received times... ", end='', flush=True)
    burstCount = self.getBurstCount()

    pi1RecvTimes = [self.piBurstTimeToRecvAll(pi1HighPrioIp, n) for n in range(burstCount)]
    pi2RecvTimes = [self.piBurstTimeToRecvAll(pi2HighPrioIp, n) for n in range(burstCount)]
    pi1SendTimes = [self.piBurstTimeToSendAll(pi1HighPrioIp, n) for n in range(burstCount)]
    pi2SendTimes = [self.piBurstTimeToSendAll(pi2HighPrioIp, n) for n in range(burstCount)]

    totalRecvTimes = [self.burstTimeToRecvAll(n) for n in range(burstCount)]

    data = { 'pi1recv': pi1RecvTimes, 'pi2recv': pi2RecvTimes, 'pi1send': pi1SendTimes, 'pi2send': pi2SendTimes, 'totalrecv': totalRecvTimes }
    df = pd.DataFrame(data=data, index=range(burstCount))
    df.plot()
    print("Done!")

  def packetsLostDuringBurst(self, piIp, burstIndex):
    piBurst = self.getPiBurst(piIp, burstIndex)
    if piIp == pi1HighPrioIp:
      return self.pi1SvPacketsPerBurst - len(piBurst)
    elif piIp == pi2HighPrioIp:
      return self.pi2SvPacketsPerBurst - len(piBurst)
    elif piIp == pi1LowPrioIp:
      return self.pi1IpPacketsPerBurst - len(piBurst)
    elif piIp == pi2LowPrioIp:
      return self.pi2IpPacketsPerBurst - len(piBurst)
    else:
      raise Exception('Unknown piIp')

  def getPacketsLostPerBurst(self):
    pi1HighPrioLost = [ self.packetsLostDuringBurst(pi1HighPrioIp, n) for n in range(self.getBurstCount()) ]
    pi2HighPrioLost = [ self.packetsLostDuringBurst(pi2HighPrioIp, n) for n in range(self.getBurstCount()) ]
    pi1LowPrioLost = [ self.packetsLostDuringBurst(pi1LowPrioIp, n) for n in range(self.getBurstCount()) ]
    pi2LowPrioLost = [ self.packetsLostDuringBurst(pi2LowPrioIp, n) for n in range(self.getBurstCount()) ]

    data = { 'Client 1 SV packets': pi1HighPrioLost, 'Client 2 SV packets': pi2HighPrioLost, 'Client 1 IP packets': pi1LowPrioLost, 'Client 2 IP packets': pi2LowPrioLost }
    return pd.DataFrame(data=data)

  def plotPacketsLost(self, xRange = None, yRange = None):
    print("Plotting packets lost... ", end='', flush=True)
    burstCount = self.getBurstCount()

    pi1HighBursts = [self.packetsLostDuringBurst(pi1HighPrioIp, n) for n in range(burstCount)]
    pi2HighBursts = [self.packetsLostDuringBurst(pi2HighPrioIp, n) for n in range(burstCount)]
    pi1LowBursts = [self.packetsLostDuringBurst(pi1LowPrioIp, n) for n in range(burstCount)]
    pi2LowBursts = [self.packetsLostDuringBurst(pi2LowPrioIp, n) for n in range(burstCount)]

    data = {}
    if len(pi1HighBursts) > 0:
      data['Client 1 SV packets'] = pi1HighBursts
    if len(pi2HighBursts) > 0:
      data['Client 2 SV packets'] = pi2HighBursts
    if len(pi1LowBursts) > 0:
      data['Client 1 IP packets'] = pi1LowBursts
    if len(pi1LowBursts) > 0:
      data['Client 1 IP packets'] = pi2LowBursts

    df = pd.DataFrame(index=[n for n in range(1, burstCount + 1)], data=data)
    df.plot(xlim=xRange, ylim=yRange, xlabel='Burst index', ylabel='Packets lost')
    print("Done!")

  def getPiQueuingDelaySec(self, piIp):
    pi = self.getPiDataFrame(piIp)
    if piIp == pi1HighPrioIp:
      piDelaySec = self.pi1HighPrioDelaySec
    elif piIp == pi2HighPrioIp:
      piDelaySec = self.pi2HighPrioDelaySec
    elif piIp == pi1LowPrioIp:
      piDelaySec = self.pi1LowPrioDelaySec
    elif piIp == pi2LowPrioIp:
      piDelaySec = self.pi2LowPrioDelaySec
    else:
      raise Exception('unknown ip')

    piTimes = pi.drop(columns=['seq', 'ip','burst']).diff(axis=1)['recv'] - piDelaySec * 10**9
    piSeq = pi['seq']
    piIndex = 1 if piIp == pi1HighPrioIp or piIp == pi1LowPrioIp else 2
    piPrio = 'SV' if piIp == pi1HighPrioIp or piIp == pi2HighPrioIp else 'IP'
    # FIXME: Going by the sequence number only works if both Pis send the same amount of SV packets per burst.
    if piPrio == 'IP':
      svPacketsPerBurst = self.pi1SvPacketsPerBurst if piIp == pi1LowPrioIp else self.pi2SvPacketsPerBurst
      piSeq *= svPacketsPerBurst or 1

    piWithSeqDf = pd.DataFrame(data={ 'seq': piSeq, f'Client {piIndex} {piPrio} packets': piTimes / 10**9 })
    return piWithSeqDf

  def plotQueuingDelay(self):
    print('Plotting queuing delay... ', end='')
    pi1WithSeqDf = self.getPiQueuingDelaySec(pi1HighPrioIp)
    pi2WithSeqDf = self.getPiQueuingDelaySec(pi2HighPrioIp)

    # If the IP packets dataframes contain a bunch of NaNs, the plot doesn't want to show the lines.
    # To get around this, we replace the NaN values with the last recorded non-NaN value for the series.
    lastNonNanValue = math.nan
    def removeNan(x):
      nonlocal lastNonNanValue
      if math.isnan(x):
        return lastNonNanValue
      lastNonNanValue = x
      return x

    highCombined = pi1WithSeqDf.merge(pi2WithSeqDf, on='seq', how='outer')
    highCombinedSeq = highCombined['seq'].to_frame()
    combined = highCombined
    if self.pi1IpPacketsPerBurst != 0:
      pi1LowWithSeqDf = self.getPiQueuingDelaySec(pi1LowPrioIp)
      lastNonNanValue = math.nan
      pi1LowCombinedSeq = highCombinedSeq.merge(pi1LowWithSeqDf, on='seq', how='outer').map(removeNan)
      combined = combined.merge(pi1LowCombinedSeq, on='seq', how='outer')
    if self.pi2IpPacketsPerBurst != 0:
      pi2LowWithSeqDf = self.getPiQueuingDelaySec(pi2LowPrioIp)
      lastNonNanValue = math.nan
      pi2LowCombinedSeq = highCombinedSeq.merge(pi2LowWithSeqDf, on='seq', how='outer').map(removeNan)
      combined = combined.merge(pi2LowCombinedSeq, on='seq', how='outer')

    combined.insert(0, 'Lower deadline threshold', self.maxLatencySec)
    # FIXME: Figure out if this is actually correct
    combined.insert(0, 'Upper deadline threshold', self.theoreticalFullLossLatencySec)

    combined.plot(x='seq', xlabel='Sequence number', ylabel='Difference from empty queue latency (s)')
    print('Done!')

  def plotQueuingDelayScatterPlot(self, xRange=None, yRange=None):
    print('Plotting queuing delay scatter plot... ', end='')

    combined = pd.DataFrame()
    if self.pi1SvPacketsPerBurst != 0:
      pi1WithSeqDf = self.getPiQueuingDelaySec(pi1HighPrioIp).drop(columns=['seq'])
      combined = combined.merge(pi1WithSeqDf, left_index=True, right_index=True, how='outer')
    if self.pi2SvPacketsPerBurst != 0:
      pi2WithSeqDf = self.getPiQueuingDelaySec(pi2HighPrioIp).drop(columns=['seq'])
      combined = combined.merge(pi2WithSeqDf, left_index=True, right_index=True, how='outer')
    if self.pi1IpPacketsPerBurst != 0:
      pi1LowWithSeqDf = self.getPiQueuingDelaySec(pi1LowPrioIp).drop(columns=['seq'])
      combined = combined.merge(pi1LowWithSeqDf, left_index=True, right_index=True, how='outer')
    if self.pi2IpPacketsPerBurst != 0:
      pi2LowWithSeqDf = self.getPiQueuingDelaySec(pi2LowPrioIp).drop(columns=['seq'])
      combined = combined.merge(pi2LowWithSeqDf, left_index=True, right_index=True, how='outer')

    combined *= 1_000_000
    print(combined)
    finalList = []
    fig, ax = plt.subplots()
    ax.set_xlabel('Order packets were received')
    ax.set_ylabel('Difference from empty queue latency (µs)')
    if xRange is not None:
      ax.set_xlim(xRange)
    if yRange is not None:
      ax.set_ylim((yRange[0] * 1000000, yRange[1] * 1000000))
    colors = ['red', 'black', 'blue', 'orange']
    for index, key in enumerate(['Client 1 SV packets', 'Client 2 SV packets', 'Client 1 IP packets', 'Client 2 IP packets']):
      try:
        seq = combined[key]
        combined = combined.drop(columns=[key])
        newseq = seq.dropna()
        yValues = newseq.to_list()
        xValues = newseq.index.to_list()
        ax.scatter(x=xValues, y=yValues, c=colors[index], s=3, label=key)
      except KeyError:
        pass

    ax.legend(frameon=True)

  def plotQueuingDelayAbsolute(self, xRange=None, yRange=None, showLatencyLimits=False):
    print('Plotting queuing delay... ', end='')

    # If the IP packets dataframes contain a bunch of NaNs, the plot doesn't want to show the lines.
    # To get around this, we replace the NaN values with the last recorded non-NaN value for the series.
    lastNonNanValue = math.nan
    def removeNan(x):
      nonlocal lastNonNanValue
      if math.isnan(x):
        return lastNonNanValue
      lastNonNanValue = x
      return x

    combined = pd.DataFrame()

    if self.pi1SvPacketsPerBurst != 0:
      pi1WithSeqDf = self.getPiQueuingDelaySec(pi1HighPrioIp).drop(columns=['seq'])
      pi1WithSeqDf *= 1_000_000
      combined = combined.merge(pi1WithSeqDf, left_index=True, right_index=True, how='outer')
    if self.pi2SvPacketsPerBurst != 0:
      pi2WithSeqDf = self.getPiQueuingDelaySec(pi2HighPrioIp).drop(columns=['seq'])
      pi2WithSeqDf *= 1_000_000
      combined = combined.merge(pi2WithSeqDf, left_index=True, right_index=True, how='outer')
    if self.pi1IpPacketsPerBurst != 0:
      pi1LowWithSeqDf = self.getPiQueuingDelaySec(pi1LowPrioIp).drop(columns=['seq'])
      pi1LowWithSeqDf *= 1_000_000
      combined = combined.merge(pi1LowWithSeqDf, left_index=True, right_index=True, how='outer')
    if self.pi2IpPacketsPerBurst != 0:
      pi2LowWithSeqDf = self.getPiQueuingDelaySec(pi2LowPrioIp).drop(columns=['seq'])
      pi2LowWithSeqDf *= 1_000_000
      combined = combined.merge(pi2LowWithSeqDf, left_index=True, right_index=True, how='outer')

    for key in ['Client 1 SV packets', 'Client 2 SV packets', 'Client 1 IP packets', 'Client 2 IP packets']:
      try:
        seq = combined[key]
        combined = combined.drop(columns=[key])
        print(combined)
        newseq = seq.map(removeNan)
        lastNonNanValue = math.nan
        combined.insert(0, key, newseq)
      except KeyError:
        pass

    if showLatencyLimits:
      combined.insert(0, 'Lower deadline threshold', self.maxLatencySec * 1_000_000)
      # FIXME: Figure out if this is actually correct
      combined.insert(0, 'Upper deadline threshold', self.theoreticalFullLossLatencySec * 1_000_000)

    combined.plot(xlim=xRange, ylim=(yRange[0] * 1000000, yRange[1] * 1000000), xlabel='Order packets were received', ylabel='Difference from empty queue latency (µs)')
    print('Done!')

  def getDelayedPacketsPerBurst(self):
    # FIXME: Check that the outdated packets is actually calculated correctly.
    piData = [ { 'ip': pi1HighPrioIp, 'delay': self.pi1HighPrioDelaySec },  { 'ip': pi2HighPrioIp, 'delay': self.pi2HighPrioDelaySec } ]
    sums = []

    for pi in piData:
      piIp = pi['ip']
      piDelay = pi['delay']

      pi = self.getPiDataFrame(piIp)
      piTimesNs = pi.drop(columns=['seq', 'ip','burst']).diff(axis=1)['recv'] - piDelay * 10**9
      piOverflows = (piTimesNs > (self.maxLatencySec * 10**9)).map(lambda x: int(x))
      piBurst = pi['burst']
      piWithBurstDf = pd.DataFrame(data={ 'burst': piBurst, 'pioverflows': piOverflows })

      piBurstsOverflows = [ getBurst(piWithBurstDf, n)['pioverflows'] for n in range(self.getBurstCount()) ]

      piBurstSums = [ sr.sum() for sr in piBurstsOverflows ]
      sums.append(piBurstSums)

    data = { 'Client 1 SV packets': sums[0], 'Client 2 SV packets': sums[1] }
    df = pd.DataFrame(data=data)
    return df

  def plotDelayedPacketsPerBurst(self, xRange = None, yRange = None):
    print('Plotting delayed packets per burst... ', end='')
    df = self.getDelayedPacketsPerBurst()
    # FIXME: Make into percentage?
    df.plot(xlim=xRange, ylim=yRange, xlabel='Burst index', ylabel='Packets in burst that miss their deadline')
    print('Done!')

  def plotLatencyLossAndPacketLossPerBurst(self):
    print('Plotting latency loss and packet loss per burst... ', end='')
    delayedPackets = self.getDelayedPacketsPerBurst()
    lostPackets = self.getPacketsLostPerBurst()
    total = delayedPackets + lostPackets
    print(delayedPackets)
    print(lostPackets)
    print(total)
    data = { 'pi1Delayed': delayedPackets['Client 1 SV packets'], 'pi2Delayed': delayedPackets['Client 2 SV packets'], 'pi1lost': lostPackets['Client 1 SV packets'], 'pi2lost': lostPackets['Client 2 SV packets'], 'pi1total':  total['Client 1 SV packets'], 'pi2total': total['Client 2 SV packets'], 'pi1lowlost': lostPackets['Client 1 IP packets'], 'pi2lowlost': lostPackets['Client 2 IP packets'] }

    df = pd.DataFrame(data=data)
    df.plot(xlabel='burst', ylabel='packets lost')
    print('Done!')

  def plotLatencyOverflow(self):
    print('Plotting latency overflows... ', end='')
    pi1 = self.getPiDataFrame(pi1HighPrioIp)
    pi1TimesNs = pi1.drop(columns=['seq', 'ip','burst']).diff(axis=1)['recv'] - self.pi1HighPrioDelaySec * 10**9
    pi1Overflows = (pi1TimesNs > (self.maxLatencySec * 10**9)).map(lambda x: int(x))
    pi1Seq = pi1['seq']
    pi1WithSeqDf = pd.DataFrame(data={ 'seq': pi1Seq, f'pi1overflows': pi1Overflows })

    pi2 = self.getPiDataFrame(pi2HighPrioIp)
    pi2TimesNs = pi2.drop(columns=['seq', 'ip','burst']).diff(axis=1)['recv'] - self.pi2HighPrioDelaySec * 10**9
    pi2Overflows = (pi2TimesNs > (self.maxLatencySec * 10**9)).map(lambda x: int(x))
    pi2Seq = pi2['seq']
    pi2WithSeqDf = pd.DataFrame(data={ 'seq': pi2Seq, f'pi2overflows': pi2Overflows })

    combined = pi1WithSeqDf.merge(pi2WithSeqDf, on='seq', how='outer')
    if len(combined) == 0:
      print('No values to plot')
      return
    combined.plot(x='seq', xlabel='Sequence number', ylabel='Exceed maximum latency or not')
    print('Done!')

  # FIXME: This is borked. The amount of data that gets sent should be used instead of all data in the burst?
  # def getRecvBandwidthsDuringBurst(self):
    # burstRecvTimes = [ self.getBurst(n)['recv'] for n in range(self.getBurstCount()) ]
    # burstBandwidths = [ self.burstSizeInTotalPackets * self.packetSizeInBits / ((burst.iat[-1] - burst.iat[0]) / 10**9) for burst in burstRecvTimes ]
    # return burstBandwidths

  # def getMedianRecvBandwidthDuringBurst(self):
    # bandwidths = self.getRecvBandwidthsDuringBurst()
    # median = sorted(bandwidths)[len(bandwidths) // 2]
    # return median

  # def plotRecvBandwidthDuringBurst(self):
    # print('Plotting receive bandwidths during bursts... ', end='')
    # bandwidths = self.getRecvBandwidthsDuringBurst()
    # median = self.getMedianRecvBandwidthDuringBurst()
    # data = { 'recv bandwidths': bandwidths }
    # df = pd.DataFrame(data=data)
    # df.insert(0, 'median recv bw', median)
    # df.plot()
    # print('Done!')

def main():
  if len(sys.argv) < 2:
    print("Please provide some amount of csv filenames")
    exit(1)

  for fileName in sys.argv[1:]:
    m = Measurement(fileName)

    df = m.df

    saveFolderPath = f"../images/{fileName.split('/')[-1][:-4]}"
    try:
      os.mkdir(saveFolderPath)
    except FileExistsError:
      pass


    packetLossXRange = None
    packetLossYRange = None
    outdatedXRange = None
    outdatedYRange = None
    latencyXRange = None
    latencyYRange = None
    showLatencyLimits = False

    packetLegendLoc = 'best'
    outdatedLegendLoc = 'best'
    latencyLegendLoc = 'best'

    oneAndTwoZoomedXRange = (-50,1400)
    oneAndTwoZoomedYRange =  (-0.00005, 0.0003)

    if "01" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,9)
      outdatedXRange = None
      outdatedYRange = (-1,9)
      latencyXRange = oneAndTwoZoomedXRange
      latencyYRange = oneAndTwoZoomedYRange
    elif "02" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,9)
      outdatedXRange = None
      outdatedYRange = (-1,9)
      latencyXRange = None
      latencyYRange = (-0.0005, 0.011)
    elif "04" in fileName:
      packetLossXRange = None
      packetLossYRange = None
      outdatedXRange = None
      outdatedYRange = None
      latencyXRange = None
      latencyYRange = (-0.0005, 0.011)
    elif "09" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,9)
      outdatedXRange = None
      outdatedYRange = (-1,9)
      latencyXRange = (-500,10000)
      latencyYRange = (-0.00004,0.0005)
      showLatencyLimits = True
    elif "11" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,9)
      outdatedXRange = None
      outdatedYRange = (-1,9)
      latencyXRange = (-500,10000)
      latencyYRange = (-0.00004,0.0005)
      showLatencyLimits = True
    elif "12" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,9)
      outdatedXRange = None
      outdatedYRange = (-1,9)
      latencyXRange = (-500,10000)
      latencyYRange = (-0.00004,0.0005)
      showLatencyLimits = True
    elif "17" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,8)
      outdatedXRange = None
      outdatedYRange = (-1,8)
      latencyXRange = None
      latencyYRange = (-0.00006,0.0015)
    elif "18" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,8)
      outdatedXRange = None
      outdatedYRange = (-1,8)
      latencyXRange = None
      latencyYRange = (-0.00006,0.0015)
    elif "19" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,8)
      outdatedXRange = None
      outdatedYRange = (-1,8)
      latencyXRange = None
      latencyYRange = (-0.00006,0.0015)
    elif "20" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,8)
      outdatedXRange = None
      outdatedYRange = (-1,8)
      latencyXRange = None
      latencyYRange = (-0.00006,0.0015)
    elif "21" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,8)
      outdatedXRange = None
      outdatedYRange = (-1,8)
      latencyXRange = None
      latencyYRange = (-0.00006,0.0015)
    elif "22" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,8)
      outdatedXRange = None
      outdatedYRange = (-1,8)
      latencyXRange = None
      latencyYRange = (-0.00006,0.0015)
    elif "23" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,8)
      outdatedXRange = None
      outdatedYRange = (-1,8)
      latencyXRange = None
      latencyYRange = (-0.00006,0.0015)
    elif "24" in fileName:
      packetLossXRange = None
      packetLossYRange = (-1,8)
      outdatedXRange = None
      outdatedYRange = (-1,8)
      latencyXRange = None
      latencyYRange = (-0.00006,0.0015)

    # TODO: Is jitter interesting??

    # m.plotLatencyLossAndPacketLossPerBurst()
    # plt.savefig(f'{saveFolderPath}/plotLatencyLossAndPacketLossPerBurst.png')
    plt.legend(loc=outdatedLegendLoc)
    m.plotDelayedPacketsPerBurst(outdatedXRange, outdatedYRange)
    plt.savefig(f'{saveFolderPath}/plotDelayedPacketsPerBurst.pdf', format='pdf')

    # m.plotRecvBandwidthDuringBurst()
    # plt.savefig(f'{saveFolderPath}/plotRecvBandwidthDuringBurst.png')

    # m.plotCumulativeReceiveTimes()
    # plt.savefig(f'{saveFolderPath}/plotCumulativeReceiveTimes.png')
    # plotCumulativeSendTimes(df)
    # plt.savefig(f'{saveFolderPath}/plotCumulativeSendTimes.png')
    plt.legend(loc=packetLegendLoc)
    m.plotPacketsLost(packetLossXRange, packetLossYRange)
    plt.savefig(f'{saveFolderPath}/plotPacketsLost.pdf', format='pdf')

    # plotTimeBetweenReceivedPackets(df)
    # plt.savefig(f'{saveFolderPath}/plotTimeBetweenReceivedPackets.png')
    # plotTimeBetweenSentPackets(df)
    # plt.savefig(f'{saveFolderPath}/plotTimeBetweenSentPackets.png')

    # m.plotBurstTotalSentAndReceiveTimes()
    # plt.savefig(f'{saveFolderPath}/plotBurstTotalSentAndReceiveTimes.png')

    # m.plotQueuingDelay()
    # plt.savefig(f'{saveFolderPath}/plotQueuingDelay.png')


    plt.legend(loc=latencyLegendLoc)
    m.plotQueuingDelayAbsolute(latencyXRange, latencyYRange, showLatencyLimits)
    plt.savefig(f'{saveFolderPath}/plotQueuingDelayAbsolute.pdf', format='pdf')
    #plt.show()

    # m.plotLatencyOverflow()

    if "02" in fileName:
      packetLossXRange = None
      packetLossYRange = None
      outdatedXRange = None
      outdatedYRange = None
      latencyXRange = oneAndTwoZoomedXRange
      latencyYRange = oneAndTwoZoomedYRange
      packetLegendLoc = 'best'
      latencyLegendLoc = 'best'
      outdatedLegendLoc = 'best'

      m.plotDelayedPacketsPerBurst(outdatedXRange, outdatedYRange)
      plt.savefig(f'{saveFolderPath}/zoomedPlotDelayedPacketsPerBurst.pdf', format = 'pdf')

      m.plotPacketsLost(packetLossXRange, packetLossYRange)
      plt.savefig(f'{saveFolderPath}/zoomedPlotPacketsLost.pdf', format = 'pdf')

      m.plotQueuingDelayAbsolute(latencyXRange, latencyYRange, showLatencyLimits)
      plt.savefig(f'{saveFolderPath}/zoomedPlotQueuingDelayAbsolute.pdf', format = 'pdf')

      singleBurstLatencyXRange = (1598,1619)
      singleBurstLatencyYRange = (0.00013, 0.0004)
      m.plotQueuingDelayScatterPlot(xRange=singleBurstLatencyXRange, yRange=singleBurstLatencyYRange)
      plt.savefig(f'{saveFolderPath}/oneBurstDelayZoom.pdf', format = 'pdf')
      #plt.show()
      

    # Calculate packets lost
    pi1ExpectedSvPacketAmount = m.pi1SvPacketsPerBurst * m.burstCount
    pi1SvPacketAmount = len(m.getPiDataFrame(pi1HighPrioIp))
    pi1SvPacketsLost = pi1ExpectedSvPacketAmount - pi1SvPacketAmount 

    pi2ExpectedSvPacketAmount = m.pi2SvPacketsPerBurst * m.burstCount
    pi2SvPacketAmount = len(m.getPiDataFrame(pi2HighPrioIp))
    pi2SvPacketsLost = pi2ExpectedSvPacketAmount - pi2SvPacketAmount 

    pi1ExpectedIpPacketAmount = m.pi1IpPacketsPerBurst * m.burstCount
    pi1IpPacketAmount = len(m.getPiDataFrame(pi1LowPrioIp))
    pi1IpPacketsLost = pi1ExpectedIpPacketAmount - pi1IpPacketAmount 

    pi2ExpectedIpPacketAmount = m.pi2IpPacketsPerBurst * m.burstCount
    pi2IpPacketAmount = len(m.getPiDataFrame(pi2LowPrioIp))
    pi2IpPacketsLost = pi2ExpectedIpPacketAmount - pi2IpPacketAmount

    print(f'client 1 sv packets lost: {pi1SvPacketsLost}')
    print(f'client 2 sv packets lost: {pi2SvPacketsLost}')
    print(f'client 1 ip packets lost: {pi1IpPacketsLost}')
    print(f'client 2 ip packets lost: {pi2IpPacketsLost}')

    # Calculate packets outdated

    foo = m.getDelayedPacketsPerBurst()
    pi1HighPrio = foo['Client 1 SV packets']
    pi1outdatedsum = pi1HighPrio.sum()
    pi2HighPrio = foo['Client 2 SV packets']
    pi2outdatedsum = pi2HighPrio.sum()

    print(f'client 1 high prio outdated: {pi1outdatedsum}')
    print(f'client 2 high prio outdated: {pi2outdatedsum}')
    newMetaFilename = fileName[:-3] + 'meta2'
    with open(newMetaFilename, 'w') as f:
      f.write(f'pi1svpacketloss = {pi1SvPacketsLost} / {pi1ExpectedSvPacketAmount}\n')
      f.write(f'pi2svpacketloss = {pi2SvPacketsLost} / {pi2ExpectedSvPacketAmount}\n')
      f.write(f'pi1ippacketloss = {pi1IpPacketsLost} / {pi1ExpectedIpPacketAmount}\n')
      f.write(f'pi2ippacketloss = {pi2IpPacketsLost} / {pi2ExpectedIpPacketAmount}\n')
      f.write(f'pi1svpacketoutdated = {pi1outdatedsum} / {pi1ExpectedSvPacketAmount}\n')
      f.write(f'pi2svpacketoutdated = {pi2outdatedsum} / {pi2ExpectedSvPacketAmount}\n')
      f.write(f'fulldelaylatency = {m.theoreticalFullLossLatencySec}\n')


    # TODO: Add line for the expected burst when the buffer should first fill up? (ceil(bufSize / bits in buffer after period))

    # Find difference between first sent packet time and first received packet time (maybe not the first but among the first couple, since the first might be slower (cold cache?)).
    # Graph the differences between received time and sent time for respective sequence numbers.

    # FIXME: How can we know the absolute latency of one packet? Can we only measure the relative latency?
    # Maybe only focus on queueing latency relative to the first packet received from a device.
    # Can separate between first packet received at all and first packet received during a burst
    # (If the buffer is not emptied between bursts, the first packet received during a burst will already have a queueing delay).

    #plt.show()
    plt.close('all')

  # Create diagram for how packet loss / outdated changes for buffer sizes
  smallBufferMeasurements = Measurement('../measurements/12-hardlimitatnolatencyloss.csv')
  mediumBufferMeasurements = Measurement('../measurements/11-hardlimitatsomelatencyloss.csv')
  largeBufferMeasurements = Measurement('../measurements/09-hardlimitatfulllatencyloss.csv')

  results = []
  for m in [smallBufferMeasurements, mediumBufferMeasurements, largeBufferMeasurements]:
    # Calculate packets lost
    pi1ExpectedSvPacketAmount = m.pi1SvPacketsPerBurst * m.burstCount
    pi1SvPacketAmount = len(m.getPiDataFrame(pi1HighPrioIp))
    pi1SvPacketsLost = pi1ExpectedSvPacketAmount - pi1SvPacketAmount 
    pi1SvPacketsLostAsPercent = pi1SvPacketsLost / pi1ExpectedSvPacketAmount * 100

    pi2ExpectedSvPacketAmount = m.pi2SvPacketsPerBurst * m.burstCount
    pi2SvPacketAmount = len(m.getPiDataFrame(pi2HighPrioIp))
    pi2SvPacketsLost = pi2ExpectedSvPacketAmount - pi2SvPacketAmount 
    pi2SvPacketsLostAsPercent = pi2SvPacketsLost / pi2ExpectedSvPacketAmount * 100

    pi1ExpectedIpPacketAmount = m.pi1IpPacketsPerBurst * m.burstCount
    pi1IpPacketAmount = len(m.getPiDataFrame(pi1LowPrioIp))
    pi1IpPacketsLost = pi1ExpectedIpPacketAmount - pi1IpPacketAmount 
    pi1IpPacketsLostAsPercent = pi1IpPacketsLost / pi1ExpectedIpPacketAmount * 100

    pi2ExpectedIpPacketAmount = m.pi2IpPacketsPerBurst * m.burstCount
    pi2IpPacketAmount = len(m.getPiDataFrame(pi2LowPrioIp))
    pi2IpPacketsLost = pi2ExpectedIpPacketAmount - pi2IpPacketAmount
    pi2IpPacketsLostAsPercent = pi2IpPacketsLost / pi2ExpectedIpPacketAmount * 100

    print(f'client 1 sv packets lost: {pi1SvPacketsLost}')
    print(f'client 2 sv packets lost: {pi2SvPacketsLost}')
    print(f'client 1 ip packets lost: {pi1IpPacketsLost}')
    print(f'client 2 ip packets lost: {pi2IpPacketsLost}')

    # Calculate packets outdated

    delayedPackets = m.getDelayedPacketsPerBurst()
    pi1HighPrio = delayedPackets['Client 1 SV packets']
    pi1outdatedsum = pi1HighPrio.sum()
    pi1outdatedsumaspercent = pi1outdatedsum / pi1ExpectedSvPacketAmount * 100
    pi2HighPrio = delayedPackets['Client 2 SV packets']
    pi2outdatedsum = pi2HighPrio.sum()
    pi2outdatedsumaspercent = pi2outdatedsum / pi2ExpectedSvPacketAmount * 100

    results.append({'pi1svpacketslost': pi1SvPacketsLostAsPercent, 'pi2svpacketslost': pi2SvPacketsLostAsPercent, 'pi1ippacketslost': pi1IpPacketsLostAsPercent, 'pi2ippacketslost': pi2IpPacketsLostAsPercent, 'pi1outdatedpackets': pi1outdatedsumaspercent,  'pi2outdatedpackets': pi2outdatedsumaspercent })

  pi1svlostperbuffersize = [result['pi1svpacketslost'] for result in results]
  pi2svlostperbuffersize = [result['pi2svpacketslost'] for result in results]
  pi1iplostperbuffersize = [result['pi1ippacketslost'] for result in results]
  pi2iplostperbuffersize = [result['pi2ippacketslost'] for result in results]
  packetLossIndex = ['12', '17', '21']
  packetsLostData = { 'Buffer size in number of packets': packetLossIndex, 'Client 1 lost packets': pi1svlostperbuffersize, 'Client 2 lost packets': pi2svlostperbuffersize }
  packetsLost = pd.DataFrame(data = packetsLostData, index = packetLossIndex)
  ax = packetsLost.plot.bar(ylim=(0,1.2), x='Buffer size in number of packets')
  ax.set_ylabel('Percentage of packets')
  plt.savefig(f'../images/packetlossbybuffersize.pdf', format = 'pdf')

  pi1ExpectedSvPacketAmount = m.pi1SvPacketsPerBurst * m.burstCount
  pi2ExpectedSvPacketAmount = m.pi2SvPacketsPerBurst * m.burstCount
  pi1delayedperbuffersize = [result['pi1outdatedpackets'] for result in results]
  pi2delayedperbuffersize = [result['pi2outdatedpackets'] for result in results]
  packetLossIndex = ['12', '17', '21']
  packetsLostData = { 'Buffer size in number of packets': packetLossIndex, 'Client 1 packets missing deadline': pi1delayedperbuffersize, 'Client 2 packets missing deadline': pi2delayedperbuffersize }
  packetsLost = pd.DataFrame(data = packetsLostData, index = packetLossIndex)
  assert(pi1ExpectedSvPacketAmount == pi2ExpectedSvPacketAmount)
  ax = packetsLost.plot.bar(ylim=(0,100), x='Buffer size in number of packets')
  ax.set_ylabel('Percentage of packets')
  plt.savefig(f'../images/outdatedpacketsbybuffersize.pdf', format = 'pdf')

if __name__ == "__main__":
  main()
