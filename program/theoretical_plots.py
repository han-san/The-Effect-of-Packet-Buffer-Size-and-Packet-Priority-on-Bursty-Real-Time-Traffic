#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os
import scienceplots

plt.style.use(['science', 'ieee'])
# Tight layout
plt.rcParams["figure.autolayout"]  = True

# These don't need to be fully accurate as they should just give an example of how the maximum allowed latency can be reached in different scenarios.
def get_max_allowed_latency():
    return 150


def get_latency_without_queue_delay():
    return 30


# Assumes: S_Buffer > L_Max.
def plot_burst_size_with_latency():
    burst_sizes = list(range(150))
    max_y = 200
    min_latency = get_latency_without_queue_delay()
    latency_per_size = (max_y - min_latency) / 150
    latencies = [x * latency_per_size + min_latency for x in burst_sizes]
    fig, ax = plt.subplots()
    ax.set_ylabel("Latency (µs)")
    ax.set_xlabel("Burst size (packets)")
    max_latency_line = get_max_allowed_latency()
    dropping_packets_line = max_latency_line + max_latency_line / 6
    ax.hlines(
        dropping_packets_line,
        burst_sizes[0],
        burst_sizes[-1],
        label="Packet loss due to full buffer",
        colors="r",
        linestyles="dotted",
    )
    # x_line_at = next(i for i, v in enumerate(latencies) if v >= dropping_packets_line)
    # ax.vlines(
    #    x_line_at,
    #    latencies[0],
    #    dropping_packets_line,
    #    label="Burst size > Buffer size",
    #    colors="k",
    #    linestyles="dashdot",
    # )
    ax.plot(burst_sizes, latencies)
    ax.legend()
    fig.tight_layout()


def plot_queue_delay_with_buffer_size():
    buffer_sizes = list(range(500))
    queue_delay_per_size = 0.4
    queue_delays = [x * queue_delay_per_size for x in buffer_sizes]
    fig, ax = plt.subplots()
    ax.set_ylabel("Maximum queue delay (µs)")
    ax.set_xlabel("Buffer size (packets)")
    max_delay_line = get_max_allowed_latency() - get_latency_without_queue_delay()
    ax.hlines(
        max_delay_line,
        buffer_sizes[0],
        buffer_sizes[-1],
        label="Deadline missed",
        colors="y",
        linestyles="dashed",
    )
    # x_line_at = next(i for i, v in enumerate(queue_delays) if v >= max_delay_line)
    # ax.vlines(
    #    x_line_at,
    #    queue_delays[0],
    #    max_delay_line,
    #    label="Buffer size > Lmax",
    #    colors="k",
    #    linestyles="dashdot",
    # )
    ax.plot(buffer_sizes, queue_delays)
    ax.legend()
    fig.tight_layout()


# Return the first x which has a y value matching the point at which the latency has dropped to after a burst.
def get_i_of_x_with_latency_drop_value(bursts, drop_value, incline, min_latency):
    for i in range(len(bursts)):
        # The "1.0" is needed for the next burst to properly align with the current latency. No idea why.
        if (incline * bursts[i] + min_latency) >= drop_value + 1.0:
            return i
    return -1


def get_scaling_for_bursts(
    incline, min_latency, bursts, latency_decrease_between_bursts
):
    scaled = []
    saved_peak = False
    peak_val = 0.0
    i_for_burst_climb = 0
    latency_drop_val = 0.0
    for i in range(len(bursts)):
        if bursts[i] % 1 > 0.95:
            if not saved_peak:
                peak_val = scaled[-1]
                latency_drop_val = peak_val - latency_decrease_between_bursts
                saved_peak = True
            scaled.append(latency_drop_val)
        else:
            if saved_peak:
                i_for_burst_climb = get_i_of_x_with_latency_drop_value(
                    bursts, latency_drop_val, incline, min_latency
                )
                saved_peak = False
            scaled.append(incline * bursts[i_for_burst_climb] + min_latency)
            i_for_burst_climb += 1
    return scaled


# Assumes S_Buffer > L_Max. And the buffer is not cleared between bursts.
def plot_latency_with_cumulative_bursts_for_filling_buffer():
    sum_bursts = 10
    bursts = np.linspace(1, sum_bursts, 500)
    latency_at_last_burst = 200
    latency_decrease_between_bursts = 12.0
    min_latency = get_latency_without_queue_delay()
    incline = 1.6 * (latency_at_last_burst - min_latency) / sum_bursts
    latency_scaling = np.array(
        get_scaling_for_bursts(
            incline, min_latency, bursts, latency_decrease_between_bursts
        )
    )
    latency_sawtooth = latency_scaling + signal.sawtooth(bursts * 2 * np.pi)
    x_values = []
    for i in range(len(bursts)):
        if bursts[i] % 1 > 0.95:
            x_values.append(latency_scaling[i])
        else:
            x_values.append(latency_sawtooth[i])
    fig, ax = plt.subplots()
    ax.set_ylabel("Latency (µs)")
    ax.set_xlabel("Burst")

    # Uncommented as deadline is not relevant at that point in the report.
    # ax.hlines(
    #    get_max_allowed_latency(),
    #    bursts[0],
    #    bursts[-1],
    #    label="Deadline missed",
    #    colors="y",
    #    linestyles="dashed",
    # )
    ax.plot(bursts, x_values)
    # ax.legend()
    fig.tight_layout()


def main():
    saveDirPath = "../images/theoretical_figures/"
    try:
        os.mkdir(saveDirPath)
    except FileExistsError:
        pass

    plot_burst_size_with_latency()
    plt.savefig(saveDirPath + "burst-size_latency.pdf", format="pdf")
    plot_queue_delay_with_buffer_size()
    plt.savefig(saveDirPath + "buffer-size_queue-delay.pdf", format="pdf")
    plot_latency_with_cumulative_bursts_for_filling_buffer()
    plt.savefig(saveDirPath + "cumulative-filling-bursts_latency.pdf", format="pdf")
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
