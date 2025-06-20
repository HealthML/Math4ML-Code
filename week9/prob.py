#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
# ]
# ///

import matplotlib.pyplot as plt

hourly_demand = {
    0: 5, 1: 3, 2: 2, 3: 2, 4: 1, 5: 4,
    6: 10, 7: 20, 8: 35, 9: 30, 10: 25, 11: 28,
    12: 40, 13: 38, 14: 35, 15: 45, 16: 50, 17: 60,
    18: 70, 19: 65, 20: 55, 21: 40, 22: 25, 23: 10
}

def compute_pmf(demand_dict):
    # Calculate the probability mass function (PMF) from demand counts
    return None

def compute_cdf(pmf):
    # Compute the cumulative distribution function (CDF) from the PMF
    return None

def expected_value(pmf):
    # Calculate the expected value (mean) of the distribution
    return None

def variance(pmf, expected_val):
    # Calculate the variance of the distribution
    return None

def median(cdf):
    # Find the median hour where cumulative probability reaches 0.5 or more
    return None 


def plot_pmf_cdf(pmf, cdf, expected_val, var, med):
    hours = sorted(pmf.keys())
    pmf_vals = [pmf[h] for h in hours]
    cdf_vals = [cdf[h] for h in hours]

    colors = ['skyblue' for _ in hours]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.bar(hours, pmf_vals, color=colors)
    plt.axvline(expected_val, color='r', linestyle='--', label='Expected Hour')
    plt.axvline(med, color='g', linestyle=':', label='Median Hour')
    plt.title('PMF of Ride Demand')
    plt.xlabel('Hour of Day')
    plt.ylabel('Probability')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hours, cdf_vals, marker='o', color='green')
    plt.axvline(expected_val, color='r', linestyle='--', label='Expected Hour')
    plt.axvline(med, color='g', linestyle=':', label='Median Hour')
    plt.title('CDF of Ride Demand')
    plt.xlabel('Hour of Day')
    plt.ylabel('Cumulative Probability')
    plt.legend()

    plt.suptitle(f'Expected Hour: {expected_val:.2f}, Variance: {var:.2f}, Median: {med}')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Your task is to implement key statistical functions—compute_pmf, compute_cdf, 
    # expected_value, variance, and median—to analyze hourly ride demand data.
    pmf = compute_pmf(hourly_demand)
    cdf = compute_cdf(pmf)
    ev = expected_value(pmf)
    var = variance(pmf, ev)
    med = median(cdf)

    print(f"Expected Value (Hour): {ev:.2f}")
    print(f"Variance: {var:.2f}")
    print(f"Median Hour: {med}")

    plot_pmf_cdf(pmf, cdf, ev, var, med)

