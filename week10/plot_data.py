#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "pandas",
# ]
# ///

"""This file helps with plotting parts of the data, with each sensor gettings its own symbol
while the color represets the passage of time."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
import math

gps_df = pd.read_csv(Path(__file__).parent / "GPS.csv")
imu_df = pd.read_csv(Path(__file__).parent / "IMU.csv")
imu2_df = pd.read_csv(Path(__file__).parent / "IMU2.csv")
tri_df = pd.read_csv(Path(__file__).parent / "TRI.csv")
tri2_df = pd.read_csv(Path(__file__).parent / "TRI2.csv")

gps_df["type"] = "GPS"
imu_df["type"] = "IMU"
imu2_df["type"] = "IMU2"
tri_df["type"] = "TRI"
tri2_df["type"] = "TRI2"

cdf = pd.concat([gps_df, imu_df, imu2_df, tri_df, tri2_df])

marker_cycle = cycle(["o", "s", "^", "D", "v", "*", "x", "+", "p", "h"])
fig, ax = plt.subplots(figsize=(20, 15), dpi=96)
for t, group in cdf.groupby("type"):
    marker = next(marker_cycle)
    ax.scatter(group["x"], group["y"], label=t, c=group["t"], marker=marker)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(title="Type")
plt.savefig("plotted_data.png")
plt.show()
