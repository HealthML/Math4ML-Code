#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "pandas",
#     "tqdm",
#     "numpy",
# ]
# ///

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns

HERE = Path(__file__).parent


def load_data():
    gps_df = pd.read_csv(HERE / "GPS.csv")
    imu_df = pd.read_csv(HERE / "IMU.csv")
    imu2_df = pd.read_csv(HERE / "IMU2.csv")
    tri_df = pd.read_csv(HERE / "TRI.csv")
    tri2_df = pd.read_csv(HERE / "TRI2.csv")

    gps_df["type"] = "GPS"
    imu_df["type"] = "IMU"
    imu2_df["type"] = "IMU2"
    tri_df["type"] = "TRI"
    tri2_df["type"] = "TRI2"

    cdf = pd.concat([gps_df, imu_df, imu2_df, tri_df, tri2_df])
    return cdf


def plot_cov_ellipse(mean, cov, n_std=1.0, ax=None, **kwargs):
    """Plots an n-std ellipse based on the 2D covariance matrix."""
    if ax is None:
        ax = plt.gca()

    # Eigen decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Angle of rotation in degrees
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height of ellipse = 2 * sqrt(eigenvalue) * n_std
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        edgecolor="red",
        facecolor="none",
        **kwargs,
    )

    ax.add_patch(ellipse)
    ax.plot(*mean, marker="o", color="black")
    return ax


def plot_position_estimate(df):
    """I take a position estimate as a dataframe with columns [x,y] and plot the joint marginal histogram.
    Hint: the position estimates are a collection of estimated means.
    """
    mean_x, mean_y = df.mean()

    # Create the jointplot
    g = sns.jointplot(
        data=df, x="x", y="y", kind="scatter", marginal_kws=dict(bins=30, fill=True)
    )

    # Add a star marker at the mean point on the joint axes
    g.ax_joint.plot(
        mean_x,
        mean_y,
        marker="*",
        color="red",
        markersize=15,
        label=f"Mean {mean_x:.2f},{mean_y:.2f}",
    )

    # Add vertical line on the marginal x histogram (top plot)
    g.ax_marg_x.axvline(mean_x, color="red", linestyle="--")

    # Add horizontal line on the marginal y histogram (right plot)
    g.ax_marg_y.axhline(mean_y, color="red", linestyle="--")

    # Add legend for the star marker
    g.ax_joint.legend()

    plt.savefig("plotting-result.png")
    plt.show()


def smooth_trajectory(points, window_size=5):
    """I take a 2D ndarray and smooth it using a moving average approach."""
    if window_size < 2:
        return points

    # Pad the trajectory at the start and end to preserve length
    pad = window_size // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")

    # Compute moving average
    smoothed = np.convolve(
        padded[:, 0], np.ones(window_size) / window_size, mode="valid"
    )
    smoothed_y = np.convolve(
        padded[:, 1], np.ones(window_size) / window_size, mode="valid"
    )

    return np.column_stack((smoothed, smoothed_y))


def plot_trajectory_estimate(trajectory: list):
    """I take a trajectory (list of t,mu,cov tuples) and plot them!
    Hint: I take the mu values and smooth them with a moving average to make the trajectory less ragged.
    """
    locs = np.array([p[1] for p in trajectory])
    locs = smooth_trajectory(locs)
    fig = plt.plot(locs[:, 0], locs[:, 1])
    for pos in trajectory:
        t, mean, cov = pos
        plot_cov_ellipse(mean, cov, n_std=1)
        plot_cov_ellipse(mean, cov, n_std=2, linestyle="dashed")
    return fig


def estimate_position(df, trials=50):
    """I should output a dataframe with columns x,y
    representing the calculated mean position after resampling the data for each trial in trials.
    It is important that I output this as a dataframe :)
    """
    pass


def estimate_trajectory_over_t(df):
    """I should output a list of tuples each containing:
    - the timestep t
    - the estimated position at t (mean)
    - the estimated uncertainty over that position (covariance)
    """
    trajectory = []
    for t in tqdm(np.sort(df.t.unique())):
        # fill me in
        # Hint: I should make use of the estimate_position function ;)
        continue
    return trajectory


if __name__ == "__main__":
    cdf = load_data()
    plot_position_estimate(estimate_position(cdf[cdf.t == 0.0], trials=2000))
    fig = plot_trajectory_estimate(estimate_trajectory_over_t(cdf))
    # pd.read_csv(HERE / "true_trajectory.csv").plot(x="x", y="y", ax=plt.gca())
    plt.show()
