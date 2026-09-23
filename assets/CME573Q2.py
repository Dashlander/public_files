#!/usr/bin/env python3
"""
Plot 2theta vs counts data, find peaks, fit each peak with a Gaussian
(plus a local linear background), and report the FWHM.

Usage:
    python peak_fit.py data.csv
    python peak_fit.py data.csv --prominence 50 --distance 10 --out results.csv

Requires: numpy, pandas, scipy, matplotlib
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.3548


def gaussian_with_bg(x, amp, center, sigma, slope, offset):
    """Gaussian peak on top of a linear background."""
    return amp * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2)) + slope * x + offset


def load_data(path):
    """Load a two-column CSV. Works with or without a header row."""
    df = pd.read_csv(path)
    # If the first row was actually numeric data, re-read without a header
    try:
        [float(c) for c in df.columns[:2]]
        df = pd.read_csv(path, header=None)
    except ValueError:
        pass
    df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values(df.columns[0])
    return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()


def find_data_peaks(x, y, prominence=None, distance=None, height=None):
    """Locate peaks. Default prominence is 5% of the data range."""
    if prominence is None:
        prominence = 0.05 * (y.max() - y.min())
    peaks, props = find_peaks(y, prominence=prominence, distance=distance, height=height)
    return peaks, props


def fit_peak(x, y, peak_idx, window_factor=3.0):
    """Fit a single peak with a Gaussian + linear background in a local window."""
    # Rough width estimate (in samples) at half the peak's prominence
    width_samples = peak_widths(y, [peak_idx], rel_height=0.5)[0][0]
    half_win = max(int(np.ceil(window_factor * width_samples)), 5)
    lo = max(peak_idx - half_win, 0)
    hi = min(peak_idx + half_win + 1, len(x))
    xw, yw = x[lo:hi], y[lo:hi]

    # Initial guesses
    dx = np.median(np.diff(x))
    bg0 = min(yw[0], yw[-1])
    amp0 = y[peak_idx] - bg0
    sigma0 = max(width_samples * dx / SIGMA_TO_FWHM, dx)
    slope0 = (yw[-1] - yw[0]) / (xw[-1] - xw[0]) if xw[-1] != xw[0] else 0.0
    offset0 = yw[0] - slope0 * xw[0]
    p0 = [amp0, x[peak_idx], sigma0, slope0, offset0]

    bounds = (
        [0, xw[0], dx / 10, -np.inf, -np.inf],
        [np.inf, xw[-1], xw[-1] - xw[0], np.inf, np.inf],
    )
    popt, pcov = curve_fit(gaussian_with_bg, xw, yw, p0=p0, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, (lo, hi)


def main():
    parser = argparse.ArgumentParser(description="Peak finding and Gaussian FWHM fitting.")
    parser.add_argument("csv", help="Path to CSV with columns: 2theta, counts")
    parser.add_argument("--prominence", type=float, default=None,
                        help="Minimum peak prominence (default: 5%% of counts range)")
    parser.add_argument("--distance", type=int, default=None,
                        help="Minimum spacing between peaks, in data points")
    parser.add_argument("--height", type=float, default=None, help="Minimum peak height")
    parser.add_argument("--window", type=float, default=3.0,
                        help="Fit window half-width as a multiple of the rough FWHM")
    parser.add_argument("--out", default="peak_fit_results.csv", help="Output CSV for fit results")
    parser.add_argument("--plot", default="peak_fit.png", help="Output image for the plot")
    args = parser.parse_args()

    x, y = load_data(args.csv)

    # 1. Plot the raw data
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, y, color="black", lw=1, label="Data")

    # 2. Find peaks
    peaks, _ = find_data_peaks(x, y, args.prominence, args.distance, args.height)
    print(f"Found {len(peaks)} peak(s).\n")
    ax.plot(x[peaks], y[peaks], "rv", ms=7, label="Detected peaks")

    # 3. Fit each peak and compute FWHM
    results = []
    for i, p in enumerate(peaks, start=1):
        try:
            popt, perr, (lo, hi) = fit_peak(x, y, p, args.window)
        except (RuntimeError, ValueError) as e:
            print(f"Peak {i} at 2θ={x[p]:.4f}: fit failed ({e})")
            continue

        amp, center, sigma, slope, offset = popt
        fwhm = SIGMA_TO_FWHM * abs(sigma)
        fwhm_err = SIGMA_TO_FWHM * perr[2]
        results.append({
            "peak": i,
            "2theta_center": center,
            "center_err": perr[1],
            "amplitude": amp,
            "sigma": abs(sigma),
            "FWHM": fwhm,
            "FWHM_err": fwhm_err,
        })

        xf = np.linspace(x[lo], x[hi - 1], 400)
        ax.plot(xf, gaussian_with_bg(xf, *popt), lw=1.5,
                label=f"Fit {i}: 2θ={center:.3f}, FWHM={fwhm:.4f}")

        # Mark the FWHM on the plot
        half = amp / 2.0 + slope * center + offset
        ax.hlines(half, center - fwhm / 2, center + fwhm / 2, colors="gray", linestyles="--", lw=1)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        pd.set_option("display.float_format", "{:.5f}".format)
        print(res_df.to_string(index=False))
        res_df.to_csv(args.out, index=False)
        print(f"\nResults saved to {args.out}")

    ax.set_xlabel("2θ (degrees)")
    ax.set_ylabel("Counts")
    ax.set_title("Peak detection and Gaussian fits")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=200)
    print(f"Plot saved to {args.plot}")
    plt.show()


if __name__ == "__main__":
    main()
