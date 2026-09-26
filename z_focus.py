#!/usr/bin/env python3
"""Focal scan analysis: fit the spot width at each focal position and save the plots as PDF."""
import argparse
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.special import erf, erfc


def erfc_model(x, p0, p1, p2, p3):
    return (p0 / 2.0) * erfc((x - p1) / p2) + p3


def erf_model(x, p0, p1, p2, p3):
    return (p0 / 2.0) * erf((x - p1) / p2) + p3


def abs_model(x, norm, x0, q):
    return norm * abs(x - x0) + q


def output_path(input_path: Path) -> tuple[Path, str]:
    parts = input_path.parts
    if "data" not in parts:
        raise ValueError(f"No 'data' folder in input path: {input_path}")
    idx = len(parts) - 1 - parts[::-1].index("data")
    subdirs = parts[idx + 1:-1]
    if len(subdirs) < 2:
        raise ValueError(f"Expected data/<sample dirs>/<measurement>/<file>, got: {input_path}")
    sample = "_".join(subdirs[:-1])
    measurement = subdirs[-1]
    out = Path(*parts[:idx], "plots", sample, measurement, input_path.stem + ".pdf")
    return out, sample


def extract_focus(filepath: Path, sample: str, scan_coord: str, focal_coord: str,
                  time_integ: float, cmap_name: str):
    cmap = plt.get_cmap(cmap_name)

    with uproot.open(filepath) as infile:
        tree = infile["ch0"]
        volt = tree["volt"].array(library="ak")
        time = tree["time"].array(library="ak")
        tleft = tree["tleft"].array(library="ak")
        bline_mean = tree["BlineMean"].array(library="ak")
        z_all = tree[focal_coord].array(library="np")
        x_all = tree[scan_coord].array(library="np")

    dt = time - tleft
    in_window = (dt > 0) & (dt < time_integ)
    area_all = np.abs(ak.to_numpy(ak.sum((volt - bline_mean) * in_window, axis=1)))

    unique_z = np.sort(np.unique(z_all))
    unique_x = np.sort(np.unique(x_all))
    dense_x = np.linspace(np.min(unique_x), np.max(unique_x), 200)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), width_ratios=[2, 1], layout="tight", dpi=200)
    norm = Normalize(vmin=np.min(unique_z), vmax=np.max(unique_z))

    df_pd = pd.DataFrame({focal_coord: z_all, scan_coord: x_all, "area": area_all})
    grouped = df_pd.groupby([focal_coord, scan_coord])["area"].agg(["mean", "std"]).reset_index()

    spot_widths = []
    spot_err = []

    for z in unique_z:
        z_sub = grouped[grouped[focal_coord] == z].sort_values(scan_coord)

        x_vals = z_sub[scan_coord].values
        area_arr = z_sub["mean"].values
        area_std = z_sub["std"].values
        fit_sigma = area_std if np.all(np.isfinite(area_std) & (area_std > 0)) else None

        axes[0].errorbar(x_vals, area_arr, yerr=area_std, fmt="o", color=cmap(norm(z)),
                         alpha=0.5, markersize=0.7)

        low, high = np.min(area_arr), np.max(area_arr)
        mid = 0.5 * (low + high)
        x_cross = x_vals[np.argmin(np.abs(area_arr - mid))]
        n_edge = max(1, len(area_arr) // 10)
        if np.mean(area_arr[:n_edge]) < np.mean(area_arr[-n_edge:]):
            model = erf_model
            p0 = [high - low, x_cross, 0.01, mid]
        else:
            model = erfc_model
            p0 = [high - low, x_cross, 0.01, low]

        popt, pcov = curve_fit(model, x_vals, area_arr, p0=p0, sigma=fit_sigma)
        perr = np.sqrt(np.diag(pcov))
        axes[0].plot(dense_x, model(dense_x, *popt), color=cmap(norm(z)), linestyle="--", alpha=0.5)
        spot_widths.append(abs(popt[2]) * 1000)
        spot_err.append(perr[2] * 1000)

    spot_widths = np.asarray(spot_widths)
    spot_err = np.asarray(spot_err)
    min_arg = np.argmin(spot_widths)
    axes[1].errorbar(unique_z, spot_widths, yerr=spot_err, fmt="o", color="black")
    print("Minimum spot width = {:.2f} µm @ z = {} mm".format(spot_widths[min_arg], unique_z[min_arg]))

    width_sigma = spot_err if np.all(np.isfinite(spot_err) & (spot_err > 0)) else None
    try:
        params, _ = curve_fit(abs_model, unique_z, spot_widths,
                              p0=[1, unique_z[min_arg], spot_widths[min_arg]], sigma=width_sigma)
        print(params)
        axes[1].plot(unique_z, abs_model(unique_z, *params), color="red")
        min_z = params[1]
        print("Fit results: minimum expected width = {:.2f} µm @ z = {:.2f} mm".format(
            abs_model(min_z, *params), min_z))
    except Exception as e:
        print(e)

    laser = "RED" if "red" in filepath.name.lower() else "IR"
    axes[0].set_title(f"Focal scan - {sample} / {filepath.stem} ({laser})")
    axes[0].set_xlabel(f"{scan_coord} [mm]")
    axes[0].set_ylabel("Area [nWb]")
    legend_elements = [
        Line2D([], [], color="black", marker="o", label="Data points"),
        Line2D([], [], color="black", linestyle="--", label="Fit line"),
    ]
    axes[0].legend(handles=legend_elements)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes[0], pad=0.001)
    cbar.set_label(f"{focal_coord} [mm]", fontsize=12)

    axes[1].set_xlabel(f"{focal_coord} [mm]")
    axes[1].set_ylabel("Spot width [µm]")

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_files", nargs="+", metavar="input_file",
                        help="Input ROOT files, located in data/<sample dirs>/<measurement>/")
    parser.add_argument("--time-integ", type=float, default=15.0,
                        help="Integration window after tleft; default: %(default)s")
    parser.add_argument("--scan-coord", default="x", help="Scan coordinate; default: %(default)s")
    parser.add_argument("--focal-coord", default="z", help="Focal coordinate; default: %(default)s")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap; default: %(default)s")
    args = parser.parse_args()

    for ifp in args.input_files:
        input_path = Path(ifp)
        out_path, sample = output_path(input_path)
        print(f"Analyzing file: {input_path}")
        fig = extract_focus(input_path, sample, args.scan_coord, args.focal_coord,
                            args.time_integ, args.cmap)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()