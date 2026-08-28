#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable

from step2point.io.step2point_hdf5 import Step2PointHDF5Reader
from step2point.metrics.spatial import estimate_shower_axis, _normalized_axis, _transverse_basis



SCALAR_KEYS = ("mean_long", "mean_r", "var_long", "var_r", "total_energy", "num_steps")


####### Prepare showers #######
def read_showers(path: str):
    reader = Step2PointHDF5Reader(path)
    return list(reader.iter_showers())


def apply_time_cut_shower(s, tmin=0, tmax=None):
    if s.t is None:
        return s

    mask = np.isfinite(s.t)

    if tmin is not None:
        mask &= (s.t >= tmin)
    if tmax is not None:
        mask &= (s.t <= tmax)

    # skip empty shower
    if np.sum(mask) == 0:
        return None

    return s.copy().__class__(
        shower_id=s.shower_id,
        x=s.x[mask],
        y=s.y[mask],
        z=s.z[mask],
        E=s.E[mask],
        t=s.t[mask],
        cell_id=None if s.cell_id is None else s.cell_id[mask],
        pdg=None if s.pdg is None else s.pdg[mask],
        track_id=None if s.track_id is None else s.track_id[mask],
        primary=s.primary,
        metadata=s.metadata,
    )

def _upper_percentile_limit(values: np.ndarray, percentile: float = 99.0, pad: float = 0.05) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 1.0
    upper = float(np.percentile(values, percentile))
    return max(upper * (1.0 + pad), 1.0)

def aggregate_observables(observables: Iterable[dict[str, object]]) -> dict[str, list[float]]:
    summary = {key: [] for key in SCALAR_KEYS}
    for row in observables:
        for key in SCALAR_KEYS:
            summary[key].append(float(row[key]))
    return summary

def make_linear_shared_bins(all_data_list, key, nbins=20, pad=0.05):
    vals = np.array(
        [r[key] for d in all_data_list for r in d if key in r and np.isfinite(r[key])],
        dtype=float,
    )

    if vals.size == 0:
        return np.linspace(0.0, 1.0, nbins + 1)

    vmin = vals.min()
    vmax = vals.max()

    if vmin == vmax:
        vmax = vmin + 1.0

    dv = vmax - vmin
    return np.linspace(vmin - pad * dv, vmax + pad * dv, nbins + 1)


def longitudinal_radial_phi(
    shower,
    centroid=None,
    axis=None,
    *,
    axis_override=None,
    longitudinal_origin: str = "centroid",
    shift_longitudinal_min: bool = False,
):
    """Return cylindrical shower coordinates around the shower axis.

    If `centroid` and `axis` are not provided, the axis is estimated with
    `estimate_shower_axis`, which means PCA is the default and
    `axis_override` is optional.
    """
    if centroid is None or axis is None:
        centroid, axis = estimate_shower_axis(shower, axis_override=axis_override)
    else:
        axis = _normalized_axis(axis)
    coords = np.stack([shower.x, shower.y, shower.z], axis=1)
    rel = coords - centroid
    long = rel @ axis
    radial_vec = rel - np.outer(long, axis)
    radial = np.linalg.norm(radial_vec, axis=1)
    e1, e2 = _transverse_basis(axis)
    phi = np.arctan2(radial_vec @ e2, radial_vec @ e1)
    if shift_longitudinal_min:
        longitudinal_origin = "min_projection"
    if longitudinal_origin not in {"centroid", "first_deposit", "min_projection"}:
        raise ValueError(f"Unsupported longitudinal_origin: {longitudinal_origin}")
    if longitudinal_origin == "first_deposit" and long.size:
        time = np.asarray(shower.t, dtype=np.float64)
        order = np.lexsort((long, time))
        first_index = order[0]
        long = long - long[first_index]
    if longitudinal_origin == "min_projection" and long.size:
        long = long - np.min(long)
    return long, radial, phi


####### Calculate obserables #######
def _weighted_moment(values: np.ndarray, weights: np.ndarray, order: int) -> float:
    if values.size == 0 or np.sum(weights) <= 0.0:
        return float("nan")
    return float(np.average(values**order, weights=weights))


def _safe_log10(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log10(np.clip(np.asarray(values, dtype=np.float64), eps, None))
    
def compute_shower_observables(shower, *, axis_override=None) -> dict[str, object]:
    """Compute summary observables for one shower.

    The default axis comes from PCA-based shower-axis estimation. Pass
    `axis_override` only when a manual physics-motivated direction should
    replace that default.
    """
    centroid, axis = estimate_shower_axis(shower, axis_override=axis_override)
    long, radial, phi = longitudinal_radial_phi(
        shower,
        centroid=centroid,
        axis=axis,
        longitudinal_origin="first_deposit",
    )
    weights = np.asarray(shower.E, dtype=np.float64)
    mean_long = _weighted_moment(long, weights, 1)
    mean_r = _weighted_moment(radial, weights, 1)
    var_long = float(np.average((long - mean_long) ** 2, weights=weights)) if long.size else float("nan")
    var_r = float(np.average((radial - mean_r) ** 2, weights=weights)) if radial.size else float("nan")
    if shower.t is not None:
        time = np.asarray(shower.t, dtype=np.float64)
    else:
        time = np.array([], dtype=np.float64)
    return {
        "long_values": long,
        "radial_values": radial,
        "phi_values": phi,
        "log_energy_values": _safe_log10(weights),
        "weights": weights,
        "long_profile": np.histogram(long, bins=30, weights=weights),
        "r_profile": np.histogram(radial, bins=50, weights=weights),
        "phi_profile": np.histogram(phi, bins=np.linspace(-np.pi, np.pi, 51), weights=weights),
        "log_energy": np.histogram(_safe_log10(weights), bins=50),
        "mean_long": mean_long,
        "mean_r": mean_r,
        "var_long": var_long,
        "var_r": var_r,
        "total_energy": float(np.sum(weights)),
        "num_steps": int(len(weights)),
        "axis": axis,
        "centroid": centroid,
        "time": time,
    }


####### Plotting #######
def generate_observables_matrix(
    showers_by_dataset, 
    labels,
    outpath: str | Path, 
    tmax,
    axis_override=None,
    colors = ["#1f77b4", "#ff7f0e"], 
    detector = "ODD",
    energy = "100GeV",
    selected_index: int | None = None
):
    # check length
    assert(len(colors) == len(showers_by_dataset))

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    all_data_list = []
    average_data_list = []

    for showers in showers_by_dataset:
        processed = []

        for s in showers:
            s_cut = apply_time_cut_shower(s, tmax=tmax)
            if s_cut is None:
                continue
            if np.sum(s_cut.E) <= 0:
                continue

            obs = compute_shower_observables(
                s_cut,
                axis_override=axis_override
            )

            processed.append(obs)

        all_data_list.append(processed)
        average_data_list.append(aggregate_observables(processed))
    
    event_count_list = []
    for all_data in all_data_list:
        event_count_list.append(len(all_data))

    shared_bins = {
        "mean_long": make_linear_shared_bins(all_data_list, "mean_long", nbins=20),
        "mean_r": make_linear_shared_bins(all_data_list, "mean_r", nbins=20),
        "total_energy": make_linear_shared_bins(all_data_list, "total_energy", nbins=20),
        "var_long": make_linear_shared_bins(all_data_list, "var_long", nbins=20),
        "var_r": make_linear_shared_bins(all_data_list, "var_r", nbins=20),
        "num_steps": make_linear_shared_bins(all_data_list, "num_steps", nbins=20),
    }

    all_long_values = np.concatenate(
        [np.asarray(row["long_values"], dtype=np.float64) for row in all_data if np.size(row["long_values"]) > 0]
    )
    long_bins = np.linspace(0.0, _upper_percentile_limit(all_long_values), 16)
    radial_bins = np.linspace(
        0.0,
        max(float(np.max(row["radial_values"])) for row in all_data if np.size(row["radial_values"]) > 0),
        51,
    )
    log_energy_min = min(float(np.min(row["log_energy_values"])) for row in all_data if np.size(row["log_energy_values"]) > 0)
    log_energy_max = max(float(np.max(row["log_energy_values"])) for row in all_data if np.size(row["log_energy_values"]) > 0)
    log_energy_bins = np.linspace(log_energy_min, log_energy_max, 51)

    def histogram_values(row, key):
        if key == "long_profile":
            return np.histogram(row["long_values"], bins=long_bins, weights=row["weights"])[0]
        if key == "r_profile":
            return np.histogram(row["radial_values"], bins=radial_bins, weights=row["weights"])[0]
        if key == "log_energy":
            return np.histogram(row["log_energy_values"], bins=log_energy_bins)[0]
        raise ValueError(f"Unsupported profile key: {key}")

    def plot_avg(ax, all_data_list, key, xlabel, logy: bool = False):
        for i, dataset in enumerate(all_data_list):
            if key == "long_profile":
                bins = long_bins
            elif key == "r_profile":
                bins = radial_bins
            elif key == "log_energy":
                bins = log_energy_bins
            else:
                raise ValueError(f"Unsupported profile key: {key}")
            values = np.array([histogram_values(row, key) for row in dataset], dtype=np.float64)
            centers = 0.5 * (bins[:-1] + bins[1:])
            mean_values = np.mean(values, axis=0)
            ax.plot(centers, mean_values, color=colors[i], linewidth=2.2, alpha=0.22, linestyle="--", zorder=3)
            ax.scatter(
                centers,
                mean_values,
                color=colors[i],
                s=32,
                marker="o",
                linewidths=0.0,
                label=labels[i],
                zorder=4,
            )
            ax.set_xlabel(xlabel)
            ax.legend()
            if logy:
                ax.set_yscale("log")
            if key == "long_profile":
                ax.set_xlim(0.0, long_bins[-1])

    plot_avg(axes[0, 0], all_data_list, "long_profile", "longitudinal (from first deposit) [mm]")
    plot_avg(axes[0, 1], all_data_list, "r_profile", "radial [mm]", logy=True)
    plot_avg(axes[0, 2], all_data_list, "log_energy", "log10(energy [GeV])", logy=True)

    for i, (key, label) in enumerate(
        zip(
            ["mean_long", "mean_r", "total_energy"],
            ["first longitudinal moment [mm]", "first radial moment [mm]", "total deposited energy [GeV]"],
            strict=True,
        )
    ):
        for j, dataset in enumerate(all_data_list):
            counts, bins, _ = axes[1, i].hist(average_data_list[j][key], color=colors[j], bins=shared_bins[key], alpha=0.45, label=labels[j])
            axes[1, i].set_xlabel(label)
            axes[1, i].legend()

    for i, (key, label) in enumerate(
        zip(
            ["var_long", "var_r", "num_steps"],
            ["second longitudinal moment [mm²]", "second radial moment [mm²]", "number of steps"],
            strict=True,
        )
    ):
        for j, dataset in enumerate(all_data_list):
            counts, bins, _ = axes[2, i].hist(average_data_list[j][key], color=colors[j], bins=shared_bins[key], alpha=0.45, label=labels[j])
            axes[2, i].set_xlabel(label)
            axes[2, i].legend()

    fig.suptitle(
        f"Shower Observables Comparison for {detector} at {energy} with tmax={tmax} ns",
        fontsize=16
    )
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--outdir", default="shower_cut")
    parser.add_argument("--colors", nargs="+", default = ["#1f77b4", "#ff7f0e"])
    parser.add_argument("--axis", nargs=3, type=float, default=[0, 1, 0])
    parser.add_argument("--detector", default="ODD")
    parser.add_argument("--energy", default="100GeV")
    parser.add_argument("--tmax", type=float, default=400.0)
    parser.add_argument("--label", required=True)

    args = parser.parse_args()

    showers_by_dataset = []

    for input in args.input:
        showers_by_dataset.append(read_showers(input))

    generate_observables_matrix(
        showers_by_dataset =showers_by_dataset,
        labels=args.labels,
        outpath=Path(args.outdir) / f"{args.detector}_{args.energy}_dataset_observables_{args.label}_{args.tmax}ns.png",
        tmax=args.tmax,
        axis_override=args.axis,
        colors=args.colors,
        detector=args.detector,
        energy=args.energy
    )


if __name__ == "__main__":
    main()