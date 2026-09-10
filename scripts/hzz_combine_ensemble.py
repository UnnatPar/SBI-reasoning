"""Combine trained members into ensembles of size K and profile mu.

Members are combined by averaging per-event log-probabilities (logmean), the
correct pooling if the classifier is read as a log-likelihood-ratio estimator.
Size-K ensembles are formed by drawing SUBSETS of the trained members, several
draws per K, so every point carries a spread rather than a single number.

Profiling is the toy's, unmodified: mean per-event log-probs -> -2 dlogL ->
quadratic vertex -> calibration against monotonic anchors.  Anchors come from
the shared CALIB set (disjoint from every member's training data).  When the
anchors are not monotonic the toy refuses to calibrate, and the raw estimate is
reported with calibrated left as nan -- that refusal is itself a result.

Observed data is weighted by its per-event column n, because obs_mu0p4 and
obs_mu2p5 share their events and the mu-dependence lives entirely in n.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sbi_reasoning.pipeline import profile_mass_estimate  # noqa: E402

GRIDS = {
    "L": {"templates": [0.3, 1.3, 2.3], "sweep": [0.4, 0.8, 1.3, 1.8],
          "observed": "obs_mu0p4", "truth": 0.4},
    "H": {"templates": [1.0, 2.0, 3.0], "sweep": [1.5, 2.0, 2.5, 2.8],
          "observed": "obs_mu2p5", "truth": 2.5},
}
# perfect analytic classifier, zero ensemble noise: the floor this scan is measured against
FLOOR = {"L": {0.4: 0.341}, "H": {2.5: 2.550}}
K_VALUES = [1, 2, 4, 8]
N_DRAWS = 8
SEEDS = [0, 1, 2]
CACHE = Path("outputs/toy_verbatim_mu")


def weighted_profile(log_probabilities: np.ndarray, grid: list[float],
                     weight: np.ndarray, calibration) -> float:
    """profile_mass_estimate with an n-weighted mean, for the observed shards."""
    grid_array = np.asarray(grid, dtype=np.float64)
    profile = (log_probabilities * weight[:, None]).sum(axis=0) / weight.sum()
    n2dll = -2.0 * (profile - profile.max())
    quadratic, linear, _ = np.polyfit(grid_array, n2dll, 2)
    if quadratic <= 0.0:
        estimate = float(grid_array[np.argmin(n2dll)])
    else:
        estimate = float(np.clip(-linear / (2.0 * quadratic),
                                 grid_array.min(), grid_array.max()))
    if calibration is not None:
        estimate = float(np.interp(estimate,
                                   calibration["raw_mass_gev"],
                                   calibration["calibrated_mass_gev"]))
    return estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=sorted(GRIDS), default="A")
    args = parser.parse_args()
    spec = GRIDS[args.grid]
    grid = spec["templates"]
    sweep = spec["sweep"]
    obs_tag = spec["observed"]
    obs_truth = spec["truth"]
    directory = CACHE / f"grid{args.grid}"

    members = sorted(directory.glob("member_*.npz"))
    loaded = [np.load(path) for path in members]
    meta = np.load(directory / "eval_meta.npz")
    calib_labels = meta["calib_labels"]
    print(f"[grid {args.grid}] {len(loaded)} members, templates {grid}", flush=True)

    print("\n=== learning summary ===", flush=True)
    losses = [float(d["loss"][-1]) for d in loaded]
    accuracies = [float(d["accuracy"][-1]) for d in loaded]
    print(f"final train loss  {np.mean(losses):.4f} +/- {np.std(losses):.4f}   "
          f"(ln(3) = {np.log(3):.4f})", flush=True)
    print(f"validation accuracy {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}   "
          f"(chance = {1/3:.4f})", flush=True)

    rows = []
    rng = np.random.default_rng(0)
    for k in K_VALUES:
        if k == 1:
            subsets = [(i,) for i in range(min(len(loaded), N_DRAWS))]
        else:
            all_subsets = list(itertools.combinations(range(len(loaded)), k))
            picked = rng.choice(len(all_subsets), size=min(N_DRAWS, len(all_subsets)),
                                replace=False)
            subsets = [all_subsets[i] for i in picked]

        anchors_monotonic = []
        per_truth: dict[float, list[float]] = {mu: [] for mu in sweep}
        observed: dict[str, list[float]] = {obs_tag: []}

        for subset in subsets:
            calib = np.mean([loaded[i]["calib"] for i in subset], axis=0)
            anchors = [profile_mass_estimate(calib[calib_labels == c], grid)[0]
                       for c in range(len(grid))]
            monotonic = bool(np.all(np.diff(anchors) > 0.0))
            anchors_monotonic.append(monotonic)
            calibration = ({"raw_mass_gev": anchors, "calibrated_mass_gev": grid}
                           if monotonic else None)

            for mu in sweep:
                for seed in SEEDS:
                    key = f"sweep_{mu}_s{seed}"
                    stacked = np.mean([loaded[i][key] for i in subset], axis=0)
                    per_truth[mu].append(profile_mass_estimate(stacked, grid, calibration)[0])
            for tag in observed:
                stacked = np.mean([loaded[i][tag] for i in subset], axis=0)
                observed[tag].append(
                    weighted_profile(stacked, grid, meta[f"{tag}_weight"], calibration))

        monotonic_rate = float(np.mean(anchors_monotonic))
        for mu in sweep:
            values = np.asarray(per_truth[mu])
            rows.append({
                "grid": args.grid, "dataset": "simulated", "truth_mu": mu, "K": k,
                "recovered_mu": round(float(values.mean()), 4),
                "bias": round(float(values.mean() - mu), 4),
                "spread": round(float(values.std()), 4),
                "monotonic_anchor_rate": round(monotonic_rate, 3),
            })
        for tag, truth in ((obs_tag, obs_truth),):
            values = np.asarray(observed[tag])
            rows.append({
                "grid": args.grid, "dataset": "observed", "truth_mu": truth, "K": k,
                "recovered_mu": round(float(values.mean()), 4),
                "bias": round(float(values.mean() - truth), 4),
                "spread": round(float(values.std()), 4),
                "monotonic_anchor_rate": round(monotonic_rate, 3),
            })
        print(f"  K={k}: monotonic anchors in {monotonic_rate:.0%} of draws", flush=True)

    out = CACHE / f"ensemble_scan_grid{args.grid}.csv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[done] {out}", flush=True)

    print(f"\n=== grid {args.grid}: bias and spread vs ensemble size ===", flush=True)
    header = "truth | " + " | ".join(f"K={k:<2}  bias    spread" for k in K_VALUES)
    print(header, flush=True)
    for dataset in ("simulated", "observed"):
        for mu in (sweep if dataset == "simulated" else [obs_truth]):
            cells = []
            for k in K_VALUES:
                match = [r for r in rows if r["dataset"] == dataset
                         and r["truth_mu"] == mu and r["K"] == k]
                if match:
                    cells.append(f"{match[0]['bias']:+7.3f} {match[0]['spread']:7.3f}")
            print(f"{dataset[:3]} {mu:<4} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
