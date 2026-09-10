"""Perfect-analytic-classifier floor for the hZZ mu profiler, across template grids.

No neural network here.  The classifier is replaced by the exact analytic
posterior built from the matrix-element columns, so the only thing under test is
the ESTIMATOR (mean log-probs -> quadratic vertex -> monotonic-anchor
calibration, all unmodified toy code).  Any bias that survives here is method
bias, not learning failure.

Grids compared:
  A  {0.0, 1.0, 3.0}    status quo; uneven gaps (1, 2) and a template sitting
                        exactly on the mu = 0 boundary, where the sqrt(mu)
                        interference term has infinite derivative
  B  {0.4, 1.45, 2.5}   endpoints + midpoint of the pseudo-data range, mirroring
                        the toy's 400/500/600 over [400, 600]; even gaps
  C  {0.5, 1.5, 2.5}    even, and offset so no template coincides with a tested
                        truth value
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sbi_reasoning.pipeline import profile_mass_estimate  # noqa: E402

RAW = [f"l{i}_{v}" for i in (1, 2, 3, 4) for v in ("pt", "eta", "phi", "energy")]
FEATURES = RAW + ["4l_mass"]
MSQ = ["msq_sig_sm", "msq_int_sm", "msq_sbi_sm", "msq_bkg_sm"]
MU_SWEEP = [0.4, 1.0, 1.5, 2.5, 3.0]
GRIDS = {
    "A_{0,1,3}": [0.0, 1.0, 3.0],
    "B_{0.4,1.45,2.5}": [0.4, 1.45, 2.5],
    "C_{0.5,1.5,2.5}": [0.5, 1.5, 2.5],
}
N_EVAL = 200_000
N_CALIB = 120_000
CACHE = Path("outputs/toy_verbatim_mu")


def load_pool() -> np.ndarray:
    cached = CACHE / "pool.npy"
    if cached.exists():
        return np.load(cached)
    columns = FEATURES + MSQ
    rows = []
    with open("sbi_reduced.csv") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        index = [header.index(c) for c in columns]
        for row in reader:
            rows.append([row[i] for i in index])
    pool = np.asarray(rows, dtype=np.float64)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cached, pool)
    return pool


def density_ratio(msq: np.ndarray, mu: float) -> np.ndarray:
    """p(x | mu) / q(x) up to a constant; q is the generation density msq_sbi."""
    sig, interference, sbi, bkg = (msq[:, k] for k in range(4))
    weight = mu * sig + np.sqrt(mu) * interference + bkg
    return np.clip(weight, 0.0, None) / sbi


def resample(index: np.ndarray, msq: np.ndarray, mu: float, count: int,
             rng: np.random.Generator) -> np.ndarray:
    weight = density_ratio(msq[index], mu)
    return rng.choice(index, size=count, replace=True, p=weight / weight.sum())


def analytic_log_posterior(msq: np.ndarray, index: np.ndarray,
                           grid: list[float], norms: dict[float, float]) -> np.ndarray:
    stack = []
    for mu in grid:
        ratio = density_ratio(msq[index], mu) / norms[mu]
        stack.append(np.log(np.clip(ratio, 1e-300, None)))
    logits = np.stack(stack, axis=1)
    maximum = logits.max(axis=1, keepdims=True)
    return logits - (maximum + np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True)))


def main() -> None:
    pool = load_pool()
    msq = pool[:, len(FEATURES):]
    print(f"[pool] {len(pool)} events", flush=True)

    splitter = np.random.default_rng(240513)
    order = splitter.permutation(len(pool))
    n_train = int(0.60 * len(pool))
    n_calib = int(0.15 * len(pool))
    calib_pool = order[n_train:n_train + n_calib]
    eval_pool = order[n_train + n_calib:]

    rows = []
    for grid_name, grid in GRIDS.items():
        norms = {mu: float(np.mean(density_ratio(msq, mu))) for mu in grid}
        rng = np.random.default_rng(7)

        # calibration anchors: one template class at a time, exactly as the toy does
        anchors = []
        for mu in grid:
            drawn = resample(calib_pool, msq, mu, N_CALIB, rng)
            log_posterior = analytic_log_posterior(msq, drawn, grid, norms)
            anchors.append(profile_mass_estimate(log_posterior, grid)[0])
        monotonic = bool(np.all(np.diff(anchors) > 0.0))
        calibration = (
            {"raw_mass_gev": [float(a) for a in anchors], "calibrated_mass_gev": grid}
            if monotonic else None
        )
        print(f"\n[{grid_name}] anchors={np.round(anchors, 4).tolist()} "
              f"monotonic={monotonic}", flush=True)

        for truth in MU_SWEEP:
            estimates = []
            for seed in range(3):
                seed_rng = np.random.default_rng(100 + seed)
                drawn = resample(eval_pool, msq, truth, N_EVAL, seed_rng)
                log_posterior = analytic_log_posterior(msq, drawn, grid, norms)
                estimates.append(profile_mass_estimate(log_posterior, grid, calibration)[0])
            mean = float(np.mean(estimates))
            rows.append({
                "grid": grid_name, "truth_mu": truth,
                "recovered_mu": round(mean, 4),
                "bias": round(mean - truth, 4),
                "seed_spread": round(float(np.std(estimates)), 4),
                "monotonic_anchors": monotonic,
            })
            print(f"  truth={truth:<5} recovered={mean:7.4f} bias={mean-truth:+7.4f} "
                  f"spread={np.std(estimates):.4f}", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "perfect_classifier_grid_floor.csv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[done] {out}", flush=True)

    print("\n=== RMSE by grid (bias over the sweep) ===", flush=True)
    for grid_name in GRIDS:
        biases = [r["bias"] for r in rows if r["grid"] == grid_name]
        print(f"  {grid_name:<20} RMSE={np.sqrt(np.mean(np.square(biases))):.4f}", flush=True)


if __name__ == "__main__":
    main()
