"""Toy model, verbatim, applied to hZZ signal-strength (mu) regression.

The MODEL is the toy's, unchanged: TemplateDiscriminator([128, 256, 128]),
100 epochs, lr 3e-3, weight decay 1e-4, batch 2048, 80/20 split, 100k events
per template class, trained by the unmodified `train_discriminator` and
profiled by the unmodified `profile_mass_estimate` (mean per-event log-probs ->
-2 dlogL -> quadratic vertex -> monotonic-anchor calibration).  No soft
targets, no temperature scaling, no architecture or schedule changes.

Only the TEMPLATE CONSTRUCTION differs, and it is forced: the toy simulates a
separate sample per mass hypothesis, whereas hZZ provides ONE event pool that
must be reweighted into mu-hypotheses via
    w(mu) = mu * msq_sig + sqrt(mu) * msq_int + msq_bkg,
divided by the generation density msq_sbi.  Templates are built by HARD-LABEL
importance resampling, the closest available analogue of separate simulations.

Two template grids are compared, since the grid is part of the template
construction and therefore exempt from the verbatim rule:
  A {0.0, 1.0, 3.0}    uneven gaps (1, 2); a template sits on the mu = 0
                       boundary where sqrt(mu) has infinite derivative, and
                       the mu = 1 template coincides with the sample's own
                       generation point (its resampling weights are identically 1)
  B {0.4, 1.45, 2.5}   endpoints + midpoint of the pseudo-data range, mirroring
                       the toy's 400/500/600 over [400, 600]; even gaps, and
                       avoids both the boundary and the generation point

The pool is cut three disjoint ways: TRAIN (members resample from here), CALIB
(calibration anchors), EVAL (reported sweep).  No member is profiled or
calibrated on events it trained on.  A shared CALIB set is used because an
ensemble has no train-time validation split of its own; being disjoint from
every member's training data, it is the conservative choice.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sbi_reasoning.models import (  # noqa: E402
    Standardizer,
    TemplateDiscriminator,
    predict_log_probabilities,
    train_discriminator,
)

RAW = [f"l{i}_{v}" for i in (1, 2, 3, 4) for v in ("pt", "eta", "phi", "energy")]
FEATURES = RAW + ["4l_mass"]
MSQ = ["msq_sig_sm", "msq_int_sm", "msq_sbi_sm", "msq_bkg_sm"]
# One grid per analysis region.  0.4 and 2.5 are too far apart for a single
# three-node grid to bracket both without extrapolation, so each region gets a
# grid that brackets its truth in interpolation, sits clear of the mu = 0
# boundary (where d sqrt(mu)/d mu diverges), and does not coincide with the
# truth value, so no estimate can be anchored onto a template by np.clip.
GRIDS = {
    "L": {"templates": [0.3, 1.3, 2.3], "sweep": [0.4, 0.8, 1.3, 1.8],
          "observed": "obs_mu0p4", "truth": 0.4},
    "H": {"templates": [1.0, 2.0, 3.0], "sweep": [1.5, 2.0, 2.5, 2.8],
          "observed": "obs_mu2p5", "truth": 2.5},
}

# config/toy.yaml, verbatim
TOY = {
    "profiler_learning_rate": 0.003,
    "weight_decay": 0.0001,
    "profiler_epochs": 100,
    "batch_size": 2048,
    "mixed_precision": "none",
}
HIDDEN = [128, 256, 128]
VALIDATION_FRACTION = 0.2
N_PER_CLASS = 100_000          # toy: template_events_per_mass
N_MEMBERS = 16
N_EVAL = 200_000
N_CALIB = 120_000
OBS_SHARDS = 8
EVAL_BATCH = 1024              # toy: evaluation.batch_size
CACHE = Path("outputs/toy_verbatim_mu")
DEVICE = torch.device("cpu")  # overridden by --device


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


def invariant_mass(raw: np.ndarray) -> np.ndarray:
    """Observed shards ship no 4l_mass column, so it is rebuilt from the four-vectors."""
    e = px = py = pz = 0.0
    for i in range(4):
        pt, eta, phi, energy = (raw[:, 4 * i + k] for k in range(4))
        px = px + pt * np.cos(phi)
        py = py + pt * np.sin(phi)
        pz = pz + pt * np.sinh(eta)
        e = e + energy
    return np.sqrt(np.clip(e**2 - px**2 - py**2 - pz**2, 0.0, None))


def load_observed(tag: str) -> tuple[np.ndarray, np.ndarray]:
    cached = CACHE / f"obs_{tag}.npy"
    if cached.exists():
        data = np.load(cached)
        return data[:, :-1], data[:, -1]
    rows = []
    for shard in sorted(Path(tag).glob("observed_*.csv"))[:OBS_SHARDS]:
        with open(shard) as stream:
            reader = csv.reader(stream)
            header = next(reader)
            index = [header.index(c) for c in RAW] + [header.index("n")]
            for row in reader:
                if row:
                    rows.append([row[i] for i in index])
    array = np.asarray(rows, dtype=np.float64)
    data = np.column_stack([array[:, :16], invariant_mass(array[:, :16]), array[:, 16]])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cached, data)
    return data[:, :-1], data[:, -1]


def density_ratio(msq: np.ndarray, mu: float) -> np.ndarray:
    sig, interference, sbi, bkg = (msq[:, k] for k in range(4))
    weight = mu * sig + np.sqrt(mu) * interference + bkg
    return np.clip(weight, 0.0, None) / sbi


def resample(index: np.ndarray, msq: np.ndarray, mu: float, count: int,
             rng: np.random.Generator) -> np.ndarray:
    weight = density_ratio(msq[index], mu)
    return rng.choice(index, size=count, replace=True, p=weight / weight.sum())


def build_templates(index: np.ndarray, msq: np.ndarray, grid: list[float],
                    count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    draws, labels = [], []
    for label, mu in enumerate(grid):
        draws.append(resample(index, msq, mu, count, rng))
        labels.append(np.full(count, label))
    return np.concatenate(draws), np.concatenate(labels)


def split_by_event_id(draw_index: np.ndarray, rng: np.random.Generator,
                      pool_size: int) -> np.ndarray:
    """80/20 split keeping a pool event wholly in train or wholly in validation.

    The toy's templates are separate simulations, so its permutation split can
    never place the same event on both sides.  Resampling one shared pool can,
    so the split is made on pool event id to preserve that property.
    """
    unique = np.unique(draw_index)
    shuffled = rng.permutation(unique)
    cut = int((1.0 - VALIDATION_FRACTION) * len(shuffled))
    is_train_id = np.zeros(pool_size, dtype=bool)
    is_train_id[shuffled[:cut]] = True
    return is_train_id[draw_index]


def train_member(member: int, grid_name: str, grid: list[float], features: np.ndarray,
                 msq: np.ndarray, train_pool: np.ndarray,
                 eval_sets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1000 + member)
    torch.manual_seed(2000 + member)

    draw_index, labels = build_templates(train_pool, msq, grid, N_PER_CLASS, rng)
    is_train = split_by_event_id(draw_index, rng, len(features))

    train_features = features[draw_index[is_train]].astype(np.float32)
    train_labels = labels[is_train].astype(np.int64)
    validation_features = features[draw_index[~is_train]].astype(np.float32)
    validation_labels = labels[~is_train].astype(np.int64)

    scaler = Standardizer.fit(train_features)
    model = TemplateDiscriminator(len(FEATURES), len(grid), HIDDEN)
    history = train_discriminator(
        model, scaler.transform(train_features), train_labels,
        scaler.transform(validation_features), validation_labels,
        TOY, DEVICE, rng, f"{grid_name}-m{member:02d}",
    )

    out = {
        "loss": np.asarray(history["loss"]),
        "accuracy": np.asarray(history["validation_accuracy"]),
        "validation_labels": validation_labels,
    }
    out["validation"] = predict_log_probabilities(
        model, scaler.transform(validation_features), DEVICE, EVAL_BATCH
    ).astype(np.float32)
    for name, array in eval_sets.items():
        out[name] = predict_log_probabilities(
            model, scaler.transform(array.astype(np.float32)), DEVICE, EVAL_BATCH
        ).astype(np.float32)
    return out


def build_eval_sets(features: np.ndarray, msq: np.ndarray, grid: list[float],
                    sweep: list[float], observed_tag: str,
                    calib_pool: np.ndarray, eval_pool: np.ndarray
                    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    eval_sets: dict[str, np.ndarray] = {}
    meta: dict[str, np.ndarray] = {}

    calib_index, calib_labels = build_templates(calib_pool, msq, grid, N_CALIB, rng)
    eval_sets["calib"] = features[calib_index]
    meta["calib_labels"] = calib_labels

    for mu in sweep:
        for seed in range(3):
            drawn = resample(eval_pool, msq, mu, N_EVAL, np.random.default_rng(100 + seed))
            eval_sets[f"sweep_{mu}_s{seed}"] = features[drawn]
    observed_features, observed_weight = load_observed(observed_tag)
    eval_sets[observed_tag] = observed_features
    meta[f"{observed_tag}_weight"] = observed_weight
    return eval_sets, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=sorted(GRIDS), default="A")
    parser.add_argument("--members-from", type=int, default=0)
    parser.add_argument("--members-to", type=int, default=N_MEMBERS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    global DEVICE
    DEVICE = torch.device(args.device)
    print(f"[device] {DEVICE}", flush=True)

    spec = GRIDS[args.grid]
    grid = spec["templates"]
    out_dir = CACHE / f"grid{args.grid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool()
    features = pool[:, :len(FEATURES)]
    msq = pool[:, len(FEATURES):]
    print(f"[pool] {len(pool)} events | grid {args.grid} = {grid}", flush=True)

    splitter = np.random.default_rng(240513)
    order = splitter.permutation(len(pool))
    n_train = int(0.60 * len(pool))
    n_calib = int(0.15 * len(pool))
    train_pool = order[:n_train]
    calib_pool = order[n_train:n_train + n_calib]
    eval_pool = order[n_train + n_calib:]

    eval_sets, meta = build_eval_sets(features, msq, grid, spec["sweep"],
                                      spec["observed"], calib_pool, eval_pool)
    meta_path = out_dir / "eval_meta.npz"
    if not meta_path.exists():
        np.savez_compressed(meta_path, **meta)

    for member in range(args.members_from, args.members_to):
        target = out_dir / f"member_{member:02d}.npz"
        if target.exists():
            print(f"[member {member}] cached, skipping", flush=True)
            continue
        result = train_member(member, args.grid, grid, features, msq, train_pool, eval_sets)
        np.savez_compressed(target, **result)
        print(f"[member {member}] done acc={result['accuracy'][-1]:.4f} "
              f"loss={result['loss'][-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
