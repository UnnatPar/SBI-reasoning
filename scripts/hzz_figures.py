"""Figures for the hZZ mu study, from the CSVs in results/ (FIGURE_QA.md style).

floor.csv, mlp_grid*.csv      produced by hzz_mu.py (floor; mlp on the 8 cached members)
heads_grid*.csv               token heads, CPU run; hzz_mu.py heads reproduces it bit-exactly
heads_scan_gpu.csv            head-capacity scan incl. EveNet's own 1.19M head (GPU; nats/accuracy)
mlp_ensemble16.csv, mlp_losses16.csv   the 16-member K scan (members 8-15 were not kept)
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path("results")
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
MODEL = {"MLP": OI[0], "linear": OI[1], "tiny": OI[2], "small": OI[3], "full": OI[4]}
GRID_C = {"L": OI[0], "H": OI[1]}
SPEC = {"L": ([0.3, 1.3, 2.3], 0.4), "H": ([1.0, 2.0, 3.0], 2.5)}
AVAILABLE = 0.018471  # Jeffreys divergence between the extreme templates, nats/event
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": False, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "font.family": ["Arial", "DejaVu Sans", "Liberation Sans"]})


def rows(name):
    with (R / name).open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(R / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def is_edge(r):
    v = float(r["clipped_mu"] if "clipped_mu" in r else r["recovered_mu"])
    lo, hi = (float(r["grid_min"]), float(r["grid_max"])) if "grid_min" in r else \
        (min(SPEC[r["grid"]][0]), max(SPEC[r["grid"]][0]))
    return abs(v - lo) < 1e-9 or abs(v - hi) < 1e-9


def mean_sd(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


# 1 -- template-grid geometry with the perfect classifier
def fig_geometry():
    groups = defaultdict(list)
    for r in rows("floor.csv"):
        groups[(r["grid_label"], float(r["truth_mu"]))].append(r)
    grids = ["{0,1,3}", "{0,1,2}", "{0.3,1.3,2.3}", "{0.5,1.5,2.5}", "{1,2,3}"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, truth in zip(axes, (0.4, 2.5)):
        for i, g in enumerate(grids):
            rs = groups[(g, truth)]
            v = [float(r["clipped_mu"]) for r in rs]
            m, sd = mean_sd(v)
            lo, hi = float(rs[0]["grid_min"]), float(rs[0]["grid_max"])
            inside = lo < truth < hi
            color = OI[3] if lo == 0.0 else OI[0]
            ax.errorbar(i, m - truth, yerr=sd / math.sqrt(len(v)), fmt="o" if inside else "s",
                        mfc=color if inside else "white", color=color, ms=5, capsize=2.5, lw=1.2)
            clipped = sum(is_edge(r) for r in rs)
            note = ("extrapolation\n" if not inside else "") + (f"{clipped}/8 at grid edge" if clipped else "")
            if note:
                ax.annotate(note, (i, m - truth), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=5.5, color="0.35")
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xticks(range(len(grids)), [g.replace(",", ", ") for g in grids], rotation=25, ha="right")
        ax.set_xlabel("template grid (mu values)")
        ax.set_title(f"truth mu = {truth}")
        ax.set_xlim(-0.6, len(grids) - 0.4)
    axes[0].set_ylabel("bias in recovered mu (+/- SE, 8 seeds)")
    handles = [plt.Line2D([], [], ls="", marker="o", color=OI[3]), plt.Line2D([], [], ls="", marker="o", color=OI[0]),
               plt.Line2D([], [], ls="", marker="s", mfc="white", color="0.4")]
    fig.legend(handles, ["grid has a node at mu = 0", "no node at mu = 0", "truth outside grid (extrapolation)"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Perfect analytic classifier: a node on the mu = 0 boundary dominates the bias", y=1.02)
    save(fig, "fig1_grid_geometry")


# 2 -- floor response on the two analysis grids
def fig_floor():
    groups = defaultdict(list)
    for r in rows("floor.csv"):
        groups[(r["grid_label"], float(r["truth_mu"]))].append(float(r["clipped_mu"]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for g, label in (("L", "{0.3,1.3,2.3}"), ("H", "{1,2,3}")):
        ts = sorted(t for (gl, t) in groups if gl == label and min(SPEC[g][0]) < t < max(SPEC[g][0]))
        ms = [mean_sd(groups[(label, t)]) for t in ts]
        a.errorbar(ts, [m for m, _ in ms], yerr=[s for _, s in ms], fmt="o", color=GRID_C[g],
                   ms=4, capsize=2, label=f"grid {g} {label}")
        b.errorbar(ts, [m - t for (m, _), t in zip(ms, ts)], yerr=[s / math.sqrt(8) for _, s in ms],
                   fmt="o-", color=GRID_C[g], ms=4, capsize=2, lw=1)
    a.plot([0.2, 3.0], [0.2, 3.0], "--", color="0.5", lw=0.8, label="recovered = truth")
    a.set_xlabel("truth mu")
    a.set_ylabel("recovered mu (+/- sd, 8 seeds)")
    a.legend(frameon=False)
    b.axhline(0, color="0.4", lw=0.8)
    b.set_xlabel("truth mu")
    b.set_ylabel("bias (+/- SE)")
    b.set_title("precise (sd 0.03-0.06) but biased up to 0.17: a parabola\nfitted to a non-quadratic likelihood")
    fig.suptitle("Estimator floor: exact posterior, toy profiling unchanged", y=1.04)
    save(fig, "fig2_floor_response")


# 3 -- memorisation: information captured on train vs what physically exists
def fig_memorisation():
    nats = defaultdict(list)
    for r in rows("mlp_losses16.csv"):
        nats[(r["grid"], "MLP")].append(math.log(3) - float(r["loss"]))
    seen = set()
    for r in rows("heads_scan_gpu.csv"):
        key = (r["grid"], r["head"], r["seed"])
        if key not in seen:
            seen.add(key)
            nats[(r["grid"], r["head"])].append(float(r["nats"]))
    models = ["MLP", "linear", "tiny", "small", "full"]
    labels = {"MLP": "toy MLP\n68,611", "linear": "EveNet linear\n771", "tiny": "EveNet tiny\n16,643",
              "small": "EveNet small\n66,563", "full": "EveNet full\n1,186,563"}
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    width = 0.38
    for j, g in enumerate("LH"):
        for i, mdl in enumerate(models):
            m, sd = mean_sd(nats[(g, mdl)])
            ax.bar(i + (j - 0.5) * width, m, width * 0.92, yerr=sd, capsize=2, color=MODEL[mdl],
                   alpha=1.0 if g == "L" else 0.55, edgecolor=MODEL[mdl], lw=0.6)
            ax.annotate(f"{m / AVAILABLE:.1f}x", (i + (j - 0.5) * width, max(m, 0) + sd),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=5.5)
    ax.axhline(AVAILABLE, color="0.2", ls="--", lw=0.9)
    ax.text(len(models) - 0.5, AVAILABLE, " physically available\n 0.0185 nats/event", va="bottom",
            ha="right", fontsize=6)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xticks(range(len(models)), [labels[m] for m in models])
    ax.set_ylabel("nats/event captured on train\n(ln 3 - final loss)")
    ax.bar(0, 0, color="0.3", label="grid L (solid)")
    ax.bar(0, 0, color="0.3", alpha=0.55, label="grid H (faded)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Above the dashed line is memorisation: no model can learn more than the densities contain")
    save(fig, "fig3_memorisation")


# 4 -- MLP ensemble scan (16 members)
def fig_ensemble():
    data = rows("mlp_ensemble16.csv")
    ks = [1, 2, 4, 8]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for g in "LH":
        for ds, ls in (("simulated", "-"), ("observed", "--")):
            sel = lambda k: [r for r in data if r["grid"] == g and r["dataset"] == ds and int(r["K"]) == k]  # noqa
            a.plot(ks, [np.mean([float(r["spread"]) for r in sel(k)]) for k in ks], ls, marker="o",
                   color=GRID_C[g], ms=4, lw=1.2, label=f"grid {g}, {ds}")
            b.plot(ks, [np.mean([abs(float(r["bias"])) for r in sel(k)]) for k in ks], ls, marker="o",
                   color=GRID_C[g], ms=4, lw=1.2)
    for ax, ylab in ((a, "spread of recovered mu across draws"), (b, "|bias| in recovered mu")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("ensemble size K (members averaged)")
        ax.set_ylabel(ylab)
    a.annotate("all 8 draws pinned\nto the grid floor", (8, 0.0), textcoords="offset points", xytext=(-6, 12),
               ha="right", fontsize=5.5, color="0.35")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    a.set_title("spread: lower by K = 8, but not monotonic")
    b.set_title("|bias|: no trend with K")
    fig.suptitle("Toy MLP ensembles, 16 trained members (simulated: mean over truth sweep)", y=1.03)
    save(fig, "fig4_mlp_ensemble")


# 5 -- observed datasets: every config, with what fraction of draws are real estimates
def fig_observed():
    floor = defaultdict(list)
    for r in rows("floor.csv"):
        floor[(r["grid_label"], float(r["truth_mu"]))].append(float(r["clipped_mu"]))
    ens16 = {(r["grid"], int(r["K"])): r for r in rows("mlp_ensemble16.csv") if r["dataset"] == "observed"}
    cats = [("interior, calibrated", OI[2]), ("interior, calibration refused", OI[5]),
            ("vertex inside grid, calibration clamps it to an edge", OI[4]),
            ("vertex outside grid (np.clip)", OI[1]), ("no vertex (concave)", OI[3]),
            ("at edge, cause not kept", "0.7")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [1.6, 1]}, sharex="col")
    for col, g in enumerate("LH"):
        grid, truth = SPEC[g]
        configs = []
        for h in ["linear", "tiny", "small"]:
            configs.append((f"EveNet\n{h}", MODEL[h],
                            [r for r in rows(f"heads_grid{g}.csv") if r["config"] == h and r["dataset"] == "observed"]))
        full = [dict(r, grid=g) for r in rows("heads_scan_gpu.csv")
                if r["grid"] == g and r["head"] == "full" and r["dataset"] == "observed"]
        configs.append(("EveNet\nfull", MODEL["full"], full))
        configs.append(("MLP\nK=1", MODEL["MLP"],
                        [r for r in rows(f"mlp_grid{g}.csv") if r["config"] == "MLP K=1" and r["dataset"] == "observed"]))
        top, bot = axes[0, col], axes[1, col]
        top.axhspan(min(grid), max(grid), color="0.93", lw=0)
        for i, (name, color, rs) in enumerate(configs):
            v = [float(r.get("clipped_mu") or r["recovered_mu"]) for r in rs]
            m, sd = mean_sd(v)
            top.errorbar(i, m, yerr=sd, fmt="o", color=color, ms=5, capsize=2.5)
            frac = np.zeros(len(cats))
            for r in rs:
                if "no_vertex" not in r:
                    k = 5 if is_edge(r) else (0 if r["monotonic"] == "True" else 1)
                elif r["no_vertex"] == "True":
                    k = 4
                elif not min(grid) <= float(r["raw_vertex"]) <= max(grid):
                    k = 3
                else:
                    k = 2 if is_edge(r) else (0 if r["monotonic"] == "True" else 1)
                frac[k] += 1 / len(rs)
            bottom = 0.0
            for (lab, c), f in zip(cats, frac):
                bot.bar(i, f, bottom=bottom, color=c, width=0.7)
                bottom += f
        e = ens16[(g, 8)]
        i8 = len(configs)
        top.errorbar(i8, float(e["recovered_mu"]), yerr=float(e["spread"]), fmt="D", color=MODEL["MLP"],
                     ms=4.5, capsize=2.5)
        bot.text(i8, 0.5, f"16-member\nper-draw\nnot kept\ncal {float(e['monotonic_anchor_rate']):.0%}",
                 ha="center", va="center", fontsize=5.5, color="0.35")
        fm, fsd = mean_sd(floor[("{" + ",".join(f"{x:g}" for x in grid) + "}", truth)])
        top.axhline(truth, color="0.2", ls="--", lw=0.9)
        top.axhline(fm, color="0.2", ls=":", lw=0.9)
        up = fm > truth
        top.text(-0.45, truth, "truth", va="top" if up else "bottom", ha="left", fontsize=5.5)
        top.text(-0.45, fm, f"floor {fm:.3f}", va="bottom" if up else "top", ha="left", fontsize=5.5)
        names = [c[0] for c in configs] + ["MLP\nK=8"]
        bot.set_xticks(range(len(names)), names)
        top.set_title(f"grid {g} {grid}, observed, truth mu = {truth}")
        top.set_ylabel("recovered mu (pooled, +/- sd)")
        bot.set_ylabel("fraction of draws")
        bot.set_ylim(0, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in cats]
    fig.legend(handles, [lab for lab, _ in cats], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Observed data: pooled estimates, and how many draws are actual estimates (grey band = grid range)",
                 y=1.01)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig5_observed")


# 6 -- the clip manufactures in-range answers
def fig_unclipped():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    configs = [("EveNet linear", "heads", "linear", MODEL["linear"]), ("EveNet tiny", "heads", "tiny", MODEL["tiny"]),
               ("EveNet small", "heads", "small", MODEL["small"]), ("MLP K=1", "mlp", "MLP K=1", MODEL["MLP"]),
               ("MLP K=2", "mlp", "MLP K=2", MODEL["MLP"]), ("MLP K=4", "mlp", "MLP K=4", MODEL["MLP"])]
    rng = np.random.default_rng(0)
    for ax, g in zip(axes, "LH"):
        grid, truth = SPEC[g]
        ax.axhspan(min(grid), max(grid), color="0.93", lw=0)
        ax.axhline(truth, color="0.2", ls="--", lw=0.9)
        for i, (name, src, cfg, color) in enumerate(configs):
            rs = [r for r in rows(f"{src}_grid{g}.csv") if r["config"] == cfg and r["dataset"] == "observed"]
            nov = 0
            for r in rs:
                x = i + rng.uniform(-0.18, 0.18)
                clipped = float(r["clipped_mu"])
                if r["no_vertex"] == "True":
                    nov += 1
                    ax.plot(x, clipped, "x", color=color, ms=4.5, mew=1)
                    continue
                raw = float(r["raw_vertex"])
                ax.plot([x, x], [raw, clipped], color=color, lw=0.6, alpha=0.6)
                ax.plot(x, raw, "o", color=color, ms=3.5)
                ax.plot(x, clipped, "o", mfc="white", color=color, ms=3.5)
            ax.annotate(f"{nov}/{len(rs)} no vtx", (i, 1.01), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=5.5, color="0.35")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(range(len(configs)), [c[0].replace(" ", "\n") for c in configs])
        ax.set_title(f"grid {g} {grid}, observed, truth {truth}", pad=14)
    axes[0].set_ylabel("mu (symlog)")
    axes[0].plot([], [], "o", color="0.3", ms=3.5, label="raw vertex -b/2a")
    axes[0].plot([], [], "o", mfc="white", color="0.3", ms=3.5, label="reported (after clip + calibration)")
    axes[0].plot([], [], "x", color="0.3", label="no vertex: argmin fallback")
    axes[0].legend(frameon=False, loc="lower left", fontsize=6)
    fig.suptitle("Raw vertex vs reported value: clip and calibration fold estimates onto the grid (grey band);\n"
                 "x = concave profile, no estimate exists", y=1.08)
    save(fig, "fig6_unclipped")


if __name__ == "__main__":
    for f in (fig_geometry, fig_floor, fig_memorisation, fig_ensemble, fig_observed, fig_unclipped):
        f()
