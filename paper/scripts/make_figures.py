"""Builds every figure in the manuscript from paper/data (real rollouts only)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "research_paper_code"))
from geometry_utils import points_to_polyline_distance  # noqa: E402
DATA = ROOT / "paper" / "data"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

C = {"ref": "#555555", "bead_ppo": "#2ca02c", "joint": "#1f77b4", "ld_only": "#ff7f0e",
     "adaptive": "#9467bd", "fixed": "#d62728"}
LBL = {"bead_ppo": "Bead-PPO (ours)", "joint": r"RL–PP (joint $L_d,g$)", "ld_only": r"RL–PP ($L_d$ only)",
       "adaptive": r"Adaptive PP ($v\mapsto L_d$)", "fixed": "Fixed PP"}
ORDER = ["bead_ppo", "joint", "ld_only", "adaptive", "fixed"]
TRACKS = ["Hockenheim", "Montreal", "Yas Marina"]
S = json.loads((DATA / "summary.json").read_text())


def trace(track, ctrl):
    return np.load(DATA / f"trace_{track.replace(' ', '')}_{ctrl}.npz")


# ---------------------------------------------------------------- Fig. 1 pipeline
def fig_pipeline():
    # Single-column layout: three stacked stage bands, each holding three steps left to right.
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 90); ax.axis("off")
    bands = [("Data / Perception", "#0f7b7b", 61), ("Learning (PPO + DAgger)", "#c9c9c9", 31),
             ("Evaluation / Benchmark", "#f5b700", 1)]
    for title, col, y in bands:
        ax.add_patch(Rectangle((0.5, y), 99, 28, color=col, zorder=0))
        ax.add_patch(Rectangle((28, y + 21.5), 44, 5, facecolor="white", edgecolor="#333", lw=0.5))
        ax.text(50, y + 24, title, ha="center", va="center", fontsize=6.4, weight="bold")

    def box(x, y, txt, fs=5.4):
        ax.add_patch(FancyBboxPatch((x, y), 28, 13, boxstyle="round,pad=0.6", facecolor="white",
                                    edgecolor="#333", lw=0.5))
        ax.text(x + 14, y + 6.5, txt, ha="center", va="center", fontsize=fs, linespacing=1.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#1b1b1b", mutation_scale=6))

    rows = [("Raster image\nOtsu threshold\n+ morphology",
             "Zhang–Suen skeleton\n/ contour → ordered\npolyline $\\{p_i\\}$",
             "Arc-length resampling\n(350 pts) → 20×20 m\narena frame"),
            ("Bead environment\n(Gymnasium) 16-D obs,\n2-D accel. action",
             "Geometric expert\n+ DAgger (β: 0.4 → 0),\n3 circuits",
             "PPO actor–critic\nMLP [256, 256]\ntanh, SB3"),
            ("Fixed / Adaptive PP,\nRL–PP ($L_d$) and\nRL–PP ($L_d,g$)",
             "Common metrics:\nGeom@τ, RMSE,\nP95, traversal time",
             "Held-out shapes\n(zero-shot) +\nF1TENTH circuits")]
    for (t0, t1, t2), (_, _, y) in zip(rows, bands):
        yb = y + 4
        box(3, yb, t0); box(36, yb, t1); box(69, yb, t2)
        arrow(31.8, yb + 6.5, 35.2, yb + 6.5); arrow(64.8, yb + 6.5, 68.2, yb + 6.5)
    arrow(83, 64.4, 83, 51.6); arrow(83, 34.4, 83, 21.6)
    fig.savefig(FIG / "fig1_pipeline.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 2 data
def fig_dataset():
    imgs = [("(a) Hockenheim", ROOT / "datasets/test_images/hockenheim_track.png"),
            ("(b) Montreal", ROOT / "datasets/test_images/montreal_track.png"),
            ("(c) Yas Marina", ROOT / "datasets/test_images/yasmarina_track.png")]
    closed = sorted((ROOT / "datasets/closed_shapes_dataset/train").glob("*.jpg"))
    curve = sorted((ROOT / "datasets/curve_dataset/train").glob("*.jpg"))
    imgs += [("(d) closed-shape", closed[3]), ("(e) closed-shape", closed[11]),
             ("(f) curve", curve[2]), ("(g) curve", curve[9])]
    fig = plt.figure(figsize=(3.5, 1.95))
    gs = fig.add_gridspec(2, 12, hspace=0.35, wspace=0.1)
    slots = [gs[0, 0:4], gs[0, 4:8], gs[0, 8:12], gs[1, 0:3], gs[1, 3:6], gs[1, 6:9], gs[1, 9:12]]
    for sl, (t, p) in zip(slots, imgs):
        ax = fig.add_subplot(sl)
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        ax.imshow(im); ax.set_title(t, fontsize=6.3, y=-0.3); ax.axis("off")
    fig.savefig(FIG / "fig2_dataset.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 3 bead traces
def fig_bead_traces():
    fig = plt.figure(figsize=(3.5, 3.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.3], width_ratios=[0.75, 1.0], hspace=0.78, wspace=0.3)
    axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, t in zip(axs, TRACKS):
        d = trace(t, "bead_ppo"); p, tr = d["path"], d["traj"]
        dr = points_to_polyline_distance(p, tr)
        ax.plot(p[:, 0], p[:, 1], color="#bbbbbb", lw=3.2, solid_capstyle="round", label="Reference", zorder=1)
        ax.plot(tr[:, 0], tr[:, 1], color=C["bead_ppo"], lw=0.9, label="Bead-PPO trajectory", zorder=2)
        bad = dr > 0.10
        ax.scatter(p[bad, 0], p[bad, 1], marker="x", s=7, lw=0.6, color="#d62728", zorder=3,
                   label=r"ref. point $>0.10$ m")
        ax.scatter(*tr[0], s=16, color="k", zorder=4, label="start")
        m = S[t]["bead_ppo"]
        ax.set_title(f"({'abc'[TRACKS.index(t)]}) {t}\nGeom@0.10 = {100*m['geom010']['mean']:.1f}%\n"
                     f"RMSE = {100*m['rmse']['mean']:.2f} cm", fontsize=7)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.tick_params(length=2)
        ax.set_xlabel("$x$ [m]")
    for ax in axs[:2]:
        ax.set_ylabel("$y$ [m]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=6,
               columnspacing=0.9, handlelength=1.4)
    fig.savefig(FIG / "fig3_bead_traces.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 4 zoom comparison
def fig_zoom():
    wins = {"Montreal": (-3.9, -2.6, -8.85, -6.9), "Yas Marina": (-4.7, -3.2, -5.2, -3.6)}
    fig = plt.figure(figsize=(3.5, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.55, 1.0], hspace=0.55, wspace=0.12)
    for j, (t, (x0, x1, y0, y1)) in enumerate(wins.items()):
        axf = fig.add_subplot(gs[j, 0])
        p = trace(t, "fixed")["path"]
        axf.plot(p[:, 0], p[:, 1], color=C["ref"], lw=0.8)
        axf.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#d62728", ls="--", lw=0.9))
        axf.set_aspect("equal"); axf.set_title(f"({'ac'[j]}) {t}"); axf.set_xticks([]); axf.set_yticks([])
        axz = fig.add_subplot(gs[j, 1])
        axz.set_title(f"({'bd'[j]}) {t}: zoom")
        axz.plot(p[:, 0], p[:, 1], color="#999", lw=4.5, alpha=0.5, label="Reference")
        for c in ORDER[::-1]:
            tr = trace(t, c)["traj"]
            axz.plot(tr[:, 0], tr[:, 1], color=C[c], lw=1.1 if c == "bead_ppo" else 0.9, label=LBL[c])
        axz.set_xlim(x0, x1); axz.set_ylim(y0, y1); axz.set_aspect("equal"); axz.grid(alpha=0.3)
        axz.tick_params(length=2); axz.set_xlabel("$x$ [m]")
        axz.set_ylabel("$y$ [m]")
    h, l = axz.get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.11), frameon=False,
               fontsize=6)
    fig.savefig(FIG / "fig4_zoom.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 5 bars
def fig_bars():
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75))
    x = np.arange(3); w = 0.16
    for k, c in enumerate(ORDER):
        g = [100 * S[t][c]["geom010"]["mean"] for t in TRACKS]
        r = [100 * S[t][c]["rmse"]["mean"] for t in TRACKS]
        axs[0].bar(x + (k - 2) * w, g, w, color=C[c], label=LBL[c])
        axs[1].bar(x + (k - 2) * w, r, w, color=C[c])
    axs[0].set_ylim(60, 101); axs[0].set_ylabel("Geom@0.10 [%] ↑")
    axs[1].set_ylabel("Tracking RMSE [cm] ↓")
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels(["Hock.", "Mont.", "Yas M."]); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(length=2)
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=6)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "fig5_bars.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 6 error CDF
def fig_cdf():
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.9), sharex=True, sharey=True)
    axs = axs.flat
    for ax, t in zip(axs, TRACKS):
        for c in ORDER:
            e = np.sort(trace(t, c)["dt"]) * 100
            ax.plot(e, np.linspace(0, 1, len(e)), color=C[c], lw=1.0, label=LBL[c])
        ax.axvline(10, color="k", ls=":", lw=0.6)
        ax.set_xlim(0, 30); ax.set_title(t, pad=2); ax.grid(alpha=0.3)
        ax.tick_params(length=2, labelbottom=True)
    for ax in (axs[1], axs[2]):
        ax.set_xlabel("Lateral error $e$ [cm]")
    for ax in (axs[0], axs[2]):
        ax.set_ylabel(r"$\Pr(E \leq e)$")
    h, l = axs[0].get_legend_handles_labels()
    axs[3].axis("off"); axs[3].legend(h, l, loc="center", frameon=False, fontsize=6.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(FIG / "fig6_cdf.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 7 RL-PP interpretability
def fig_interp():
    d = trace("Montreal", "joint")
    L, G, K, V = d["L"], d["G"], d["K"], d["V"]
    fig = plt.figure(figsize=(3.5, 3.15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.95, wspace=0.42)
    for i, (y, yl) in enumerate([(L, "$L_d$ [m]"), (G, "$g$")]):
        for j, (xv, xl) in enumerate([(K, r"$\kappa^{\max}$ [m$^{-1}$]"), (V, "$v$ [m/s]")]):
            ax = fig.add_subplot(gs[i, j])
            ax.scatter(xv, y, s=1.5, color=C["joint"], alpha=0.5, rasterized=True)
            ax.set_xlabel(xl, labelpad=1); ax.set_ylabel(yl, labelpad=1); ax.grid(alpha=0.3); ax.tick_params(length=2)
    ax = fig.add_subplot(gs[2, :])
    s = np.arange(len(L))
    ax.plot(s, L, color=C["joint"], lw=0.9, label="$L_d$")
    ax2 = ax.twinx(); ax2.plot(s, G, color=C["adaptive"], lw=0.9, label="$g$")
    ax.fill_between(s, 0, K / max(K.max(), 1e-6) * (L.max() - L.min()) * 1.2 + L.min() * 0.98,
                    color="#bbbbbb", alpha=0.45, lw=0, label=r"$\kappa^{\max}$ (norm.)")
    ax.set_ylim(L.min() * 0.98, L.max() * 1.02)
    ax.set_xlabel("Control step", labelpad=1); ax.set_ylabel("$L_d$ [m]", color=C["joint"], labelpad=1)
    ax2.set_ylabel("$g$", color=C["adaptive"], labelpad=1); ax.tick_params(length=2); ax2.tick_params(length=2)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.3), frameon=False)
    fig.savefig(FIG / "fig7_interp.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 8 zero-shot shapes
def fig_shapes():
    files = [DATA / f"shape_{s}_{k}.npz" for s in ("closed", "curve") for k in range(4)]
    fig, axs = plt.subplots(2, 4, figsize=(3.5, 1.85))
    for ax, f in zip(axs.flat, files):
        d = np.load(f)
        ax.plot(d["path"][:, 0], d["path"][:, 1], color="#bbbbbb", lw=2.6)
        ax.plot(d["joint"][:, 0], d["joint"][:, 1], color=C["joint"], lw=0.6)
        ax.plot(d["bead"][:, 0], d["bead"][:, 1], color=C["bead_ppo"], lw=0.7)
        ax.set_title(f"{100*float(d['gb']):.1f}% / {100*float(d['gj']):.1f}%", fontsize=6.2, pad=1.5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
    fig.subplots_adjust(wspace=0.06, hspace=0.25)
    fig.savefig(FIG / "fig8_shapes.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 9 along-lap error
def fig_along():
    d = trace("Yas Marina", "bead_ppo"); j = trace("Yas Marina", "joint")
    fig, ax = plt.subplots(figsize=(3.5, 1.45))
    for dd, c in ((j, "joint"), (d, "bead_ppo")):
        e = dd["dt"] * 100
        ax.plot(np.linspace(0, 1, len(e)), e, color=C[c], lw=0.7, label=LBL[c])
    ax.axhline(10, color="k", ls=":", lw=0.6)
    ax.set_xlabel("Normalised lap progress $s/S$"); ax.set_ylabel("$e$ [cm]"); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.legend(loc="upper left", ncol=2, frameon=False); ax.tick_params(length=2)
    fig.savefig(FIG / "fig9_along.png"); plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    fig_pipeline(); fig_dataset(); fig_bead_traces(); fig_zoom(); fig_bars(); fig_cdf()
    fig_interp(); fig_shapes(); fig_along()
    print("figures ->", FIG)
