"""
visualise.py — Publication-ready figures for Phase 1.

Figures produced
----------------
    fig01_model_selection.pdf
    fig02_parameter_evolution.pdf
    fig03_temporal_preference.pdf
    fig04_ks_distributions.pdf
    fig05_akaike_weights.pdf

Usage
-----
    python src/visualise.py              # all figures
    python src/visualise.py --fig 2      # single figure
    python src/visualise.py --no-show    # save only, do not display
    python src/visualise.py --format pdf # pdf only (default: both)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
SCORES_DIR   = ROOT / "outputs" / "model_scores"
TEMPORAL_DIR = ROOT / "outputs" / "temporal"
FIG_DIR      = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "visualise.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

# Colorblind-safe palette (Wong, 2011)
PALETTE = {
    "johnson_su":   "#0072B2",
    "lognormal":    "#E69F00",
    "gamma":        "#009E73",
    "weibull":      "#CC79A7",
    "gev":          "#56B4E9",
    "pearson3":     "#D55E00",
    "gen_logistic": "#F0E442",
    "gumbel":       "#999999",
    "normal":       "#000000",
}

DIST_LABELS = {
    "johnson_su":   "Johnson SU",
    "gen_logistic": "Gen. Logistic",
    "pearson3":     "Pearson III",
    "normal":       "Normal",
    "lognormal":    "Lognormal",
    "gamma":        "Gamma",
    "weibull":      "Weibull",
    "gev":          "GEV",
    "gumbel":       "Gumbel",
}

# Key UK climate events (window mid-year, short label, type)
EVENTS = [
    (1998, "Drought\n1995-96", "drought"),
    (2005, "Floods\n2000",     "flood"),
    (2008, "Drought\n2003",    "drought"),
    (2012, "Floods\n2007",     "flood"),
    (2018, "Floods\n2013-14",  "flood"),
    (2022, "Drought\n2022",    "drought"),
]
EVENT_COLOURS = {"drought": "#D55E00", "flood": "#0072B2"}


def _save(fig: plt.Figure, name: str, fmt: str) -> None:
    for ext in (["pdf", "png"] if fmt == "both" else [fmt]):
        path = FIG_DIR / f"{name}.{ext}"
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(path, **kw)
        log.info("Saved: %s", path.name)


# ── Figure 1: Model selection ──────────────────────────────────────────────────

def fig_model_selection(fmt: str, show: bool) -> None:
    summary = pd.read_parquet(SCORES_DIR / "selection_summary.parquet")
    summary = summary.sort_values("pct_win_aic")   # best at top
    labels  = [DIST_LABELS[d] for d in summary["distribution"]]
    colours = [PALETTE[d]     for d in summary["distribution"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(summary))
    h = 0.26

    # Draw bars per distribution so each has its own colour
    for i, (col, pct_aic, pct_bic, pct_ks) in enumerate(zip(
        colours,
        summary["pct_win_aic"],
        summary["pct_win_bic"],
        summary["pct_win_ks"],
    )):
        ax.barh(y[i] + h, pct_aic, h, color=col, alpha=0.92, edgecolor="white", lw=0.4)
        ax.barh(y[i],     pct_bic, h, color=col, alpha=0.55, edgecolor="white", lw=0.4)
        ax.barh(y[i] - h, pct_ks,  h, color=col, alpha=0.28, edgecolor="white", lw=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("% of stations where distribution ranks first")
    ax.set_title("Distribution model selection — daily fits (~16,000 obs per station)")
    ax.set_xlim(left=0)

    # Criterion legend
    ax.legend(handles=[
        Patch(facecolor="silver", alpha=0.92, edgecolor="grey", label="AIC  (top bar)"),
        Patch(facecolor="silver", alpha=0.55, edgecolor="grey", label="BIC  (middle)"),
        Patch(facecolor="silver", alpha=0.28, edgecolor="grey", label="KS   (bottom)"),
    ], loc="lower right", framealpha=0.9)

    # Annotate all three Johnson SU bars
    js_idx = list(summary["distribution"]).index("johnson_su")
    for bar_offset, col_name in [(h, "pct_win_aic"), (0, "pct_win_bic"), (-h, "pct_win_ks")]:
        val = summary[col_name].iloc[js_idx]
        if val > 0:
            ax.text(val + 0.6, y[js_idx] + bar_offset,
                    f"{val:.1f}%", fontsize=8, fontweight="bold",
                    color=PALETTE["johnson_su"], va="center")

    fig.tight_layout()
    _save(fig, "fig01_model_selection", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 2: Parameter evolution (4-panel) ───────────────────────────────────

def fig_parameter_evolution(fmt: str, show: bool) -> None:
    nat = pd.read_parquet(TEMPORAL_DIR / "national_trends.parquet")
    trt = pd.read_parquet(TEMPORAL_DIR / "trend_tests.parquet")

    params = [
        ("scale", "scale_median", "scale_q25", "scale_q75",
         "Scale  (variance proxy, m)", False),
        ("a",     "a_median",     "a_q25",     "a_q75",
         "Shape  a  (skewness)", False),
        ("b",     "b_median",     "b_q25",     "b_q75",
         "Shape  b  (tail weight)", True),   # inverted: smaller b = heavier tail
        ("loc",   "loc_median",   "loc_q25",   "loc_q75",
         "Location  (m above datum)", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()
    x = nat["window_mid_year"].values

    for ax, (param, med, q25, q75, ylabel, invert) in zip(axes, params):
        col = PALETTE["johnson_su"]

        ax.fill_between(x, nat[q25], nat[q75], color=col, alpha=0.15,
                        label="IQR across stations")
        ax.plot(x, nat[med], color=col, lw=2, label="National median")

        # Linear trend guide
        m, c = np.polyfit(x, nat[med], 1)
        ax.plot(x, m * x + c, "--", color=col, lw=1.2, alpha=0.55, label="Linear trend")

        # Event lines — solid, labelled once via legend on first panel
        for yr, _, etype in EVENTS:
            ax.axvline(yr, color=EVENT_COLOURS[etype], lw=1.3, alpha=0.7, ls="-")

        # Mann-Kendall subtitle
        row = trt[trt["parameter"] == param].iloc[0]
        sig = ("***" if row["pvalue"] < 0.001
               else "**"  if row["pvalue"] < 0.01
               else "*"   if row["pvalue"] < 0.05
               else "ns")
        arrow = "up" if row["trend"] == "increasing" else (
                "dn" if row["trend"] == "decreasing" else "--")
        arrow_sym = {"up": "↑", "dn": "↓", "--": "—"}[arrow]
        ax.set_title(
            f"{arrow_sym} {row['trend'].capitalize()}  "
            f"(tau={row['tau']:+.3f}, p={row['pvalue']:.3f} {sig})",
            fontsize=9,
        )
        ax.set_ylabel(ylabel)

        if invert:
            ax.invert_yaxis()

    for ax in axes[2:]:
        ax.set_xlabel("Window mid-year")

    # Legends
    axes[0].legend(fontsize=8, loc="best")
    axes[1].legend(handles=[
        Line2D([0], [0], color=EVENT_COLOURS["drought"], lw=1.3, label="Drought event"),
        Line2D([0], [0], color=EVENT_COLOURS["flood"],   lw=1.3, label="Flood event"),
    ], fontsize=8, loc="best")

    fig.suptitle(
        "Temporal evolution of Johnson SU parameters — national medians\n"
        "10-year rolling windows, 257-station cohort, 1980-2024",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "fig02_parameter_evolution", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 3: Temporal distribution preference ────────────────────────────────

def fig_temporal_preference(fmt: str, show: bool) -> None:
    sel = pd.read_parquet(SCORES_DIR / "selection_rolling.parquet")
    top5 = ["johnson_su", "lognormal", "gamma", "weibull", "gev"]

    n_per_window = (
        sel.groupby(["window_start_year", "station_reference"])
        .size().reset_index()
        .groupby("window_start_year").size()
        .rename("n_stations")
    )
    wins = (
        sel[sel["rank_aic"] == 1]
        .groupby(["window_start_year", "distribution"])
        .size().reset_index(name="n_wins")
    )
    wins = wins.merge(n_per_window, on="window_start_year")
    wins["pct"] = 100 * wins["n_wins"] / wins["n_stations"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for dist in top5:
        d = wins[wins["distribution"] == dist].sort_values("window_start_year")
        if d.empty:
            continue
        pct_smooth = (d.set_index("window_start_year")["pct"]
                       .rolling(3, center=True, min_periods=1).mean())
        mid = pct_smooth.index + 5
        ax.plot(mid, pct_smooth.values, color=PALETTE[dist], lw=2.2,
                label=DIST_LABELS[dist])
        ax.fill_between(mid, pct_smooth.values, color=PALETTE[dist], alpha=0.07)

    # Event lines — stagger labels above/below axis to avoid overlap
    ymax = ax.get_ylim()[1]
    stagger = [1.02, 1.10, 1.18, 1.10, 1.02, 1.10]   # alternating heights
    for (yr, label, etype), frac in zip(EVENTS, stagger):
        ax.axvline(yr, color=EVENT_COLOURS[etype], lw=1.0, ls="--", alpha=0.65,
                   zorder=1)
        ax.annotate(
            label,
            xy=(yr, 1.0), xycoords=("data", "axes fraction"),
            xytext=(yr, frac), textcoords=("data", "axes fraction"),
            fontsize=7, color=EVENT_COLOURS[etype], ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color=EVENT_COLOURS[etype],
                            lw=0.7, alpha=0.6),
            annotation_clip=False,
        )

    ax.set_xlabel("Window mid-year")
    ax.set_ylabel("% of stations where distribution ranks first by AIC")
    ax.set_title(
        "Rolling distribution preference — top 5 distributions\n"
        "10-year windows, 3-year smoothed"
    )
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, "fig03_temporal_preference", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 4: KS statistic box plot ───────────────────────────────────────────

def fig_ks_distributions(fmt: str, show: bool) -> None:
    dg    = pd.read_parquet(SCORES_DIR / "daily_global_scores.parquet")
    order = (dg.groupby("distribution")["ks_statistic"]
               .median().sort_values().index.tolist())

    data   = [dg[dg["distribution"] == d]["ks_statistic"].dropna().values for d in order]
    labels = [DIST_LABELS[d] for d in order]
    cols   = [PALETTE[d]     for d in order]

    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(
        data, patch_artist=True, notch=False, widths=0.55,
        medianprops={"color": "black", "lw": 2},
        flierprops={"marker": ".", "ms": 3, "alpha": 0.35, "markeredgewidth": 0},
        whiskerprops={"lw": 1},
        capprops={"lw": 1},
    )
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("KS statistic  (lower = better fit)")
    ax.set_title(
        "Kolmogorov-Smirnov statistic by distribution\n"
        "Daily global fits, 257 stations  (ordered by median)"
    )

    # Median annotation for Johnson SU
    js_pos = order.index("johnson_su") + 1
    js_med = float(np.median(dg[dg["distribution"] == "johnson_su"]["ks_statistic"]))
    ax.annotate(
        f"Median\n{js_med:.4f}",
        xy=(js_pos, js_med),
        xytext=(js_pos + 1.4, js_med + 0.012),
        arrowprops=dict(arrowstyle="->", color=PALETTE["johnson_su"], lw=1.2),
        fontsize=8, color=PALETTE["johnson_su"], ha="center",
    )

    fig.tight_layout()
    _save(fig, "fig04_ks_distributions", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 5: Akaike weights ───────────────────────────────────────────────────

def fig_akaike_weights(fmt: str, show: bool) -> None:
    sel   = pd.read_parquet(SCORES_DIR / "selection_global.parquet")
    order = (sel.groupby("distribution")["akaike_weight"]
               .mean().sort_values(ascending=False).index.tolist())

    data   = [sel[sel["distribution"] == d]["akaike_weight"].dropna().values for d in order]
    labels = [DIST_LABELS[d] for d in order]
    cols   = [PALETTE[d]     for d in order]

    # Two-panel: linear scale (top, Johnson SU context) +
    #            log scale (bottom, shows spread for all others)
    fig, (ax_lin, ax_log) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [1, 1]},
        sharex=True,
    )

    for ax, yscale in [(ax_lin, "linear"), (ax_log, "log")]:
        bp = ax.boxplot(
            data, patch_artist=True, notch=False, widths=0.55,
            medianprops={"color": "black", "lw": 2},
            flierprops={"marker": ".", "ms": 3, "alpha": 0.3, "markeredgewidth": 0},
            whiskerprops={"lw": 1},
            capprops={"lw": 1},
        )
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        ax.set_yscale(yscale)
        ax.set_ylabel("Akaike weight" + (" (log scale)" if yscale == "log" else ""))
        # Equal-support reference
        ax.axhline(1 / 9, color="grey", lw=1, ls="--", alpha=0.7,
                   label="Equal support  (1/9 = 0.111)")
        if yscale == "linear":
            ax.legend(fontsize=8)

    ax_log.set_xticks(range(1, len(labels) + 1))
    ax_log.set_xticklabels(labels, rotation=30, ha="right")

    # Annotate mean weights for top 3
    # For JS (i=0) whose box extends near 1.0, place label above the top whisker;
    # for others place just above their mean value.
    js_max = float(np.max(sel[sel["distribution"] == order[0]]["akaike_weight"]))
    for i, dist in enumerate(order[:3]):
        mean_w = float(np.mean(sel[sel["distribution"] == dist]["akaike_weight"]))
        y_pos  = js_max + 0.03 if i == 0 else mean_w + 0.02
        ax_lin.text(i + 1, y_pos, f"{mean_w:.3f}",
                    ha="center", fontsize=8, color=cols[i], fontweight="bold")

    fig.suptitle(
        "Akaike weights by distribution — daily global fits\n"
        "257 stations; top panel linear scale, bottom panel log scale",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "fig05_akaike_weights", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

FIGURE_MAP = {
    1: ("fig01_model_selection",     fig_model_selection),
    2: ("fig02_parameter_evolution", fig_parameter_evolution),
    3: ("fig03_temporal_preference", fig_temporal_preference),
    4: ("fig04_ks_distributions",    fig_ks_distributions),
    5: ("fig05_akaike_weights",      fig_akaike_weights),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig",     type=int, default=None,
                   help="Single figure number (1-5); omit for all")
    p.add_argument("--no-show", action="store_true",
                   help="Save files only, do not display")
    p.add_argument("--format",  choices=["pdf", "png", "both"], default="both")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    show    = not args.no_show
    targets = [args.fig] if args.fig else list(FIGURE_MAP.keys())

    for n in targets:
        if n not in FIGURE_MAP:
            log.error("Unknown figure number %d  (choose 1-5)", n)
            continue
        name, fn = FIGURE_MAP[n]
        log.info("Producing %s ...", name)
        try:
            fn(args.format, show)
        except Exception as exc:
            log.error("Figure %d failed: %s", n, exc)
            raise

    log.info("Done. Figures in %s", FIG_DIR)


if __name__ == "__main__":
    main()
