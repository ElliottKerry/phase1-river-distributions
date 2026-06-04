"""
app.py — Interactive Streamlit dashboard for Phase 1 river distribution analysis.

Run from the project root:
    streamlit run src/app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
SCORES_DIR   = ROOT / "outputs" / "model_scores"
TEMPORAL_DIR = ROOT / "outputs" / "temporal"
DATA_DIR     = ROOT / "data" / "processed"

# ── Constants ──────────────────────────────────────────────────────────────────
PALETTE = {
    "johnson_su":   "#0072B2",
    "lognormal":    "#E69F00",
    "gamma":        "#009E73",
    "weibull":      "#CC79A7",
    "gev":          "#56B4E9",
    "pearson3":     "#D55E00",
    "gen_logistic": "#F0E442",
    "gumbel":       "#999999",
    "normal":       "#4d4d4d",
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

PARAM_META = {
    "p1": {"js_name": "a",     "label": "a  (skewness shape)",          "invert": False},
    "p2": {"js_name": "b",     "label": "b  (tail weight — lower=heavier)", "invert": True},
    "p3": {"js_name": "loc",   "label": "loc  (location, m)",           "invert": False},
    "p4": {"js_name": "scale", "label": "scale  (variance proxy, m)",   "invert": False},
}

EVENTS = [
    (1998, "Drought 1995-96", "drought"),
    (2005, "Floods 2000",     "flood"),
    (2008, "Drought 2003",    "drought"),
    (2012, "Floods 2007",     "flood"),
    (2018, "Floods 2013-14",  "flood"),
    (2022, "Drought 2022",    "drought"),
]
EVENT_COLOURS = {"drought": "#D55E00", "flood": "#0072B2"}


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_selection_global() -> pd.DataFrame:
    return pd.read_parquet(SCORES_DIR / "selection_global.parquet")

@st.cache_data
def load_selection_rolling() -> pd.DataFrame:
    return pd.read_parquet(SCORES_DIR / "selection_rolling.parquet")

@st.cache_data
def load_selection_summary() -> pd.DataFrame:
    return pd.read_parquet(SCORES_DIR / "selection_summary.parquet")

@st.cache_data
def load_national_trends() -> pd.DataFrame:
    return pd.read_parquet(TEMPORAL_DIR / "national_trends.parquet")

@st.cache_data
def load_trend_tests() -> pd.DataFrame:
    return pd.read_parquet(TEMPORAL_DIR / "trend_tests.parquet")

@st.cache_data
def load_event_context() -> pd.DataFrame:
    return pd.read_parquet(TEMPORAL_DIR / "event_context.parquet")

@st.cache_data
def load_station_trajectories() -> pd.DataFrame:
    return pd.read_parquet(TEMPORAL_DIR / "station_trajectories.parquet")

@st.cache_data
def load_cohort() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "cohort.parquet")

@st.cache_data
def compute_rolling_win_rates() -> pd.DataFrame:
    """Win rates per distribution per window year, from selection_rolling."""
    sr = load_selection_rolling()
    n_per = sr.groupby("window_start_year")["station_reference"].nunique().rename("n_stations")
    wins  = (
        sr[sr["rank_aic"] == 1]
        .groupby(["window_start_year", "distribution"])
        .size().reset_index(name="n_wins")
    )
    wins = wins.merge(n_per, on="window_start_year")
    wins["pct"]             = 100 * wins["n_wins"] / wins["n_stations"]
    wins["window_mid_year"] = wins["window_start_year"] + 5
    return wins


# ── Plotly helpers ─────────────────────────────────────────────────────────────

def _add_event_vrects(fig: go.Figure, row: int = 1, col: int = 1) -> None:
    """Add vertical event lines to a plotly figure."""
    for yr, label, etype in EVENTS:
        fig.add_vline(
            x=yr, line_width=1.2, line_dash="dot",
            line_color=EVENT_COLOURS[etype],
            annotation_text=label.replace("\n", " "),
            annotation_position="top",
            annotation_font_size=9,
            annotation_font_color=EVENT_COLOURS[etype],
            row=row, col=col,
        )


def _plotly_layout(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(
        font_family="serif",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", borderwidth=1),
        **kwargs,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0", gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", gridwidth=1, zeroline=False)
    return fig


# ── Page: Model Selection ──────────────────────────────────────────────────────

def page_model_selection() -> None:
    st.header("Model Selection")
    st.markdown(
        "Compare distribution fit quality across all 257 stations. "
        "**Daily global** uses each station's full 45-year record (~16,000 observations). "
        "**Daily rolling** uses 10-year windows stepped annually."
    )

    tab_global, tab_rolling = st.tabs(["Daily Global", "Daily Rolling"])

    # ── Tab: Daily Global ──────────────────────────────────────────────────────
    with tab_global:
        summary = load_selection_summary().sort_values("pct_win_aic", ascending=False)
        sg      = load_selection_global()

        criterion = st.radio(
            "Ranking criterion", ["AIC", "BIC", "KS"],
            horizontal=True, key="glob_criterion",
        )
        col_map = {"AIC": "pct_win_aic", "BIC": "pct_win_bic", "KS": "pct_win_ks"}
        sort_col = col_map[criterion]

        df_bar = summary.sort_values(sort_col).copy()
        df_bar["label"]  = df_bar["distribution"].map(DIST_LABELS)
        df_bar["colour"] = df_bar["distribution"].map(PALETTE)
        df_bar["pct_win_aic_fmt"]  = df_bar["pct_win_aic"].map("{:.1f}%".format)
        df_bar["pct_win_bic_fmt"]  = df_bar["pct_win_bic"].map("{:.1f}%".format)
        df_bar["pct_win_ks_fmt"]   = df_bar["pct_win_ks"].map("{:.1f}%".format)
        df_bar["mean_akaike_fmt"]  = df_bar["mean_akaike"].map("{:.4f}".format)

        fig = go.Figure()
        for _, row in df_bar.iterrows():
            fig.add_trace(go.Bar(
                y=[row["label"]],
                x=[row["pct_win_aic"]],
                name=row["label"],
                orientation="h",
                marker_color=row["colour"],
                opacity=0.95 if criterion == "AIC" else 0.3,
                customdata=[[
                    row["pct_win_aic_fmt"],
                    row["pct_win_bic_fmt"],
                    row["pct_win_ks_fmt"],
                    row["mean_akaike_fmt"],
                ]],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "AIC wins: %{customdata[0]}<br>"
                    "BIC wins: %{customdata[1]}<br>"
                    "KS wins:  %{customdata[2]}<br>"
                    "Mean Akaike wt: %{customdata[3]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))
            # Add the chosen criterion bar on top (full opacity)
            if criterion != "AIC":
                pct_val = row[sort_col]
                fig.add_trace(go.Bar(
                    y=[row["label"]],
                    x=[pct_val],
                    name=row["label"],
                    orientation="h",
                    marker_color=row["colour"],
                    opacity=0.92,
                    showlegend=False,
                    hoverinfo="skip",
                ))

        _plotly_layout(
            fig,
            title=f"Win rate by {criterion}  —  daily global fits, 257 stations",
            xaxis_title=f"% of stations where distribution ranks first ({criterion})",
            barmode="overlay",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics row for Johnson SU
        js = summary[summary["distribution"] == "johnson_su"].iloc[0]
        runner = summary[summary["distribution"] != "johnson_su"].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Johnson SU — AIC wins", f"{js['pct_win_aic']:.1f}%")
        c2.metric("Mean Akaike weight",     f"{js['mean_akaike']:.4f}")
        c3.metric("Median KS statistic",    f"{js['median_ks']:.4f}")
        c4.metric(
            f"Evidence vs {DIST_LABELS[runner['distribution']]}",
            f"{js['mean_akaike'] / runner['mean_akaike']:.1f}×",
        )

        with st.expander("Full summary table"):
            cols_show = ["distribution", "pct_win_aic", "pct_win_bic", "pct_win_ks",
                         "mean_akaike", "median_ks", "evidence_vs_johnson_su"]
            df_show = summary[cols_show].copy()
            df_show["distribution"] = df_show["distribution"].map(DIST_LABELS)
            df_show.columns = ["Distribution", "AIC wins %", "BIC wins %",
                                "KS wins %", "Mean Akaike wt.", "Median KS", "Evidence vs JS"]
            st.dataframe(df_show.round(4), use_container_width=True, hide_index=True)

        with st.expander("KS statistic distribution by model"):
            order = (sg.groupby("distribution")["ks_statistic"]
                       .median().sort_values().index.tolist())
            fig_ks = go.Figure()
            for dist in order:
                vals = sg[sg["distribution"] == dist]["ks_statistic"].dropna()
                fig_ks.add_trace(go.Box(
                    x=vals,
                    name=DIST_LABELS[dist],
                    marker_color=PALETTE[dist],
                    boxmean=True,
                    orientation="h",
                ))
            _plotly_layout(
                fig_ks,
                title="KS statistic by distribution  (lower = better fit)",
                xaxis_title="KS statistic",
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig_ks, use_container_width=True)

    # ── Tab: Daily Rolling ─────────────────────────────────────────────────────
    with tab_rolling:
        sr       = load_selection_rolling()
        win_rates = compute_rolling_win_rates()

        top5 = ["johnson_su", "lognormal", "gamma", "weibull", "gev"]
        sel_dists = st.multiselect(
            "Distributions to show",
            options=list(DIST_LABELS.keys()),
            default=top5,
            format_func=lambda x: DIST_LABELS[x],
            key="roll_dists",
        )

        smooth = st.slider("Smoothing window (years)", 1, 7, 3, step=2, key="roll_smooth")
        show_events = st.checkbox("Show climate events", value=True, key="roll_events")

        fig_roll = go.Figure()
        for dist in sel_dists:
            d = win_rates[win_rates["distribution"] == dist].sort_values("window_mid_year")
            if d.empty:
                continue
            pct_s = (d.set_index("window_mid_year")["pct"]
                      .rolling(smooth, center=True, min_periods=1).mean())
            fig_roll.add_trace(go.Scatter(
                x=pct_s.index, y=pct_s.values,
                mode="lines",
                name=DIST_LABELS[dist],
                line=dict(color=PALETTE[dist], width=2.5),
                hovertemplate=(
                    f"<b>{DIST_LABELS[dist]}</b><br>"
                    "Year: %{x}<br>"
                    "Win rate: %{y:.1f}%<extra></extra>"
                ),
            ))

        if show_events:
            for yr, label, etype in EVENTS:
                fig_roll.add_vline(
                    x=yr, line_width=1, line_dash="dot",
                    line_color=EVENT_COLOURS[etype],
                    annotation_text=label,
                    annotation_position="top",
                    annotation_font_size=8,
                    annotation_font_color=EVENT_COLOURS[etype],
                )

        _plotly_layout(
            fig_roll,
            title=f"Rolling distribution preference — 10-year windows, {smooth}-yr smoothed",
            xaxis_title="Window mid-year",
            yaxis_title="% of stations where distribution ranks first (AIC)",
            height=420,
            yaxis_rangemode="tozero",
        )
        st.plotly_chart(fig_roll, use_container_width=True)

        with st.expander("Win rate by decade (AIC)"):
            temporal = pd.read_parquet(SCORES_DIR / "selection_temporal.parquet")
            pivot = (
                temporal[temporal["distribution"].isin(sel_dists)]
                .pivot(index="distribution", columns="decade", values="pct_win_aic")
                .rename(index=DIST_LABELS)
                .round(1)
            )
            pivot.columns = [f"{c}s" for c in pivot.columns]
            st.dataframe(pivot, use_container_width=True)


# ── Page: Parameter Evolution ──────────────────────────────────────────────────

def page_parameter_evolution() -> None:
    st.header("Johnson SU Parameter Evolution")
    st.markdown(
        "National median (± IQR) of each Johnson SU parameter, "
        "derived from 10-year rolling daily fits across 257 cohort stations."
    )

    nat = load_national_trends()
    trt = load_trend_tests()
    ctx = load_event_context()

    show_events = st.checkbox("Show climate events", value=True, key="param_events")
    show_iqr    = st.checkbox("Show IQR band",       value=True, key="param_iqr")

    param_opts = {
        "scale": ("scale_median", "scale_q25", "scale_q75",  "Scale  (variance proxy, m)"),
        "a":     ("a_median",     "a_q25",     "a_q75",      "Shape  a  (skewness)"),
        "b":     ("b_median",     "b_q25",     "b_q75",      "Shape  b  (tail weight)"),
        "loc":   ("loc_median",   "loc_q25",   "loc_q75",    "Location  (m above datum)"),
    }
    param_labels = {"scale": "Scale", "a": "a (skewness)", "b": "b (tail)", "loc": "Location"}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[param_labels[p] for p in param_opts],
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )
    col_js = "#0072B2"
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (pkey, (med, q25, q75, ylabel)), (row, col) in zip(param_opts.items(), positions):
        x = nat["window_mid_year"].values

        if show_iqr:
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([nat[q75].values, nat[q25].values[::-1]]),
                fill="toself", fillcolor=f"rgba(0,114,178,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="IQR", showlegend=(pkey == "scale"),
                hoverinfo="skip",
            ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=x, y=nat[med].values,
            mode="lines", line=dict(color=col_js, width=2.5),
            name="National median", showlegend=(pkey == "scale"),
            hovertemplate=(
                f"<b>{param_labels[pkey]}</b><br>"
                "Year: %{x}<br>"
                "Median: %{y:.4f}<extra></extra>"
            ),
        ), row=row, col=col)

        # Linear trend
        m, c = np.polyfit(x, nat[med].values, 1)
        fig.add_trace(go.Scatter(
            x=x, y=m * x + c,
            mode="lines", line=dict(color=col_js, width=1.2, dash="dash"),
            opacity=0.5, name="Linear trend", showlegend=(pkey == "scale"),
            hoverinfo="skip",
        ), row=row, col=col)

        # MK annotation in subtitle
        mk = trt[trt["parameter"] == pkey].iloc[0]
        sig = "***" if mk["pvalue"] < 0.001 else "**" if mk["pvalue"] < 0.01 else "*" if mk["pvalue"] < 0.05 else "ns"
        arrow = "↑" if mk["trend"] == "increasing" else "↓" if mk["trend"] == "decreasing" else "—"
        fig.layout.annotations[list(param_opts.keys()).index(pkey)].text = (
            f"{param_labels[pkey]}  {arrow} {mk['trend']} "
            f"(τ={mk['tau']:+.3f}, p={mk['pvalue']:.3f} {sig})"
        )

        if show_events:
            for yr, label, etype in EVENTS:
                fig.add_vline(
                    x=yr, line_width=1, line_dash="dot",
                    line_color=EVENT_COLOURS[etype],
                    row=row, col=col,
                )

        # Invert b axis (smaller b = heavier tail)
        if pkey == "b":
            fig.update_yaxes(autorange="reversed", row=row, col=col)

    fig.update_xaxes(title_text="Window mid-year", row=2, col=1)
    fig.update_xaxes(title_text="Window mid-year", row=2, col=2)
    _plotly_layout(
        fig,
        title="Temporal evolution of Johnson SU parameters — national medians<br>"
              "<sup>10-year rolling windows, 257-station cohort, 1980–2024</sup>",
        height=620,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mann-Kendall trend tests")
        df_mk = trt[["label", "tau", "pvalue", "trend", "change_pct"]].copy()
        df_mk.columns = ["Parameter", "Kendall τ", "p-value", "Trend", "Change %"]
        st.dataframe(df_mk.round(4), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Parameter values at key climate events")
        st.dataframe(ctx.round(4), use_container_width=True, hide_index=True)


# ── Page: Station Explorer ─────────────────────────────────────────────────────

def page_station_explorer() -> None:
    st.header("Station Explorer")
    st.markdown(
        "Drill into any individual gauging station — see which distribution "
        "fitted best in each rolling window and how Johnson SU parameters evolved."
    )

    cohort = load_cohort()
    sr     = load_selection_rolling()
    traj   = load_station_trajectories()

    station_opts = {
        row["station_reference"]: f"{row['station_reference']}  —  {row['label']}  ({row['river_name']})"
        for _, row in cohort.sort_values("station_reference").iterrows()
    }
    selected_ref = st.selectbox(
        "Select station",
        options=list(station_opts.keys()),
        format_func=lambda x: station_opts[x],
    )

    info = cohort[cohort["station_reference"] == selected_ref].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Station",    info["station_reference"])
    c2.metric("River",      info["river_name"])
    c3.metric("Period",     f"{info['first_year']}–{info['last_year']}")
    c4.metric("Coverage",   f"{info['year_coverage_frac']:.0%}")

    tab_winners, tab_params, tab_mk = st.tabs(
        ["Distribution Winners", "JS Parameters", "Trend Summary"]
    )

    # ── Tab: Distribution Winners ──────────────────────────────────────────────
    with tab_winners:
        station_sr = sr[sr["station_reference"] == selected_ref].copy()
        winners = (
            station_sr[station_sr["rank_aic"] == 1]
            .sort_values("window_start_year")
        )
        winners["label"]          = winners["distribution"].map(DIST_LABELS)
        winners["window_mid_year"]= winners["window_start_year"] + 5
        winners["colour"]         = winners["distribution"].map(PALETTE)

        criterion_w = st.radio(
            "Ranking criterion", ["AIC", "BIC", "KS"],
            horizontal=True, key="win_criterion",
        )
        rank_col = {"AIC": "rank_aic", "BIC": "rank_bic", "KS": "rank_ks"}[criterion_w]
        winners_w = (
            station_sr[station_sr[rank_col] == 1]
            .sort_values("window_start_year")
            .copy()
        )
        winners_w["label"]           = winners_w["distribution"].map(DIST_LABELS)
        winners_w["window_mid_year"] = winners_w["window_start_year"] + 5
        winners_w["colour"]          = winners_w["distribution"].map(PALETTE)

        fig_w = go.Figure()
        for dist in DIST_LABELS:
            d = winners_w[winners_w["distribution"] == dist]
            if d.empty:
                continue
            fig_w.add_trace(go.Scatter(
                x=d["window_mid_year"],
                y=[DIST_LABELS[dist]] * len(d),
                mode="markers",
                marker=dict(color=PALETTE[dist], size=14, symbol="square"),
                name=DIST_LABELS[dist],
                hovertemplate=(
                    f"<b>{DIST_LABELS[dist]}</b><br>"
                    "Window mid: %{x}<br>"
                    f"AIC: %{{customdata[0]:.1f}}<br>"
                    f"KS:  %{{customdata[1]:.4f}}"
                    "<extra></extra>"
                ),
                customdata=d[["aic", "ks_statistic"]].values,
            ))

        _plotly_layout(
            fig_w,
            title=f"Best distribution per window ({criterion_w}) — {station_opts[selected_ref]}",
            xaxis_title="Window mid-year",
            height=350,
        )
        st.plotly_chart(fig_w, use_container_width=True)

        # Show counts
        counts = (winners_w.groupby("distribution").size()
                  .rename("n_windows").reset_index()
                  .assign(distribution=lambda d: d["distribution"].map(DIST_LABELS))
                  .sort_values("n_windows", ascending=False))
        st.dataframe(counts, use_container_width=True, hide_index=True)

    # ── Tab: JS Parameters ─────────────────────────────────────────────────────
    with tab_params:
        js_data = (
            sr[
                (sr["station_reference"] == selected_ref) &
                (sr["distribution"] == "johnson_su") &
                (sr["converged"] == True)
            ]
            .sort_values("window_start_year")
            .copy()
        )
        js_data["window_mid_year"] = js_data["window_start_year"] + 5

        if js_data.empty:
            st.warning("No converged Johnson SU fits for this station.")
        else:
            # Rename p1-p4 for display
            js_data = js_data.rename(columns={
                "p1": "a (skewness)",
                "p2": "b (tail weight)",
                "p3": "loc",
                "p4": "scale",
            })
            params_js = ["a (skewness)", "b (tail weight)", "loc", "scale"]
            show_events_js = st.checkbox("Show climate events", value=True, key="js_events")

            fig_js = make_subplots(
                rows=2, cols=2,
                subplot_titles=params_js,
                shared_xaxes=True,
                vertical_spacing=0.12,
                horizontal_spacing=0.10,
            )
            positions_js = [(1, 1), (1, 2), (2, 1), (2, 2)]
            for param, (row, col) in zip(params_js, positions_js):
                fig_js.add_trace(go.Scatter(
                    x=js_data["window_mid_year"],
                    y=js_data[param],
                    mode="lines+markers",
                    line=dict(color="#0072B2", width=2),
                    marker=dict(size=5),
                    name=param, showlegend=False,
                    hovertemplate=(
                        f"<b>{param}</b><br>"
                        "Year: %{x}<br>"
                        "Value: %{y:.4f}<extra></extra>"
                    ),
                ), row=row, col=col)

                if show_events_js:
                    for yr, label, etype in EVENTS:
                        fig_js.add_vline(
                            x=yr, line_width=1, line_dash="dot",
                            line_color=EVENT_COLOURS[etype],
                            row=row, col=col,
                        )

                if param == "b (tail weight)":
                    fig_js.update_yaxes(autorange="reversed", row=row, col=col)

            fig_js.update_xaxes(title_text="Window mid-year", row=2, col=1)
            fig_js.update_xaxes(title_text="Window mid-year", row=2, col=2)
            _plotly_layout(
                fig_js,
                title=f"Johnson SU parameters — {station_opts[selected_ref]}",
                height=520,
            )
            st.plotly_chart(fig_js, use_container_width=True)

    # ── Tab: Trend Summary ─────────────────────────────────────────────────────
    with tab_mk:
        station_traj = traj[traj["station_reference"] == selected_ref].copy()
        if station_traj.empty:
            st.info("No trajectory data for this station.")
        else:
            station_traj["param_label"] = station_traj["parameter"].map({
                "scale": "Scale (variance proxy)",
                "a":     "a (skewness)",
                "b":     "b (tail weight)",
                "loc":   "Location",
            })
            station_traj["sig"] = station_traj["pvalue"].apply(
                lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            cols_show = ["param_label", "tau", "pvalue", "sig", "trend", "start_value", "end_value"]
            df_traj   = station_traj[cols_show].copy()
            df_traj.columns = ["Parameter", "Kendall τ", "p-value", "Sig.", "Trend", "Start", "End"]
            st.dataframe(df_traj.round(4), use_container_width=True, hide_index=True)

            # Compare to national trend direction
            nat_trt = load_trend_tests()
            st.markdown("**National trend direction (for comparison)**")
            nat_show = nat_trt[["parameter", "trend", "tau", "change_pct"]].copy()
            nat_show["parameter"] = nat_show["parameter"].map({
                "scale": "Scale", "a": "a", "b": "b", "loc": "Location"
            })
            nat_show.columns = ["Parameter", "National trend", "National τ", "National change %"]
            st.dataframe(nat_show.round(4), use_container_width=True, hide_index=True)


# ── Page: Station Winners Map ──────────────────────────────────────────────────

@st.cache_data
def load_station_winners(criterion: str) -> pd.DataFrame:
    """
    Join per-station best-fit distribution with cohort metadata.
    criterion: 'AIC' | 'BIC' | 'KS'
    """
    rank_col = {"AIC": "rank_aic", "BIC": "rank_bic", "KS": "rank_ks"}[criterion]
    sg      = load_selection_global()
    cohort  = load_cohort()
    winners = sg[sg[rank_col] == 1][
        ["station_reference", "distribution", "aic", "bic", "ks_statistic",
         "delta_aic", "akaike_weight"]
    ].copy()
    merged = winners.merge(
        cohort[["station_reference", "label", "river_name", "lat", "long",
                "first_year", "last_year", "year_coverage_frac"]],
        on="station_reference",
    )
    merged["dist_label"] = merged["distribution"].map(DIST_LABELS)
    return merged


def page_station_winners() -> None:
    st.header("Per-Station Best-Fit Distribution")
    st.markdown(
        "Which distribution fits best at each of the 257 gauging stations, "
        "based on the full daily record (~16,000 observations per station)."
    )

    criterion = st.radio(
        "Ranking criterion", ["AIC", "BIC", "KS"],
        horizontal=True, key="map_criterion",
    )

    df = load_station_winners(criterion)

    # Join RBD if available
    try:
        rbd_col = load_rbd_cohort()[["station_reference", "rbd_name"]]
        df = df.merge(rbd_col, on="station_reference", how="left")
        has_rbd = True
    except Exception:
        has_rbd = False

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            all_dists = [DIST_LABELS[d] for d in PALETTE if d in df["distribution"].unique()]
            sel_dists = st.multiselect(
                "Distribution", options=all_dists, default=all_dists,
                key="map_f_dist",
            )

        with col2:
            if has_rbd:
                all_rbds = sorted(df["rbd_name"].dropna().unique().tolist())
                sel_rbds = st.multiselect(
                    "River Basin District", options=all_rbds, default=all_rbds,
                    key="map_f_rbd",
                )
            else:
                sel_rbds = None

        with col3:
            ks_max = float(df["ks_statistic"].max())
            ks_thresh = st.slider(
                "Max KS statistic", min_value=0.01, max_value=round(ks_max, 2),
                value=round(ks_max, 2), step=0.01, key="map_f_ks",
            )

    # Apply filters
    mask = (
        df["dist_label"].isin(sel_dists) &
        (df["ks_statistic"] <= ks_thresh)
    )
    if has_rbd and sel_rbds is not None:
        mask &= df["rbd_name"].isin(sel_rbds)
    df_filtered = df[mask].copy()

    n_total    = len(df)
    n_filtered = len(df_filtered)
    st.caption(f"Showing {n_filtered} of {n_total} stations")

    # ── Win-count summary bar ──────────────────────────────────────────────────
    counts = (
        df_filtered.groupby("distribution").size()
          .rename("n_stations").reset_index()
          .sort_values("n_stations", ascending=False)
    )
    counts["colour"]     = counts["distribution"].map(PALETTE)
    counts["dist_label"] = counts["distribution"].map(DIST_LABELS)
    counts["pct"]        = 100 * counts["n_stations"] / max(n_filtered, 1)

    fig_bar = go.Figure()
    for _, row in counts.iterrows():
        fig_bar.add_trace(go.Bar(
            x=[row["dist_label"]],
            y=[row["n_stations"]],
            marker_color=row["colour"],
            text=f"{row['pct']:.1f}%",
            textposition="outside",
            hovertemplate=(
                f"<b>{row['dist_label']}</b><br>"
                f"Stations: {row['n_stations']}<br>"
                f"Share: {row['pct']:.1f}%<extra></extra>"
            ),
            showlegend=False,
        ))
    _plotly_layout(
        fig_bar,
        title=f"Stations ranking first by {criterion}  ({n_filtered} shown)",
        yaxis_title="Number of stations",
        height=300,
        bargap=0.25,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Map ────────────────────────────────────────────────────────────────────
    cat_order = [DIST_LABELS[d] for d in PALETTE if d in df_filtered["distribution"].unique()]

    hover_data = {
        "river_name":        True,
        "station_reference": True,
        "dist_label":        True,
        "ks_statistic":      ":.4f",
        "akaike_weight":     ":.4f",
        "lat":               False,
        "long":              False,
    }
    if has_rbd:
        hover_data["rbd_name"] = True

    fig_map = px.scatter_mapbox(
        df_filtered,
        lat="lat", lon="long",
        color="dist_label",
        color_discrete_map={DIST_LABELS[d]: PALETTE[d] for d in PALETTE},
        category_orders={"dist_label": cat_order},
        hover_name="label",
        hover_data=hover_data,
        labels={
            "dist_label":        "Best-fit distribution",
            "ks_statistic":      "KS statistic",
            "akaike_weight":     "Akaike weight",
            "river_name":        "River",
            "station_reference": "Station ref.",
            "rbd_name":          "River Basin District",
        },
        zoom=5.2,
        center={"lat": 54.0, "lon": -2.0},
        mapbox_style="open-street-map",
        size_max=10,
    )
    fig_map.update_traces(marker_size=10, marker_opacity=0.85)
    fig_map.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Best-fit distribution by station ({criterion})",
        legend_title_text="Best-fit distribution",
        mapbox=dict(bounds=dict(west=-9, east=3, south=49, north=62)),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Sortable table ─────────────────────────────────────────────────────────
    with st.expander("Station table"):
        show_cols = ["station_reference", "label", "river_name",
                     "dist_label", "ks_statistic", "akaike_weight",
                     "first_year", "last_year", "year_coverage_frac"]
        rename = {
            "station_reference": "Ref.",
            "label":             "Station",
            "river_name":        "River",
            "dist_label":        "Best fit",
            "ks_statistic":      "KS",
            "akaike_weight":     "Akaike wt.",
            "first_year":        "From",
            "last_year":         "To",
            "year_coverage_frac":"Coverage",
        }
        if has_rbd:
            show_cols.insert(3, "rbd_name")
            rename["rbd_name"] = "RBD"

        st.dataframe(
            df_filtered[show_cols].rename(columns=rename)
                       .sort_values("KS").reset_index(drop=True).round(4),
            use_container_width=True, hide_index=True,
        )


# ── Page: River Basin Districts ───────────────────────────────────────────────

SUBGROUPS_DIR = ROOT / "outputs" / "subgroups"

@st.cache_data
def load_rbd_summary(mode: str = "daily") -> pd.DataFrame:
    return pd.read_parquet(SUBGROUPS_DIR / mode / "rbd_selection_summary.parquet")

@st.cache_data
def load_rbd_cohort() -> pd.DataFrame:
    return pd.read_parquet(SUBGROUPS_DIR / "cohort_rbd.parquet")

@st.cache_data
def load_rbd_boundaries() -> dict:
    import json
    with open(SUBGROUPS_DIR / "rbd_boundaries.geojson") as f:
        return json.load(f)

@st.cache_data
def load_rbd_js_params(mode: str = "daily") -> pd.DataFrame:
    return pd.read_parquet(SUBGROUPS_DIR / mode / "rbd_js_params.parquet")


def _rbd_mode_charts(mode: str, summary: pd.DataFrame, js_params: pd.DataFrame,
                     geojson: dict, rbd_order: list, key_sfx: str) -> None:
    """Render choropleth, bar chart, JS evolution, and summary table for one mode."""
    js_summary = summary[summary["distribution"] == "johnson_su"].copy()
    js_summary = js_summary.set_index("rbd_name").reindex(rbd_order).reset_index()

    # Choropleth
    mode_label = {
        "daily":     "daily global fits (~16 k obs/station)",
        "monthly":   "monthly pooled fits (~514 obs/station)",
        "log_daily": "log-transformed daily fits",
    }.get(mode, mode)
    fig_map = px.choropleth_map(
        js_summary,
        geojson=geojson,
        locations="rbd_name",
        featureidkey="properties.rbd_name",
        color="pct_win_aic",
        color_continuous_scale="Blues",
        range_color=(0, 100),
        hover_name="rbd_name",
        hover_data={
            "pct_win_aic": ":.1f",
            "n_stations":  True,
            "median_ks":   ":.4f",
            "mean_akaike": ":.4f",
            "rbd_name":    False,
        },
        labels={
            "pct_win_aic":  "Johnson SU AIC wins %",
            "n_stations":   "Stations",
            "median_ks":    "Median KS",
            "mean_akaike":  "Mean Akaike wt.",
        },
        map_style="open-street-map",
        opacity=0.65,
    )
    fig_map.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Johnson SU AIC win rate by RBD — {mode_label}",
        coloraxis_colorbar_title="JS wins %",
        map=dict(
            center={"lat": 55.5, "lon": -3.5},
            zoom=4.8,
            bounds=dict(west=-9, east=3, south=49, north=62),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Bar chart: distributions by RBD
    st.subheader("Win rates by distribution and RBD")
    sel_dists = st.multiselect(
        "Distributions",
        options=list(DIST_LABELS.keys()),
        default=["johnson_su", "lognormal", "gamma"],
        format_func=lambda x: DIST_LABELS[x],
        key=f"rbd_dists_{key_sfx}",
    )
    sub = summary[summary["distribution"].isin(sel_dists)].copy()
    sub["dist_label"] = sub["distribution"].map(DIST_LABELS)
    fig_bar = px.bar(
        sub,
        x="rbd_name",
        y="pct_win_aic",
        color="dist_label",
        barmode="group",
        color_discrete_map={DIST_LABELS[d]: PALETTE[d] for d in PALETTE},
        category_orders={"rbd_name": rbd_order,
                         "dist_label": [DIST_LABELS[d] for d in sel_dists]},
        labels={"pct_win_aic": "AIC win rate (%)", "rbd_name": "River Basin District",
                "dist_label": "Distribution"},
        hover_data={"n_stations": True, "median_ks": ":.4f"},
    )
    _plotly_layout(fig_bar, height=380)
    st.plotly_chart(fig_bar, use_container_width=True)

    # JS parameter chart — rolling time series (daily/monthly) or global bar (log_daily)
    param_labels_map = {
        "scale_median": "Scale (variance proxy)",
        "b_median":     "b (tail weight)",
        "a_median":     "a (skewness)",
        "loc_median":   "Location",
    }
    is_rolling = "window_mid_year" in js_params.columns

    expander_title = (
        "Johnson SU parameter evolution by RBD"
        if is_rolling else
        "Johnson SU parameters by RBD (global fit)"
    )
    with st.expander(expander_title):
        param = st.selectbox(
            "Parameter",
            list(param_labels_map.keys()),
            format_func=lambda x: param_labels_map[x],
            key=f"rbd_param_{key_sfx}",
        )
        if is_rolling:
            fig_evo = go.Figure()
            for rbd in rbd_order:
                d = js_params[js_params["rbd_name"] == rbd].sort_values("window_mid_year")
                if d.empty:
                    continue
                fig_evo.add_trace(go.Scatter(
                    x=d["window_mid_year"], y=d[param],
                    mode="lines", name=rbd, line=dict(width=2),
                    hovertemplate=f"<b>{rbd}</b><br>Year: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
                ))
            _plotly_layout(fig_evo,
                           title=f"{param_labels_map[param]} by RBD — 10-year rolling medians",
                           xaxis_title="Window mid-year",
                           height=380)
            st.plotly_chart(fig_evo, use_container_width=True)
        else:
            # Global fit: single bar per RBD
            params_ord = js_params.set_index("rbd_name").reindex(rbd_order).reset_index()
            fig_bar_p = go.Figure(go.Bar(
                x=params_ord["rbd_name"],
                y=params_ord[param],
                marker_color="#009E73",
                hovertemplate="<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>",
            ))
            _plotly_layout(fig_bar_p,
                           title=f"{param_labels_map[param]} by RBD — global log-daily fit",
                           yaxis_title=param_labels_map[param],
                           xaxis_title="River Basin District",
                           height=360,
                           showlegend=False)
            st.plotly_chart(fig_bar_p, use_container_width=True)

    # Summary table
    with st.expander("Full RBD summary table"):
        st.dataframe(
            js_summary[["rbd_name", "n_stations", "pct_win_aic", "median_ks", "mean_akaike"]]
            .rename(columns={
                "rbd_name":    "RBD",
                "n_stations":  "Stations",
                "pct_win_aic": "JS AIC wins %",
                "median_ks":   "Median KS",
                "mean_akaike": "Mean Akaike wt.",
            }).reset_index(drop=True).round(4),
            use_container_width=True, hide_index=True,
        )


def page_rbd() -> None:
    st.header("River Basin Districts")
    st.markdown(
        "257 stations split across 9 River Basin Districts (EA WFD boundaries). "
        "Compare distribution preferences across raw daily, monthly, and log-transformed daily fits."
    )

    geojson          = load_rbd_boundaries()
    summary_daily    = load_rbd_summary("daily")
    summary_monthly  = load_rbd_summary("monthly")
    summary_log      = load_rbd_summary("log_daily")
    params_daily     = load_rbd_js_params("daily")
    params_monthly   = load_rbd_js_params("monthly")
    params_log       = load_rbd_js_params("log_daily")

    # RBD order fixed by daily JS win rate (descending)
    js_daily = summary_daily[summary_daily["distribution"] == "johnson_su"].copy()
    rbd_order = js_daily.sort_values("pct_win_aic", ascending=False)["rbd_name"].tolist()

    tab_daily, tab_monthly, tab_log, tab_compare = st.tabs(
        ["Daily Fits", "Monthly Fits", "Log-transformed", "Comparison"]
    )

    with tab_daily:
        _rbd_mode_charts("daily", summary_daily, params_daily, geojson, rbd_order, "d")

    with tab_monthly:
        st.info(
            "Monthly pooled fits use ~514 monthly mean observations per station — far fewer "
            "extremes than 16,000 daily values, so heavy-tailed distributions win less often."
        )
        _rbd_mode_charts("monthly", summary_monthly, params_monthly, geojson, rbd_order, "m")

    with tab_log:
        st.info(
            "Log-transformed fits apply log(mean daily level) before fitting. "
            "This compresses the right tail, making the data closer to symmetric. "
            "Johnson SU still leads but Weibull becomes a strong competitor (~31% nationally)."
        )
        _rbd_mode_charts("log_daily", summary_log, params_log, geojson, rbd_order, "l")

    with tab_compare:
        st.markdown(
            "Johnson SU AIC win rates across all three fitting approaches. "
            "Monthly averaging and log-transformation both suppress tail behaviour, "
            "reducing Johnson SU's dominance — but it remains competitive in both cases."
        )

        js_monthly = summary_monthly[summary_monthly["distribution"] == "johnson_su"].copy()
        js_log     = summary_log[summary_log["distribution"] == "johnson_su"].copy()

        compare = (
            js_daily[["rbd_name", "n_stations", "pct_win_aic", "median_ks"]]
            .rename(columns={"pct_win_aic": "daily_pct", "median_ks": "daily_ks"})
            .merge(
                js_monthly[["rbd_name", "pct_win_aic", "median_ks"]]
                .rename(columns={"pct_win_aic": "monthly_pct", "median_ks": "monthly_ks"}),
                on="rbd_name", how="outer",
            )
            .merge(
                js_log[["rbd_name", "pct_win_aic", "median_ks"]]
                .rename(columns={"pct_win_aic": "log_pct", "median_ks": "log_ks"}),
                on="rbd_name", how="outer",
            )
        )
        compare = compare.set_index("rbd_name").reindex(rbd_order).reset_index()

        # Win-rate comparison
        fig_cmp = go.Figure()
        for label, col, colour in [
            ("Raw daily",       "daily_pct",   "#0072B2"),
            ("Log-transformed", "log_pct",     "#009E73"),
            ("Monthly",         "monthly_pct", "#E69F00"),
        ]:
            fig_cmp.add_trace(go.Bar(
                x=compare["rbd_name"],
                y=compare[col],
                name=label,
                marker_color=colour,
                hovertemplate=f"<b>%{{x}}</b><br>{label} JS wins: %{{y:.1f}}%<extra></extra>",
            ))
        _plotly_layout(
            fig_cmp,
            title="Johnson SU AIC win rate by RBD — raw daily vs log-transformed vs monthly",
            yaxis_title="% of stations where Johnson SU ranks first (AIC)",
            xaxis_title="River Basin District",
            barmode="group",
            height=440,
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # KS comparison
        fig_ks = go.Figure()
        for label, col, colour in [
            ("Raw daily",       "daily_ks",   "#0072B2"),
            ("Log-transformed", "log_ks",     "#009E73"),
            ("Monthly",         "monthly_ks", "#E69F00"),
        ]:
            fig_ks.add_trace(go.Bar(
                x=compare["rbd_name"],
                y=compare[col],
                name=label,
                marker_color=colour,
                hovertemplate=f"<b>%{{x}}</b><br>{label} median KS: %{{y:.4f}}<extra></extra>",
            ))
        _plotly_layout(
            fig_ks,
            title="Johnson SU median KS statistic — raw daily vs log-transformed vs monthly",
            yaxis_title="Median KS statistic (lower = better fit)",
            xaxis_title="River Basin District",
            barmode="group",
            height=380,
        )
        st.plotly_chart(fig_ks, use_container_width=True)

        # Summary table
        display = compare[["rbd_name", "n_stations",
                            "daily_pct", "log_pct", "monthly_pct",
                            "daily_ks",  "log_ks",  "monthly_ks"]].copy()
        display.columns = ["RBD", "Stations",
                           "Daily JS %", "Log-daily JS %", "Monthly JS %",
                           "Daily KS",   "Log-daily KS",   "Monthly KS"]
        st.dataframe(display.round(2), use_container_width=True, hide_index=True)


# ── App shell ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UK River Distributions",
    page_icon=":droplet:",
    layout="wide",
)

st.title("UK River Level Distributions — Phase 1")
st.caption("257-station cohort · daily fits · 1980–2024 · Johnson SU preferred model")

page = st.sidebar.radio(
    "Navigate",
    ["Station Winners", "River Basin Districts",
     "Model Selection", "Parameter Evolution", "Station Explorer"],
)

if page == "Station Winners":
    page_station_winners()
elif page == "River Basin Districts":
    page_rbd()
elif page == "Model Selection":
    page_model_selection()
elif page == "Parameter Evolution":
    page_parameter_evolution()
elif page == "Station Explorer":
    page_station_explorer()
