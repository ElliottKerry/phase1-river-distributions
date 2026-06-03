# Phase 1 — UK River Level Distributions

Statistical analysis of UK river level data from the Environment Agency (1980–2024).

**257-station cohort · daily fits · Johnson SU preferred model**

## Dashboard

Live app: *(add your Streamlit Cloud URL here)*

```
streamlit run src/app.py
```

## Pages

| Page | Description |
|------|-------------|
| Station Winners | UK map — which distribution fits best at each gauging station |
| Model Selection | Win rates (AIC/BIC/KS) for daily global and rolling fits |
| Parameter Evolution | Johnson SU parameter trends 1980–2024 |
| Station Explorer | Per-station distribution history and parameter trajectory |

## Pipeline

```
src/ingest.py        # Download EA 15-min water level data
src/preprocess.py    # Clean → daily averages → monthly series
src/cohort.py        # Select stable 1980–2024 station cohort
src/fit_series.py    # Fit 9 distributions (daily global + rolling)
src/selection.py     # AIC/BIC/KS model selection + Akaike weights
src/temporal.py      # Johnson SU parameter temporal trends
src/visualise.py     # Publication-ready matplotlib figures
src/app.py           # Streamlit dashboard
```

## Data

Raw EA data and the DuckDB database are not included in this repository
(~18 GB). The pre-computed outputs needed to run the dashboard are committed
under `outputs/` and `data/processed/cohort.parquet`.
