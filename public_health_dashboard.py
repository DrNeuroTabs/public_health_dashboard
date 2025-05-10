import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters
import requests
import gzip
from io import BytesIO
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pycountry

# --------------------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas numpy plotly prophet ruptures requests statsmodels scikit-learn pycountry
#
# Dashboard combining:
#  • Mortality joinpoints, segmented fits, APC, forecasts
#  • Exploration of health factors via regression
#  • Data‐driven clustering + map
#
# Uses a single generic loader for any Eurostat SDMX‐TSV dataset.
# --------------------------------------------------------------------------

EU_CODES = [
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE",
    "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"
]

SEX_NAME_MAP = {"T": "Total", "M": "Male", "F": "Female"}
REV_SEX_NAME = {v: k for k, v in SEX_NAME_MAP.items()}

# ICD‐10 cause mapping (partial; extend as needed)
CAUSE_NAME_MAP = { ... }  # same as before
REV_CAUSE_NAME_MAP = {v:k for k,v in CAUSE_NAME_MAP.items()}

# ISO α2 → full country name
COUNTRY_NAME_MAP = {c.alpha_2: c.name for c in pycountry.countries}
COUNTRY_NAME_MAP.update({
    "FX":"France (Metropolitan)",
    "EU":"European Union","Europe":"Europe"
})
REV_COUNTRY_NAME_MAP = {v:k for k,v in COUNTRY_NAME_MAP.items()}

# Health‐factor dataset IDs to explore
FACTOR_IDS = { ... }  # same as before

@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/"
        f"sdmx/2.1/data/{dataset_id}?format=TSV&compressed=true"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep="\t", low_memory=False)

    first = raw.columns[0]
    dims = first.split("\\")[0].split(",")
    raw = raw.rename(columns={first: "series_keys"})
    keys = raw["series_keys"].str.split(",", expand=True)
    keys.columns = dims

    df = pd.concat([keys, raw.drop(columns=["series_keys"])], axis=1)
    years = [c for c in df.columns if c not in dims]
    long = df.melt(
        id_vars=dims, value_vars=years,
        var_name="Year", value_name="raw_rate"
    )
    long["Year"] = long["Year"].str.strip().astype(int)
    long["Rate"] = pd.to_numeric(
        long["raw_rate"].str.strip().replace(":", np.nan),
        errors="coerce"
    )

    units = long["unit"].unique()
    unit_val = "RT" if "RT" in units else ("NR" if "NR" in units else None)
    mask = pd.Series(True, index=long.index)
    if unit_val:
        mask &= (long["unit"] == unit_val)
    mask &= (long.get("freq") == "A")
    if "age" in dims:
        mask &= (long["age"] == "TOTAL")
    if "sex" in dims:
        mask &= (long["sex"] == "T")
    if "resid" in dims:
        mask &= (long["resid"] == "TOT_IN")
    sub = long[mask].copy()

    rename = {"geo":"Region", "sex":"Sex"}
    others = [d for d in dims if d not in ("geo","sex","freq","unit","age","resid")]
    if len(others) == 1:
        rename[others[0]] = "Category"
    out = sub.rename(columns=rename)

    return out[["Region","Year","Category","Sex","Rate"]]

@st.cache_data
def load_data() -> pd.DataFrame:
    def ld(id_):
        df = load_eurostat_series(id_).rename(columns={"Region":"Country", "Category":"Cause"})
        return df.dropna(subset=["Rate"])
    hist = ld("hlth_cd_asdr")
    mod  = ld("hlth_cd_asdr2")
    mod  = mod[mod["Country"].str.fullmatch(r"[A-Z]{2}")]
    df   = pd.concat([hist, mod], ignore_index=True).sort_values(
        ["Country","Cause","Sex","Year"]
    )

    df_eu = (
        df[df["Country"].isin(EU_CODES)]
        .groupby(["Year","Cause","Sex"], as_index=False)["Rate"]
        .mean()
    ); df_eu["Country"] = "EU"
    df_eur = (
        df.groupby(["Year","Cause","Sex"], as_index=False)["Rate"]
        .mean()
    ); df_eur["Country"] = "Europe"

    return pd.concat([df, df_eu, df_eur], ignore_index=True)

# … (other helper functions remain unchanged) …

def main():
    st.set_page_config(layout="wide", page_title="Public Health Dashboard")
    st.title("Standardised Mortality Rates & Health Factors")

    df = load_data()
    df["CountryFull"] = df["Country"].map(COUNTRY_NAME_MAP)
    df["CauseFull"]   = df["Cause"].map(CAUSE_NAME_MAP)
    df["SexFull"]     = df["Sex"].map(SEX_NAME_MAP)

    # Sidebar selectors
    countries = sorted(df["CountryFull"].dropna().unique())
    country_full = st.sidebar.selectbox("Country", countries)
    country_code = REV_COUNTRY_NAME_MAP.get(country_full, country_full)

    causes = sorted(df[df["Country"]==country_code]["CauseFull"].dropna().unique())
    cause_full = st.sidebar.selectbox("Cause of Death", causes)
    cause_code = REV_CAUSE_NAME_MAP.get(cause_full, cause_full)

    sex_sel = st.sidebar.multiselect("Sex", ["Total","Male","Female"], default=["Total"])
    sex_codes = [REV_SEX_NAME[s] for s in sex_sel]

    yrs = sorted(df["Year"].unique())
    y0, y1 = yrs[0], yrs[-1]
    year_range = st.sidebar.slider("Historical Years", y0, y1, (y0, y1))

    forecast_years = st.sidebar.slider("Forecast Horizon (yrs)", 1, 30, 10)
    method = st.sidebar.selectbox("Forecast Method", ["Prophet","ARIMA","ETS","Ensemble"])

    # Filtered mortality subset
    df_f = df[
        (df["Country"]==country_code) &
        (df["Cause"]==cause_code) &
        (df["Sex"].isin(sex_codes)) &
        (df["Year"].between(*year_range))
    ]
    st.header(f"{cause_full} in {country_full} ({year_range[0]}–{year_range[1]})")
    if df_f.empty:
        st.warning("No data for selected filters.")
    else:
        # Joinpoints, segmented fits, APC, forecasts… (unchanged)
        ...

    # Health‐factor regression
    st.markdown("---")
    st.header("Health Factors – Exploratory Regression")
    factors = st.multiselect("Select health factors", list(FACTOR_IDS.keys()))
    if factors:
        model_df = df_f[["Country","Year","Sex","Rate"]].rename(columns={"Rate":"Mortality"})
        skipped = []
        for name in factors:
            ds = FACTOR_IDS[name]
            try:
                dfact = load_eurostat_series(ds).rename(columns={"Region":"Country"})
            except Exception:
                skipped.append(name)
                continue
            dfact = dfact[
                (dfact["Country"]==country_code) &
                (dfact["Sex"].isin(sex_codes)) &
                (dfact["Year"].between(*year_range))
            ][["Country","Year","Sex","Rate"]].rename(columns={"Rate":name})
            model_df = model_df.merge(dfact, on=["Country","Year","Sex"], how="left")
        if skipped:
            st.warning(f"Skipped unavailable factors: {', '.join(skipped)}")
        available = [f for f in factors if f not in skipped]
        if available:
            model_df = model_df.dropna(subset=available+\
