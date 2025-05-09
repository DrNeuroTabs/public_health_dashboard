import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters
import requests
import gzip
from io import BytesIO
import statsmodels.api as sm

# Updated imports for mapping
import pycountry
import icd10  # icd10-cm package :contentReference[oaicite:0]{index=0}

# --------------------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas numpy plotly prophet ruptures requests statsmodels pycountry icd10-cm
#
# This dashboard stitches together national standardised‐death‐rate data:
#  • hlth_cd_asdr   (1994–2010 national rates, unit="RT")
#  • hlth_cd_asdr2  (2011–present NUTS2 rates, unit="NR"), filtering
#                   only the country‐level codes (length==2)
# It then appends “EU” and “Europe” aggregates, maps codes → names,
# and runs joinpoint analysis, APC calculations, and forecasting.
# --------------------------------------------------------------------------

EU_CODES = [
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE",
    "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"
]

@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        f"{dataset_id}?format=TSV&compressed=true"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep="\t", low_memory=False)

    key_col = raw.columns[0]
    dims = key_col.split("\\")[0].split(",")
    raw = raw.rename(columns={key_col: "series_keys"})
    keys = raw["series_keys"].str.split(",", expand=True)
    keys.columns = dims
    df = pd.concat([keys, raw.drop(columns=["series_keys"])], axis=1)

    year_cols = [c for c in df.columns if c not in dims]
    long = df.melt(id_vars=dims, value_vars=year_cols,
                   var_name="Year", value_name="raw_rate")

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
    mask &= (long.get("sex") == "T")
    mask &= (long.get("age") == "TOTAL")
    if "resid" in long.columns:
        mask &= (long["resid"] == "TOT_IN")

    sub = long[mask].copy().rename(columns={"icd10": "Cause", "geo": "Region"})
    return sub[["Region", "Year", "Cause", "Rate"]]

@st.cache_data
def load_historical_rates() -> pd.DataFrame:
    df = load_eurostat_series("hlth_cd_asdr")
    df = df.rename(columns={"Region": "Country"})
    return df.dropna(subset=["Rate"]).sort_values(["Country","Cause","Year"])

@st.cache_data
def load_modern_rates() -> pd.DataFrame:
    df = load_eurostat_series("hlth_cd_asdr2")
    df["Region"] = df["Region"].astype(str)
    df_ctry = df[df["Region"].str.fullmatch(r"[A-Z]{2}")].copy()
    df_ctry = df_ctry.rename(columns={"Region": "Country"})
    return df_ctry.dropna(subset=["Rate"]).sort_values(["Country","Cause","Year"])

@st.cache_data
def load_data() -> pd.DataFrame:
    hist = load_historical_rates()
    mod  = load_modern_rates()
    df   = pd.concat([hist, mod], ignore_index=True)
    df   = df.dropna(subset=["Rate"]).sort_values(["Country","Cause","Year"])

    df_eu = (
        df[df["Country"].isin(EU_CODES)]
        .groupby(["Year","Cause"], as_index=False)["Rate"].mean()
    )
    df_eu["Country"] = "EU"

    df_eur = (
        df.groupby(["Year","Cause"], as_index=False)["Rate"].mean()
    )
    df_eur["Country"] = "Europe"

    return pd.concat([df, df_eu, df_eur], ignore_index=True)

def map_country_name(code: str) -> str:
    try:
        return pycountry.countries.get(alpha_2=code).name
    except:
        return code

def map_icd_description(icd_code: str) -> str:
    parts = icd_code.split("_")
    descs = []
    for part in parts:
        node = icd10.find(part)  # use find(), not constructor
        if node and node.description:
            descs.append(node.description)
        else:
            descs.append(part)
    return " / ".join(descs)

def detect_change_points(ts: pd.Series, pen: float = 3) -> list:
    clean = ts.dropna()
    if len(clean) < 2:
        return []
    algo = rpt.Pelt(model="l2").fit(clean.values)
    try:
        return algo.predict(pen=pen)
    except BadSegmentationParameters:
        return []

def compute_joinpoints_and_apc(df_sub: pd.DataFrame) -> pd.DataFrame:
    df_s = df_sub.sort_values("Year")
    yrs, vals = df_s["Year"].values, df_s["Rate"].values
    bkps = detect_change_points(df_s["Rate"])[:-1]
    segs = np.split(np.arange(len(yrs)), bkps) if bkps else [np.arange(len(yrs))]
    recs = []
    for seg in segs:
        sy, ey = int(yrs[seg].min()), int(yrs[seg].max())
        seg_vals = vals[seg]
        if len(seg_vals) < 2 or np.all(np.isnan(seg_vals)):
            recs.append({"start_year":sy,"end_year":ey,"slope":np.nan,"APC_pct":np.nan})
        else:
            slope = sm.OLS(seg_vals, sm.add_constant(yrs[seg])).fit().params[1]
            apc   = (slope / np.nanmean(seg_vals)) * 100
            recs.append({"start_year":sy,"end_year":ey,"slope":slope,"APC_pct":apc})
    return pd.DataFrame(recs)

def plot_joinpoints(df: pd.DataFrame, country: str, cause: str) -> None:
    sub = df[(df["Country"]==country)&(df["Cause"]==cause)].sort_values("Year")
    cps = detect_change_points(sub["Rate"])
    fig = px.line(sub, x="Year", y="Rate", title=f"{cause} Mortality Rate in {country}")
    for cp in cps:
        if 0 < cp < len(sub):
            fig.add_vline(x=sub.iloc[cp]["Year"], line_dash="dash")
    st.plotly_chart(fig)

def forecast_mortality(df_sub: pd.DataFrame, periods: int = 10) -> None:
    dfp = df_sub[["Year","Rate"]].rename(columns={"Year":"ds","Rate":"y"})
    dfp["ds"] = pd.to_datetime(dfp["ds"].astype(str), format="%Y")
    m = Prophet(yearly_seasonality=False,daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods, freq="Y")
    fc = m.predict(future)
    st.plotly_chart(px.line(fc, x="ds", y="yhat", title="Forecasted Mortality Rate"))

def main():
    st.set_page_config(layout="wide", page_title="Mortality Rates 1994–Present")
    st.title("Standardised Mortality Rates (1994–Present) by Country")

    df = load_data()
    # map codes to names
    df["CountryName"] = df["Country"].map(map_country_name)
    df["CauseName"]   = df["Cause"].map(map_icd_description)

    countries = sorted(df["CountryName"].unique())
    country    = st.sidebar.selectbox("Country", countries)
    causes     = sorted(df[df["CountryName"]==country]["CauseName"].unique())
    cause      = st.sidebar.selectbox("Cause of Death", causes)

    yrs        = sorted(df["Year"].unique())
    y0, y1     = int(yrs[0]), int(yrs[-1])
    year_range = st.sidebar.slider("Year Range", y0, y1, (y0, y1))

    df_f = df[
        (df["CountryName"]==country)&
        (df["CauseName"]  ==cause)   &
        (df["Year"].between(*year_range))
    ]

    st.header(f"{cause} Mortality Rate in {country} ({year_range[0]}–{year_range[1]})")
    if df_f.empty:
        st.warning("No data for selected filters.")
    else:
        plot_joinpoints(df_f, country, cause)
        st.markdown("### Joinpoint & Annual Percent Change (APC)")
        st.dataframe(compute_joinpoints_and_apc(df_f), use_container_width=True)
        st.markdown("### Forecast Next 10 Years")
        forecast_mortality(df_f)

    st.markdown("---")
    st.info("Data combined from national (1994–2010) and NUTS2 (2011–present) series; codes replaced with full names.")

if __name__ == "__main__":
    main()
