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

# --------------------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas numpy plotly prophet ruptures requests statsmodels
#
# This dashboard stitches together national standardised‐death‐rate data:
#  • hlth_cd_asdr   (1994–2010 national rates, unit="RT")
#  • hlth_cd_asdr2  (2011–present NUTS 2 rates, unit="NR"), but filtered
#                   for the country-level entries (geo codes length 2)
# It then runs joinpoint analysis, APC calculations, and forecasting.
# --------------------------------------------------------------------------

@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    """Generic loader for any SDMX-TSV Eurostat series, dynamic on unit."""
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        f"{dataset_id}?format=TSV&compressed=true"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep="\t", low_memory=False)

    # split the composite key into dimension columns
    key_col = raw.columns[0]
    dims = key_col.split("\\")[0].split(",")
    raw = raw.rename(columns={key_col: "series_keys"})
    keys = raw["series_keys"].str.split(",", expand=True)
    keys.columns = dims
    df = pd.concat([keys, raw.drop(columns=["series_keys"])], axis=1)

    # melt the year-columns
    year_cols = [c for c in df.columns if c not in dims]
    long = df.melt(id_vars=dims, value_vars=year_cols,
                   var_name="Year", value_name="raw_rate")

    # clean and convert
    long["Year"] = long["Year"].str.strip().astype(int)
    long["Rate"] = pd.to_numeric(
        long["raw_rate"].str.strip().replace(":", np.nan),
        errors="coerce"
    )

    # dynamic unit filter: picks "RT" for historical and "NR" for modern
    units = long["unit"].unique()
    if "RT" in units:
        unit_val = "RT"
    elif "NR" in units:
        unit_val = "NR"
    else:
        unit_val = None

    mask = pd.Series(True, index=long.index)
    if unit_val:
        mask &= (long["unit"] == unit_val)

    # common filters: annual, total population
    mask &= (long.get("freq") == "A")
    mask &= (long.get("sex") == "T")
    mask &= (long.get("age") == "TOTAL")
    if "resid" in long.columns:
        mask &= (long["resid"] == "TOT_IN")

    sub = long[mask].copy()
    # rename for clarity
    sub = sub.rename(columns={"icd10": "Cause", "geo": "Region"})
    return sub[["Region", "Year", "Cause", "Rate"]]

@st.cache_data
def load_historical_rates() -> pd.DataFrame:
    """Load 1994–2010 national rates from hlth_cd_asdr (unit=RT)."""
    df = load_eurostat_series("hlth_cd_asdr")
    # in this dataset Region==country code
    df = df.rename(columns={"Region": "Country"})
    return df.dropna(subset=["Rate"]).sort_values(["Country","Cause","Year"])

@st.cache_data
def load_modern_rates() -> pd.DataFrame:
    """Load 2011–present rates from hlth_cd_asdr2, filtering to country codes."""
    df = load_eurostat_series("hlth_cd_asdr2")
    # Region holds NUTS2 codes *and* country codes—keep only country codes (length==2)
    df["Region"] = df["Region"].astype(str)
    df_ctry = df[df["Region"].str.match(r"^[A-Z]{2}$")].copy()
    df_ctry = df_ctry.rename(columns={"Region": "Country"})
    return df_ctry.dropna(subset=["Rate"]).sort_values(["Country","Cause","Year"])

@st.cache_data
def load_data() -> pd.DataFrame:
    """Concatenate 1994–2010 and 2011–present national rates into one table."""
    hist = load_historical_rates()
    mod  = load_modern_rates()
    df = pd.concat([hist, mod], ignore_index=True)
    return df.dropna(subset=["Rate"]).sort_values(["Country","Cause","Year"])

def detect_change_points(ts: pd.Series, pen: float = 3) -> list:
    """PELT change-point detection, safely handling small series."""
    clean = ts.dropna()
    if len(clean) < 2:
        return []
    algo = rpt.Pelt(model="l2").fit(clean.values)
    try:
        return algo.predict(pen=pen)
    except BadSegmentationParameters:
        return []

def compute_joinpoints_and_apc(df_sub: pd.DataFrame) -> pd.DataFrame:
    """Compute linear‐segment slopes and Annual Percent Change (APC) for each segment."""
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
            apc = (slope / np.nanmean(seg_vals)) * 100
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
    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods, freq="Y")
    fc = m.predict(future)
    st.plotly_chart(
        px.line(fc, x="ds", y="yhat", title="Forecasted Mortality Rate")
    )

def main():
    st.set_page_config(layout="wide", page_title="Mortality Rates 1994–Present")
    st.title("Standardised Mortality Rates (1994–Present) by Country")

    df = load_data()
    countries = sorted(df["Country"].unique())
    causes    = sorted(df["Cause"].unique())
    country   = st.sidebar.selectbox("Country", countries)
    cause     = st.sidebar.selectbox("Cause of Death", causes)
    yrs       = sorted(df["Year"].unique())
    y0, y1    = int(yrs[0]), int(yrs[-1])
    year_range = st.sidebar.slider("Year Range", y0, y1, (y0, y1))

    df_f = df[
        (df["Country"]==country) &
        (df["Cause"]  ==cause)   &
        (df["Year"].between(*year_range))
    ]

    st.header(f"{cause} Mortality Rate in {country} ({year_range[0]}–{year_range[1]})")
    if df_f.empty:
        st.warning("No data available for selected filters.")
    else:
        plot_joinpoints(df_f, country, cause)
        st.markdown("### Joinpoint & Annual Percent Change (APC)")
        st.dataframe(compute_joinpoints_and_apc(df_f), use_container_width=True)
        st.markdown("### Forecast Next 10 Years")
        forecast_mortality(df_f)

    st.markdown("---")
    st.info("Data harmonised from national series (1994–2010) and NUTS2 series (2011–present) by selecting country-level codes only.")

if __name__ == "__main__":
    main()
