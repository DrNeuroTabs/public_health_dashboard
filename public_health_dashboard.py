import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
import ruptures as rpt
import requests
import gzip
from io import BytesIO
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas numpy plotly prophet ruptures requests statsmodels
#
# This dashboard uses Eurostat’s “files” API to fetch two TSVs:
#   • hlth_cd_asdr (standardised death rate, 1994–2010)
#   • hlth_cd_aro  (age-standardised death rate, 2011–present)
# It looks up the current download URLs via the inventory endpoint, so it
# automatically adapts if Eurostat moves files around. It parses & filters
# them identically, stitches them into one 1994–present series, and then
# performs joinpoint analysis, APC calculation, and forecasting.
# -----------------------------------------------------------------------------

@st.cache_data
def fetch_and_parse_bulk(dataset_id: str) -> pd.DataFrame:
    """Fetch the gzip-TSV for a Eurostat dataset via the files API and parse it."""
    # 1) Retrieve the inventory of all “data” files
    inv_url = "https://ec.europa.eu/eurostat/api/dissemination/files/inventory?type=data"
    inv = requests.get(inv_url, timeout=30)
    inv.raise_for_status()
    inventory = inv.json()  # list of dicts with at least 'path' and 'filename'

    # 2) Find the entry whose path ends with our dataset’s TSV
    target_suffix = f"/data/hlth/{dataset_id}.tsv.gz"
    entry = next(item for item in inventory if item.get("path", "").endswith(target_suffix))

    # 3) Download that file
    download_url = "https://ec.europa.eu" + entry["path"]
    resp = requests.get(download_url, timeout=30)
    resp.raise_for_status()
    buf = BytesIO(resp.content)

    # 4) Decompress & load into a DataFrame
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep="\t", low_memory=False)

    # 5) Split the composite key column into dimensions
    key_col = raw.columns[0]
    dims = key_col.split("\\")[0].split(",")
    raw = raw.rename(columns={key_col: "series_keys"})
    keys_df = raw["series_keys"].str.split(",", expand=True)
    keys_df.columns = dims
    df = pd.concat([keys_df, raw.drop(columns=["series_keys"])], axis=1)

    # 6) Melt all year-columns into long form
    year_cols = [c for c in df.columns if c not in dims]
    long = df.melt(
        id_vars=dims,
        value_vars=year_cols,
        var_name="Year",
        value_name="raw_rate"
    )

    # 7) Clean & convert
    long["Year"] = long["Year"].str.strip().astype(int)
    long["Rate"] = pd.to_numeric(
        long["raw_rate"].str.strip().replace(":", np.nan),
        errors="coerce"
    )

    # 8) Filter to annual NR rates for the total population
    mask = (
        (long.get("freq") == "A") &
        (long.get("unit") == "NR") &
        (long.get("sex") == "T") &
        (long.get("age") == "TOTAL")
    )
    if "resid" in long.columns:
        mask &= (long["resid"] == "TOT_IN")
    df_f = long[mask].copy()

    # 9) Rename for clarity
    rename = {}
    if "icd10" in df_f.columns:
        rename["icd10"] = "Cause"
    if "geo" in df_f.columns:
        rename["geo"] = "Country"
    df_f = df_f.rename(columns=rename)

    return df_f[["Country", "Year", "Cause", "Rate"]]

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load both historical (1994–2010) and modern (2011–present) series."""
    hist = fetch_and_parse_bulk("hlth_cd_asdr")  # 1994–2010
    modern = fetch_and_parse_bulk("hlth_cd_aro")  # 2011–present
    df = pd.concat([hist, modern], ignore_index=True)
    return df.dropna(subset=["Rate"]).sort_values(["Country", "Cause", "Year"])

def detect_change_points(ts: pd.Series, pen: int = 3) -> list:
    algo = rpt.Pelt(model="l2").fit(ts.values)
    return algo.predict(pen=pen)

def compute_joinpoints_and_apc(df_sub: pd.DataFrame) -> pd.DataFrame:
    df_s = df_sub.sort_values("Year")
    yrs, vals = df_s["Year"].values, df_s["Rate"].values
    bkps = detect_change_points(df_s["Rate"])[:-1]
    segs = np.split(np.arange(len(yrs)), bkps)
    recs = []
    for seg in segs:
        sy, ey = int(yrs[seg].min()), int(yrs[seg].max())
        slope = sm.OLS(vals[seg], sm.add_constant(yrs[seg])).fit().params[1]
        apc = (slope / np.nanmean(vals[seg])) * 100
        recs.append({
            "start_year": sy,
            "end_year": ey,
            "slope": slope,
            "APC_pct": apc
        })
    return pd.DataFrame(recs)

def plot_joinpoints(df: pd.DataFrame, country: str, cause: str) -> None:
    sub = df[(df["Country"] == country) & (df["Cause"] == cause)].sort_values("Year")
    cps = detect_change_points(sub["Rate"])
    fig = px.line(sub, x="Year", y="Rate", title=f"{cause} Mortality in {country}")
    for idx in cps:
        if idx < len(sub):
            fig.add_vline(x=sub.iloc[idx]["Year"], line_dash="dash")
    st.plotly_chart(fig)

def forecast_mortality(df_sub: pd.DataFrame, periods: int = 10) -> None:
    dfp = df_sub[["Year","Rate"]].rename(columns={"Year":"ds","Rate":"y"})
    dfp["ds"] = pd.to_datetime(dfp["ds"].astype(str), format="%Y")
    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods, freq="Y")
    fc = m.predict(future)
    fig = px.line(fc, x="ds", y="yhat", title="Forecasted Mortality Rates")
    st.plotly_chart(fig)

def main():
    st.set_page_config(layout="wide", page_title="Public Health Mortality Dashboard")
    st.title("Public Health Mortality Dashboard (1994–Present)")

    df = load_data()
    countries = sorted(df["Country"].unique())
    causes    = sorted(df["Cause"].unique())
    country   = st.sidebar.selectbox("Country", countries)
    cause     = st.sidebar.selectbox("Cause of Death", causes)
    yrs       = sorted(df["Year"].unique())
    y0, y1    = int(yrs[0]), int(yrs[-1])
    year_range = st.sidebar.slider("Year Range", y0, y1, (y0, y1))

    df_f = df[
        (df["Country"] == country) &
        (df["Cause"]   == cause)  &
        (df["Year"].between(*year_range))
    ]

    st.header(f"{cause} Mortality in {country} ({year_range[0]}–{year_range[1]})")
    if df_f.empty:
        st.warning("No data for selected filters.")
    else:
        plot_joinpoints(df_f, country, cause)
        st.markdown("### Joinpoint & Annual Percent Change (APC)")
        st.dataframe(compute_joinpoints_and_apc(df_f))
        st.markdown("### Forecast Next 10 Years")
        forecast_mortality(df_f)

    st.markdown("---")
    st.markdown("#### Explainability & Policy Insights (Coming Soon)")
    st.info("You can plug in SHAP analyses, scenario simulators, etc.")

if __name__ == "__main__":
    main()
