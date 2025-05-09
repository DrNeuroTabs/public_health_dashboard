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
# This dashboard pulls two Eurostat SDMX‐TSV series via the 
# dissemination API:
#   • hlth_cd_asdr (standardised death rate, 1994–2010)
#   • hlth_cd_aro  (age-standardised rate, 2011–present)
# It parses & filters both, concatenates into a 1994–present table,
# then runs joinpoint analysis, APC calculation, and forecasting.
# -----------------------------------------------------------------------------

@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    """Fetch and parse a Eurostat SDMX‐TSV series, filtering to annual total rates."""
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        f"{dataset_id}?format=TSV&compressed=true"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep="\t", low_memory=False)

    # Split the composite key into dimension columns
    key_col = raw.columns[0]
    dims = key_col.split("\\")[0].split(",")
    raw = raw.rename(columns={key_col: "series_keys"})
    keys_df = raw["series_keys"].str.split(",", expand=True)
    keys_df.columns = dims
    df = pd.concat([keys_df, raw.drop(columns=["series_keys"])], axis=1)

    # Melt all year-columns into long form
    year_cols = [c for c in df.columns if c not in dims]
    long = df.melt(
        id_vars=dims,
        value_vars=year_cols,
        var_name="Year",
        value_name="raw_rate"
    )

    # Clean & convert
    long["Year"] = long["Year"].str.strip().astype(int)
    long["Rate"] = pd.to_numeric(
        long["raw_rate"].str.strip().replace(":", np.nan),
        errors="coerce"
    )

    # Filter to annual ('A'), normalized rate ('NR'), both sexes ('T'), all ages ('TOTAL'),
    # and total residents ('TOT_IN') if that dimension is present
    mask = (
        (long.get("freq") == "A") &
        (long.get("unit") == "NR") &
        (long.get("sex") == "T") &
        (long.get("age") == "TOTAL")
    )
    if "resid" in long.columns:
        mask &= (long["resid"] == "TOT_IN")
    df_f = long[mask].copy()

    # Rename for clarity
    rename = {}
    if "icd10" in df_f.columns:
        rename["icd10"] = "Cause"
    if "geo" in df_f.columns:
        rename["geo"] = "Country"
    df_f = df_f.rename(columns=rename)

    return df_f[["Country", "Year", "Cause", "Rate"]]

@st.cache_data
def load_data() -> pd.DataFrame:
    """Concatenate 1994–2010 and 2011–present series into one table."""
    hist   = load_eurostat_series("hlth_cd_asdr")  # 1994–2010
    modern = load_eurostat_series("hlth_cd_aro")   # 2011–present
    df = pd.concat([hist, modern], ignore_index=True)
    return df.dropna(subset=["Rate"]).sort_values(["Country", "Cause", "Year"])

def detect_change_points(ts: pd.Series, pen: float = 3) -> list:
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
            "end_year":   ey,
            "slope":      slope,
            "APC_pct":    apc
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
    dfp = df_sub[["Year", "Rate"]].rename(columns={"Year": "ds", "Rate": "y"})
    dfp["ds"] = pd.to_datetime(dfp["ds"].astype(str), format="%Y")
    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods, freq="Y")
    fc = m.predict(future)
    st.plotly_chart(
        px.line(fc, x="ds", y="yhat", title="Forecasted Mortality Rates")
    )

def main():
    st.set_page_config(layout="wide", page_title="Public Health Mortality Dashboard")
    st.title("Public Health Mortality Dashboard (1994–Present)")

    df = load_data()
    countries = sorted(df["Country"].unique())
    causes    = sorted(df["Cause"].unique())
    country   = st.sidebar.selectbox("Select Country", countries)
    cause     = st.sidebar.selectbox("Select Cause of Death", causes)
    years     = sorted(df["Year"].unique())
    yr_min, yr_max = int(years[0]), int(years[-1])
    year_range = st.sidebar.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))

    df_f = df[
        (df["Country"] == country) &
        (df["Cause"]   == cause)   &
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
    st.info("You can integrate SHAP analyses, scenario simulators, etc.")

if __name__ == "__main__":
    main()
