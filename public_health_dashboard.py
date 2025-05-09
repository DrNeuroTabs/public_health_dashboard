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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import pycountry

# --------------------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas numpy plotly prophet ruptures requests statsmodels pycountry
#
# This dashboard stitches together national standardised-death-rate data:
#  • hlth_cd_asdr   (1994–2010 national rates, unit="RT")
#  • hlth_cd_asdr2  (2011–present NUTS2 rates, unit="NR"), filtering
#                   only the country-level codes (length==2)
# Then it appends two aggregated series:
#  • "EU"     – simple average across the 27 EU member states
#  • "Europe" – simple average across all countries present
# Finally runs joinpoint analysis, APC calculation, and offers multiple forecasting methods.
# --------------------------------------------------------------------------

EU_CODES = ["AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE",
            "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"]

# Disease code → full description (abbreviated here for brevity)
CAUSE_NAME_MAP = {
    "TOTAL":"Total",
    # ... include all your mappings here ...
}
REV_CAUSE_NAME_MAP = {v:k for k,v in CAUSE_NAME_MAP.items()}

# Country code ↔ full name
COUNTRY_NAME_MAP = {c.alpha_2: c.name for c in pycountry.countries}
COUNTRY_NAME_MAP["FX"] = "France (Metropolitan)"
COUNTRY_NAME_MAP["EU"] = "European Union"
COUNTRY_NAME_MAP["Europe"] = "Europe"
REV_COUNTRY_NAME_MAP = {v:k for k,v in COUNTRY_NAME_MAP.items()}


@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    url = f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{dataset_id}?format=TSV&compressed=true"
    resp = requests.get(url, timeout=30); resp.raise_for_status()
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep="\t", low_memory=False)
    key_col = raw.columns[0]
    dims = key_col.split("\\")[0].split(",")
    raw = raw.rename(columns={key_col:"series_keys"})
    keys = raw["series_keys"].str.split(",",expand=True); keys.columns=dims
    df = pd.concat([keys, raw.drop(columns=["series_keys"])],axis=1)
    years = [c for c in df.columns if c not in dims]
    long = df.melt(id_vars=dims, value_vars=years, var_name="Year", value_name="raw_rate")
    long["Year"] = long["Year"].str.strip().astype(int)
    long["Rate"] = pd.to_numeric(long["raw_rate"].str.strip().replace(":",np.nan),errors="coerce")
    units = long["unit"].unique()
    unit_val = "RT" if "RT" in units else ("NR" if "NR" in units else None)
    mask = pd.Series(True,index=long.index)
    if unit_val: mask &= (long["unit"]==unit_val)
    mask &= (long.get("freq")=="A")
    mask &= (long.get("sex")=="T")
    mask &= (long.get("age")=="TOTAL")
    if "resid" in long.columns: mask &= (long["resid"]=="TOT_IN")
    sub = long[mask].copy().rename(columns={"geo":"Region","icd10":"Cause"})
    return sub[["Region","Year","Cause","Rate"]]


@st.cache_data
def load_data() -> pd.DataFrame:
    def ld(ds): return (load_eurostat_series(ds)
                        .rename(columns={"Region":"Country"})
                        .dropna(subset=["Rate"]))
    hist = ld("hlth_cd_asdr")
    mod  = ld("hlth_cd_asdr2")
    mod  = mod[mod["Country"].str.fullmatch(r"[A-Z]{2}")]
    df   = pd.concat([hist,mod],ignore_index=True).sort_values(["Country","Cause","Year"])
    df_eu = (df[df["Country"].isin(EU_CODES)]
             .groupby(["Year","Cause"],as_index=False)["Rate"].mean())
    df_eu["Country"]="EU"
    df_eur= df.groupby(["Year","Cause"],as_index=False)["Rate"].mean()
    df_eur["Country"]="Europe"
    return pd.concat([df,df_eu,df_eur],ignore_index=True)


def detect_change_points(ts: pd.Series, pen: float=3) -> list:
    clean = ts.dropna()
    if len(clean)<2: return []
    algo = rpt.Pelt(model="l2").fit(clean.values)
    try:    return algo.predict(pen=pen)
    except BadSegmentationParameters: return []


def compute_joinpoints_and_apc(df_sub: pd.DataFrame) -> pd.DataFrame:
    df_s = df_sub.sort_values("Year")
    yrs, vals = df_s["Year"].values, df_s["Rate"].values
    bkps = detect_change_points(df_s["Rate"])[:-1]
    segs = np.split(np.arange(len(yrs)),bkps) if bkps else [np.arange(len(yrs))]
    recs=[]
    for seg in segs:
        sy,ey=int(yrs[seg].min()),int(yrs[seg].max())
        seg_vals=vals[seg]
        if len(seg_vals)<2 or np.all(np.isnan(seg_vals)):
            recs.append({"start_year":sy,"end_year":ey,"slope":np.nan,"APC_pct":np.nan})
        else:
            slope=sm.OLS(seg_vals,sm.add_constant(yrs[seg])).fit().params[1]
            apc=(slope/np.nanmean(seg_vals))*100
            recs.append({"start_year":sy,"end_year":ey,"slope":slope,"APC_pct":apc})
    return pd.DataFrame(recs)


def plot_joinpoints(df, code, cause, full, cause_full):
    sub = df[(df["Country"]==code)&(df["Cause"]==cause)].sort_values("Year")
    cps = detect_change_points(sub["Rate"])
    fig=px.line(sub,x="Year",y="Rate",title=f"{cause_full} in {full}")
    for cp in cps:
        if 0<cp<len(sub): fig.add_vline(x=sub.iloc[cp]["Year"],line_dash="dash")
    st.plotly_chart(fig)


def get_prophet_forecast(df_sub, periods):
    dfp=df_sub[["Year","Rate"]].rename(columns={"Year":"ds","Rate":"y"})
    dfp["ds"]=pd.to_datetime(dfp["ds"].astype(str),format="%Y")
    m=Prophet(yearly_seasonality=False,daily_seasonality=False)
    m.fit(dfp)
    fut=m.make_future_dataframe(periods=periods,freq="Y")
    fc=m.predict(fut)
    return pd.DataFrame({"Year":fc["ds"].dt.year,"Prophet":fc["yhat"]})


def get_arima_forecast(df_sub, periods):
    ser = df_sub.sort_values("Year").set_index("Year")["Rate"]
    # simple ARIMA(1,1,1) baseline
    model=ARIMA(ser,order=(1,1,1))
    res=model.fit()
    preds=res.forecast(periods)
    years=np.arange(ser.index.max()+1,ser.index.max()+1+periods)
    return pd.DataFrame({"Year":years,"ARIMA":preds.values})


def get_ets_forecast(df_sub, periods):
    ser=df_sub.sort_values("Year").set_index("Year")["Rate"]
    model=ExponentialSmoothing(ser,trend="add",seasonal=None).fit(optimized=True)
    preds=model.forecast(periods)
    years=np.arange(ser.index.max()+1,ser.index.max()+1+periods)
    return pd.DataFrame({"Year":years,"ETS":preds.values})


def forecast_mortality(df_sub, periods, method):
    prop=get_prophet_forecast(df_sub,periods)
    ari =get_arima_forecast(df_sub,periods)
    ets =get_ets_forecast(df_sub,periods)
    fc=prop.merge(ari,on="Year").merge(ets,on="Year")
    if method=="Prophet":
        fc["Forecast"]=fc["Prophet"]
    elif method=="ARIMA":
        fc["Forecast"]=fc["ARIMA"]
    elif method=="ETS":
        fc["Forecast"]=fc["ETS"]
    else:
        fc["Forecast"]=fc[["Prophet","ARIMA","ETS"]].mean(axis=1)
    hist=df_sub[["Year","Rate"]].rename(columns={"Rate":"History"})
    combined=hist.merge(fc[["Year","Forecast"]],on="Year",how="outer")
    fig=px.line(combined,x="Year",y=["History","Forecast"],title=f"{method} Forecast ({periods} yrs)")
    st.plotly_chart(fig)


def main():
    st.set_page_config(layout="wide",page_title="Mortality Dashboard")
    st.title("Standardised Mortality Rates")

    df=load_data()
    df["CountryFull"]=df["Country"].map(COUNTRY_NAME_MAP).fillna(df["Country"])
    df["CauseFull"]=df["Cause"].map(CAUSE_NAME_MAP).fillna(df["Cause"])

    c_full=st.sidebar.selectbox("Country",sorted(df["CountryFull"].unique()))
    c_code=REV_COUNTRY_NAME_MAP[c_full]

    cause_full=st.sidebar.selectbox(
        "Cause of Death",
        sorted(df[df["Country"]==c_code]["CauseFull"].unique())
    )
    cause_code=REV_CAUSE_NAME_MAP[cause_full]

    yrs=sorted(df["Year"].unique())
    y0,y1=yrs[0],yrs[-1]
    year_range=st.sidebar.slider("Historical Years",y0,y1,(y0,y1))

    forecast_years=st.sidebar.slider("Forecast Horizon (yrs)",1,30,10)

    method=st.sidebar.selectbox("Forecast Method",["Prophet","ARIMA","ETS","Ensemble"])

    df_f=df[
        (df["Country"]==c_code)&
        (df["Cause"]==cause_code)&
        (df["Year"].between(*year_range))
    ]

    st.header(f"{cause_full} in {c_full} ({year_range[0]}–{year_range[1]})")
    if df_f.empty:
        st.warning("No data for selected filters.")
    else:
        plot_joinpoints(df_f,c_code,cause_code,c_full,cause_full)
        st.markdown("### Joinpoint & APC")
        st.dataframe(compute_joinpoints_and_apc(df_f),use_container_width=True)
        st.markdown(f"### Forecast next {forecast_years} yrs ({method})")
        forecast_mortality(df_f,forecast_years,method)

    st.markdown("---")
    st.info("Methods: Prophet, ARIMA(1,1,1), ETS, Ensemble. For ML/DL, consider sktime/gluonts or LSTM-based models.")

if __name__=="__main__":
    main()
