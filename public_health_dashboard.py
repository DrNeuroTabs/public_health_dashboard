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
from statsmodels.tsa.stattools import grangercausalitytests
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
#  • Pairwise Granger causality analysis (global & neighbor‐only)
# --------------------------------------------------------------------------

EU_CODES = [
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE",
    "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"
]

# neighbor adjacency within EU_CODES (alpha-2)
NEIGHBORS = {
    "AT":["DE","CZ","SK","HU","SI","IT"],
    "BE":["FR","DE","NL","LU"],
    "BG":["RO","EL"],
    "HR":["SI","HU"],
    "CY":[],    # island
    "CZ":["DE","PL","SK","AT"],
    "DK":["DE"],
    "EE":["LV"],
    "FI":["SE"],
    "FR":["BE","LU","DE","IT","ES"],
    "DE":["DK","PL","CZ","AT","FR","LU","BE","NL"],
    "EL":["BG"],
    "HU":["AT","SK","RO","HR"],
    "IE":[],
    "IT":["FR","AT","SI"],
    "LV":["EE","LT"],
    "LT":["LV","PL"],
    "LU":["BE","DE","FR"],
    "MT":[],
    "NL":["BE","DE"],
    "PL":["DE","CZ","SK","LT"],
    "PT":["ES"],
    "RO":["BG","HU"],
    "SK":["CZ","PL","HU","AT"],
    "SI":["IT","AT","HU","HR"],
    "ES":["FR","PT"],
    "SE":["FI"]
}

SEX_NAME_MAP = {"T": "Total", "M": "Male", "F": "Female"}
REV_SEX_NAME = {v: k for k, v in SEX_NAME_MAP.items()}

CAUSE_NAME_MAP = {
    # ... (your existing ICD-10 mapping here) ...
}
REV_CAUSE_NAME_MAP = {v: k for k, v in CAUSE_NAME_MAP.items()}

COUNTRY_NAME_MAP = {c.alpha_2: c.name for c in pycountry.countries}
COUNTRY_NAME_MAP.update({
    "FX":"France (Metropolitan)",
    "EU":"European Union","Europe":"Europe"
})
REV_COUNTRY_NAME_MAP = {v: k for k, v in COUNTRY_NAME_MAP.items()}

FACTOR_IDS = {
    # ... (your existing health‐factor IDs here) ...
}

@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    url = f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{dataset_id}?format=TSV&compressed=true"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep="\t", low_memory=False)
    first = raw.columns[0]
    dims = first.split("\\")[0].split(",")
    raw = raw.rename(columns={first: "series_keys"})
    keys = raw["series_keys"].str.split(",", expand=True); keys.columns = dims
    df = pd.concat([keys, raw.drop(columns=["series_keys"])], axis=1)
    years = [c for c in df.columns if c not in dims]
    long = df.melt(id_vars=dims, value_vars=years, var_name="Year", value_name="raw_rate")
    long["Year"] = long["Year"].str.strip().astype(int)
    long["Rate"] = pd.to_numeric(long["raw_rate"].str.strip().replace(":", np.nan), errors="coerce")
    unit_val = "RT" if "RT" in long["unit"].unique() else ("NR" if "NR" in long["unit"].unique() else None)
    mask = pd.Series(True, index=long.index)
    if unit_val: mask &= (long["unit"]==unit_val)
    mask &= (long.get("freq")=="A")
    if "age" in dims:   mask &= (long["age"]=="TOTAL")
    if "sex" in dims:   mask &= (long["sex"]=="T")
    if "resid" in dims: mask &= (long["resid"]=="TOT_IN")
    sub = long[mask].copy()
    rename = {"geo":"Region","sex":"Sex"}
    others = [d for d in dims if d not in ("geo","sex","freq","unit","age","resid")]
    if len(others)==1: rename[others[0]]="Category"
    out = sub.rename(columns=rename)
    return out[["Region","Year","Category","Sex","Rate"]]

@st.cache_data
def load_data() -> pd.DataFrame:
    def ld(id_):
        df = load_eurostat_series(id_).rename(columns={"Region":"Country","Category":"Cause"})
        return df.dropna(subset=["Rate"])
    hist = ld("hlth_cd_asdr")
    mod  = ld("hlth_cd_asdr2")
    mod  = mod[mod["Country"].str.fullmatch(r"[A-Z]{2}")]
    df   = pd.concat([hist,mod], ignore_index=True).sort_values(["Country","Cause","Sex","Year"])
    df_eu = df[df["Country"].isin(EU_CODES)].groupby(["Year","Cause","Sex"], as_index=False)["Rate"].mean(); df_eu["Country"]="EU"
    df_eur= df.groupby(["Year","Cause","Sex"], as_index=False)["Rate"].mean(); df_eur["Country"]="Europe"
    return pd.concat([df,df_eu,df_eur], ignore_index=True)

def detect_change_points(ts: pd.Series, pen: float = 3) -> list:
    clean = ts.dropna()
    if len(clean)<2: return []
    algo = rpt.Pelt(model="l2").fit(clean.values)
    try: return algo.predict(pen=pen)
    except BadSegmentationParameters: return []

def compute_joinpoints_and_apc(df_sub: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for sex in df_sub["Sex"].unique():
        part = df_sub[df_sub["Sex"]==sex].sort_values("Year")
        yrs, vals = part["Year"].values, part["Rate"].values
        bkps = detect_change_points(part["Rate"])[:-1]
        segs = np.split(np.arange(len(yrs)), bkps) if bkps else [np.arange(len(yrs))]
        for seg in segs:
            sy, ey = int(yrs[seg].min()), int(yrs[seg].max())
            sv = vals[seg]
            if len(sv)<2 or np.all(np.isnan(sv)):
                recs.append({"Sex":SEX_NAME_MAP[sex],"start_year":sy,"end_year":ey,"slope":np.nan,"APC_pct":np.nan})
            else:
                slope = sm.OLS(sv, sm.add_constant(yrs[seg])).fit().params[1]
                apc   = (slope/np.nanmean(sv))*100
                recs.append({"Sex":SEX_NAME_MAP[sex],"start_year":sy,"end_year":ey,"slope":slope,"APC_pct":apc})
    return pd.DataFrame(recs)

def plot_joinpoints_comparative(df_sub: pd.DataFrame, title: str):
    df_sub["SexFull"] = df_sub["Sex"].map(SEX_NAME_MAP)
    fig = px.line(df_sub, x="Year", y="Rate", color="SexFull", title=title, markers=True)
    st.plotly_chart(fig)

def plot_segmented_fit_series(df_sub: pd.DataFrame, title: str):
    sub = df_sub.sort_values("Year"); yrs, rates = sub["Year"].values, sub["Rate"].values
    bkps = detect_change_points(sub["Rate"])[:-1]
    segs = np.split(np.arange(len(yrs)), bkps) if bkps else [np.arange(len(yrs))]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=yrs, y=rates, mode="markers+lines", name="Data"))
    palette = px.colors.qualitative.Dark24
    for i, seg in enumerate(segs):
        idx, vals = yrs[seg], rates[seg]
        if len(vals)>=2 and not np.all(np.isnan(vals)):
            fit = sm.OLS(vals, sm.add_constant(idx)).fit().predict(sm.add_constant(idx))
            fig.add_trace(go.Scatter(x=idx, y=fit, mode="lines",
                                     line=dict(color=palette[i%len(palette)],width=3),
                                     name=f"Segment {i+1}"))
    fig.update_layout(title=title, xaxis_title="Year", yaxis_title="Rate")
    st.plotly_chart(fig)

def get_prophet_forecast(df_sub, periods: int) -> pd.DataFrame:
    dfp = df_sub[["Year","Rate"]].rename(columns={"Year":"ds","Rate":"y"})
    dfp["ds"] = pd.to_datetime(dfp["ds"].astype(str), format="%Y")
    m = Prophet(yearly_seasonality=False, daily_seasonality=False); m.fit(dfp)
    fut = m.make_future_dataframe(periods=periods, freq="Y"); fc = m.predict(fut)
    return pd.DataFrame({"Year":fc["ds"].dt.year,"Prophet":fc["yhat"]})

def get_arima_forecast(df_sub, periods:int)->pd.DataFrame:
    ser = df_sub.set_index("Year")["Rate"]
    res = ARIMA(ser,order=(1,1,1)).fit(); preds = res.forecast(periods)
    yrs = np.arange(ser.index.max()+1,ser.index.max()+1+periods)
    return pd.DataFrame({"Year":yrs,"ARIMA":preds.values})

def get_ets_forecast(df_sub, periods:int)->pd.DataFrame:
    ser = df_sub.set_index("Year")["Rate"]
    m = ExponentialSmoothing(ser,trend="add",seasonal=None).fit(optimized=True)
    preds = m.forecast(periods); yrs = np.arange(ser.index.max()+1,ser.index.max()+1+periods)
    return pd.DataFrame({"Year":yrs,"ETS":preds.values})

def forecast_mortality(df_sub, periods:int, method:str, title:str):
    n = df_sub["Rate"].dropna().shape[0]
    if n<3:
        st.warning(f"Not enough data ({n}) to forecast.")
        return
    prop = get_prophet_forecast(df_sub,periods)
    ari  = get_arima_forecast   (df_sub,periods)
    ets  = get_ets_forecast     (df_sub,periods)
    fc   = prop.merge(ari,on="Year").merge(ets,on="Year")
    if   method=="Prophet": fc["Forecast"]=fc["Prophet"]
    elif method=="ARIMA":   fc["Forecast"]=fc["ARIMA"]
    elif method=="ETS":     fc["Forecast"]=fc["ETS"]
    else:                    fc["Forecast"]=fc[["Prophet","ARIMA","ETS"]].mean(axis=1)
    hist = df_sub[["Year","Rate"]].rename(columns={"Rate":"History"})
    combined = hist.merge(fc[["Year","Forecast"]],on="Year",how="outer")
    fig = px.line(combined,x="Year",y=["History","Forecast"],title=title)
    st.plotly_chart(fig)

def main():
    st.set_page_config(layout="wide", page_title="Public Health Dashboard")
    st.title("Standardised Mortality Rates & Health Factors")

    df = load_data()
    df["CountryFull"] = df["Country"].map(COUNTRY_NAME_MAP)
    df["CauseFull"]   = df["Cause"].map(CAUSE_NAME_MAP)
    df["SexFull"]     = df["Sex"].map(SEX_NAME_MAP)

    # Sidebar
    countries    = sorted(df["CountryFull"].dropna().unique())
    country_full = st.sidebar.selectbox("Country", countries, index=countries.index("European Union"))
    country_code = REV_COUNTRY_NAME_MAP[country_full]
    causes       = sorted(df[df["Country"]==country_code]["CauseFull"].dropna().unique())
    cause_full   = st.sidebar.selectbox("Cause of Death", causes)
    cause_code   = REV_CAUSE_NAME_MAP[cause_full]
    sex_sel      = st.sidebar.multiselect("Sex", ["Total","Male","Female"], default=["Total"])
    sex_codes    = [REV_SEX_NAME[s] for s in sex_sel]
    yrs          = sorted(df["Year"].unique())
    year_range   = st.sidebar.slider("Historical Years", yrs[0], yrs[-1], (yrs[0], yrs[-1]))
    forecast_yrs = st.sidebar.slider("Forecast Horizon (yrs)", 1, 30, 10)
    method       = st.sidebar.selectbox("Forecast Method", ["Prophet","ARIMA","ETS","Ensemble"])

    # Mortality trends & forecasts (unchanged) …
    # [joinpoint, segmented fit, APC, forecasting code as before]

    # Clustering (unchanged) …
    # [cluster analysis code as before]

    # === Global Granger causality ===
    st.markdown("---")
    st.header("Global Granger Causality Analysis")
    country_list  = sorted(df["CountryFull"].dropna().unique())
    sel_countries = st.multiselect("Select countries (default: all)", country_list, default=country_list)
    maxlag        = st.slider("Max lag (years)", 1, 5, 2, key="gl_maxlag")
    alpha_cutoff  = st.number_input("p-value cutoff", 0.01, 0.10, 0.05, 0.01, key="gl_alpha")

    if len(sel_countries) >= 2:
        df_c = df[
            (df["Cause"]==cause_code)&
            (df["CountryFull"].isin(sel_countries))&
            (df["Sex"]=="T")&
            (df["Year"].between(*year_range))
        ]
        pivot_gc = df_c.pivot_table(index="Year", columns="CountryFull", values="Rate", aggfunc="mean")
        common = [c for c in sel_countries if c in pivot_gc.columns]
        if len(common) < 2:
            st.warning("Not enough data; adjust selection or years.")
        else:
            pvals = pd.DataFrame(np.nan, index=common, columns=common)
            for causer in common:
                for caused in common:
                    if causer==caused: continue
                    data = pivot_gc[[caused,causer]].dropna()
                    if len(data)>= maxlag+1:
                        try:
                            res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                            ps = [res[l][0]["ssr_chi2test"][1] for l in range(1, maxlag+1)]
                            pvals.loc[causer,caused] = np.min(ps)
                        except: pass
            # heatmap & network (as before) …

    # === Neighbor-based Granger causality ===
    st.markdown("---")
    st.header("Neighbor-Based Causality")
    base_full = st.selectbox("Select focal country", countries, index=countries.index("Germany"))
    base_code = REV_COUNTRY_NAME_MAP[base_full]
    nbr_codes = NEIGHBORS.get(base_code, [])
    nbr_full  = [COUNTRY_NAME_MAP[c] for c in nbr_codes]
    st.subheader("Map: focal country & its neighbors")
    map_df = pd.DataFrame({
        "CountryFull": [base_full] + nbr_full,
        "Role": ["Focal"] + ["Neighbor"] * len(nbr_full)
    })
    map_df["iso_alpha"] = map_df["CountryFull"].map(
        lambda x: pycountry.countries.get(alpha_2=REV_COUNTRY_NAME_MAP[x]).alpha_3
    )
    st.plotly_chart(px.choropleth(
        map_df, locations="iso_alpha", color="Role",
        hover_name="CountryFull", locationmode="ISO-3",
        scope="europe", title="Focal + Neighbors"
    ))

    if len(nbr_full) >= 1:
        # only focal + neighbors
        gb = [base_full] + nbr_full
        df_n = df[
            (df["Cause"]==cause_code)&
            (df["CountryFull"].isin(gb))&
            (df["Sex"]=="T")&
            (df["Year"].between(*year_range))
        ]
        pivot_n = df_n.pivot_table(index="Year", columns="CountryFull", values="Rate", aggfunc="mean")
        common_n = [c for c in gb if c in pivot_n.columns]
        if len(common_n)<2:
            st.warning("Not enough neighbor data; try a different focal country or wider years.")
        else:
            # run Granger among focal & neighbors
            pvals_n = pd.DataFrame(np.nan, index=common_n, columns=common_n)
            maxlag_n = st.slider("Max lag (years) for neighbors", 1, 5, 2, key="nb_maxlag")
            alpha_n  = st.number_input("p-value cutoff for neighbors", 0.01, 0.10, 0.05, 0.01, key="nb_alpha")
            for causer in common_n:
                for caused in common_n:
                    if causer==caused: continue
                    data = pivot_n[[caused,causer]].dropna()
                    if len(data)>= maxlag_n+1:
                        try:
                            res = grangercausalitytests(data, maxlag=maxlag_n, verbose=False)
                            ps = [res[l][0]["ssr_chi2test"][1] for l in range(1, maxlag_n+1)]
                            pvals_n.loc[causer,caused] = np.min(ps)
                        except: pass
            # neighbor heatmap
            hm_n = -np.log10(pvals_n.astype(float))
            fig_hm_n = px.imshow(
                hm_n, text_auto=".2f",
                labels={"x":"Predictor →","y":"Target ↓","color":"–log₁₀(p)"},
                title="Neighbor-Based Heatmap"
            )
            st.plotly_chart(fig_hm_n)
            # neighbor network
            edges_n = [
                (i,j) for i in common_n for j in common_n
                if i!=j and pd.notna(pvals_n.loc[i,j]) and pvals_n.loc[i,j]< alpha_n
            ]
            theta = np.linspace(0,2*np.pi,len(common_n),endpoint=False)
            pos   = {n:(np.cos(t),np.sin(t)) for n,t in zip(common_n,theta)}
            ex, ey = [], []
            for src,dst in edges_n:
                x0,y0 = pos[src]; x1,y1 = pos[dst]
                ex+=[x0,x1,None]; ey+=[y0,y1,None]
            nx_, ny_ = zip(*(pos[n] for n in common_n))
            fig_net_n = go.Figure()
            fig_net_n.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1),hoverinfo="none"))
            fig_net_n.add_trace(go.Scatter(
                x=nx_,y=ny_,mode="markers+text",
                marker=dict(size=20),text=common_n,textposition="bottom center"
            ))
            fig_net_n.update_layout(title=f"Neighbor Network (p<{alpha_n})",
                                    xaxis=dict(visible=False),yaxis=dict(visible=False),height=600)
            st.plotly_chart(fig_net_n)

    st.markdown("---")
    st.info("Use the sidebar & above controls for global vs neighbor causality, years, lags, and p-values.")

if __name__ == "__main__":
    main()
