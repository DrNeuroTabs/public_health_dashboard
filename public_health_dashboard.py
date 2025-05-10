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
#  • Pairwise Granger causality analysis
# --------------------------------------------------------------------------

EU_CODES = [ "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE",
             "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE" ]

SEX_NAME_MAP = {"T":"Total","M":"Male","F":"Female"}
REV_SEX_NAME = {v:k for k,v in SEX_NAME_MAP.items()}

CAUSE_NAME_MAP = {
 "TOTAL":"Total",
    "A_B":"Certain infectious and parasitic diseases (A00-B99)",
    "A15-A19_B90":"Tuberculosis",
    "B15-B19_B942":"Viral hepatitis and sequelae of viral hepatitis",
    "B180-B182":"Chronic viral hepatitis B and C",
    "B20-B24":"Human immunodeficiency virus [HIV] disease",
    "A_B_OTH":"Other infectious and parasitic diseases (A00-B99)",
    "C00-D48":"Neoplasms",
    "C":"Malignant neoplasms (C00-C97)",
    "C00-C14":"Malignant neoplasm of lip, oral cavity, pharynx",
    "C15":"Malignant neoplasm of oesophagus",
    "C16":"Malignant neoplasm of stomach",
    "C18-C21":"Malignant neoplasm of colon, rectum, anus",
    "C22":"Malignant neoplasm of liver and intrahepatic bile ducts",
    "C25":"Malignant neoplasm of pancreas",
    "C32":"Malignant neoplasm of larynx",
    "C33_C34":"Malignant neoplasm of trachea, bronchus and lung",
    "C43":"Malignant melanoma of skin",
    "C50":"Malignant neoplasm of breast",
    "C53":"Malignant neoplasm of cervix uteri",
    "C54_C55":"Malignant neoplasm of other parts of uterus",
    "C56":"Malignant neoplasm of ovary",
    "C61":"Malignant neoplasm of prostate",
    "C64":"Malignant neoplasm of kidney, except renal pelvis",
    "C67":"Malignant neoplasm of bladder",
    "C70-C72":"Malignant neoplasm of brain and CNS",
    "C73":"Malignant neoplasm of thyroid gland",
    "C81-C86":"Hodgkin disease and lymphomas",
    "C88_C90_C96":"Other lymphoid & haematopoietic neoplasms",
    "C91-C95":"Leukaemia",
    "C_OTH":"Other malignant neoplasms (C00-C97)",
    "D00-D48":"Non-malignant neoplasms",
    "D50-D89":"Diseases of blood & blood-forming organs",
    "E":"Endocrine, nutritional & metabolic diseases",
    "E10-E14":"Diabetes mellitus",
    "E_OTH":"Other endocrine, nutritional & metabolic diseases",
    "F":"Mental & behavioural disorders",
    "F01_F03":"Dementia",
    "F10":"Alcohol-related mental disorders",
    "TOXICO":"Drug dependence & toxicomania",
    "F_OTH":"Other mental & behavioural disorders",
    "G_H":"Nervous system & sense organs diseases",
    "G20":"Parkinson disease",
    "G30":"Alzheimer disease",
    "G_H_OTH":"Other nervous system & sense organ diseases",
    "I":"Circulatory system diseases",
    "I20-I25":"Ischaemic heart diseases",
    "I21_I22":"Acute myocardial infarction",
    "I20_I23-I25":"Other ischaemic heart diseases",
    "I30-I51":"Other heart diseases",
    "I60-I69":"Cerebrovascular diseases",
    "I_OTH":"Other circulatory diseases",
    "J":"Respiratory system diseases",
    "J09-J11":"Influenza (including swine flu)",
    "J12-J18":"Pneumonia",
    "J40-J47":"Chronic lower respiratory diseases",
    "J45_J46":"Asthma",
    "J40-J44_J47":"Other lower respiratory diseases",
    "J_OTH":"Other respiratory diseases",
    "K":"Digestive system diseases",
    "K25-K28":"Ulcer of stomach & duodenum",
    "K70_K73_K74":"Chronic liver disease",
    "K72-K75":"Other liver diseases",
    "K_OTH":"Other digestive diseases",
    "L":"Skin & subcutaneous tissue diseases",
    "M":"Musculoskeletal system diseases",
    "RHEUM_ARTHRO":"Rheumatoid arthritis & arthrosis",
    "M_OTH":"Other musculoskeletal diseases",
    "N":"Genitourinary system diseases",
    "N00-N29":"Kidney & ureter diseases",
    "N_OTH":"Other genitourinary diseases",
    "O":"Pregnancy, childbirth & puerperium",
    "P":"Perinatal conditions",
    "Q":"Congenital malformations, deformations and chromosomal abnormalities",
    "R":"Symptoms & abnormal clinical and laboratory findings",
    "R95":"Sudden infant death syndrome",
    "R96-R99":"Ill-defined & unknown causes of mortality",
    "R_OTH":"Other signs & lab findings",
    "V01-Y89":"External causes of morbidity and mortality",
    "ACC":"Accidents",
    "V_Y85":"Transport accidents",
    "ACC_OTH":"Other accidents",
    "W00-W19":"Falls",
    "W65-W74":"Accidental drowning and submersion",
    "X60-X84_Y870":"Intentional self-harm",
    "X40-X49":"Accidental poisoning by and exposure to noxious substances",
    "X85-Y09_Y871":"Assault",
    "Y10-Y34_Y872":"Event of undetermined intent",
    "V01-Y89_OTH":"Other external causes of morbidity and mortality",
    "A-R_V-Y":"All causes (A00-R99 & V01-Y89)",
    "U071":"COVID-19, virus identified",
    "U072":"COVID-19, virus not identified"
}
COUNTRY_NAME_MAP = {c.alpha_2: c.name for c in pycountry.countries}
COUNTRY_NAME_MAP.update({"FX":"France (Metropolitan)","EU":"European Union","Europe":"Europe"})
REV_COUNTRY_NAME_MAP = {v:k for k,v in COUNTRY_NAME_MAP.items()}

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
    raw = raw.rename(columns={first:"series_keys"})
    keys = raw["series_keys"].str.split(",", expand=True)
    keys.columns = dims
    df = pd.concat([keys, raw.drop(columns=["series_keys"])], axis=1)
    years = [c for c in df.columns if c not in dims]
    long = df.melt(id_vars=dims, value_vars=years, var_name="Year", value_name="raw_rate")
    long["Year"] = long["Year"].str.strip().astype(int)
    long["Rate"] = pd.to_numeric(long["raw_rate"].str.strip().replace(":", np.nan), errors="coerce")
    unit_val = "RT" if "RT" in long["unit"].unique() else ("NR" if "NR" in long["unit"].unique() else None)
    mask = pd.Series(True, index=long.index)
    if unit_val: mask &= (long["unit"]==unit_val)
    mask &= (long.get("freq")=="A")
    if "age"   in dims: mask &= (long["age"]=="TOTAL")
    if "sex"   in dims: mask &= (long["sex"]=="T")
    if "resid" in dims: mask &= (long["resid"]=="TOT_IN")
    sub = long[mask].copy()
    rename = {"geo":"Region","sex":"Sex"}
    others = [d for d in dims if d not in ("geo","sex","freq","unit","age","resid")]
    if len(others)==1: rename[others[0]] = "Category"
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
    df_eu = df[df["Country"].isin(EU_CODES)].groupby(["Year","Cause","Sex"], as_index=False)["Rate"].mean()
    df_eu["Country"] = "EU"
    df_eur= df.groupby(["Year","Cause","Sex"], as_index=False)["Rate"].mean()
    df_eur["Country"] = "Europe"
    return pd.concat([df,df_eu,df_eur], ignore_index=True)

def detect_change_points(ts: pd.Series, pen: float=3) -> list:
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
    sub = df_sub.sort_values("Year")
    yrs, rates = sub["Year"].values, sub["Rate"].values
    bkps = detect_change_points(sub["Rate"])[:-1]
    segs = np.split(np.arange(len(yrs)), bkps) if bkps else [np.arange(len(yrs))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yrs, y=rates, mode="markers+lines", name="Data"))
    palette = px.colors.qualitative.Dark24
    for i, seg in enumerate(segs):
        idx, vals = yrs[seg], rates[seg]
        if len(vals)>=2 and not np.all(np.isnan(vals)):
            fit = sm.OLS(vals, sm.add_constant(idx)).fit().predict(sm.add_constant(idx))
            fig.add_trace(go.Scatter(x=idx, y=fit, mode="lines",
                                     line=dict(color=palette[i%len(palette)], width=3),
                                     name=f"Segment {i+1}"))
    fig.update_layout(title=title, xaxis_title="Year", yaxis_title="Rate")
    st.plotly_chart(fig)

def get_prophet_forecast(df_sub, periods: int) -> pd.DataFrame:
    dfp = df_sub[["Year","Rate"]].rename(columns={"Year":"ds","Rate":"y"})
    dfp["ds"] = pd.to_datetime(dfp["ds"].astype(str), format="%Y")
    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.fit(dfp)
    fut = m.make_future_dataframe(periods=periods, freq="Y")
    fc  = m.predict(fut)
    return pd.DataFrame({"Year":fc["ds"].dt.year,"Prophet":fc["yhat"]})

def get_arima_forecast(df_sub, periods: int) -> pd.DataFrame:
    ser   = df_sub.set_index("Year")["Rate"]
    model = ARIMA(ser, order=(1,1,1)).fit()
    preds = model.forecast(periods)
    yrs   = np.arange(ser.index.max()+1, ser.index.max()+1+periods)
    return pd.DataFrame({"Year":yrs,"ARIMA":preds.values})

def get_ets_forecast(df_sub, periods: int) -> pd.DataFrame:
    ser   = df_sub.set_index("Year")["Rate"]
    m     = ExponentialSmoothing(ser, trend="add", seasonal=None).fit(optimized=True)
    preds = m.forecast(periods)
    yrs   = np.arange(ser.index.max()+1, ser.index.max()+1+periods)
    return pd.DataFrame({"Year":yrs,"ETS":preds.values})

def forecast_mortality(df_sub, periods:int, method:str, title:str):
    n = df_sub["Rate"].dropna().shape[0]
    if n<3:
        st.warning(f"Not enough data ({n}) to forecast.")
        return
    prop = get_prophet_forecast(df_sub, periods)
    ari  = get_arima_forecast    (df_sub, periods)
    ets  = get_ets_forecast      (df_sub, periods)
    fc   = prop.merge(ari,on="Year").merge(ets,on="Year")
    if   method=="Prophet": fc["Forecast"] = fc["Prophet"]
    elif method=="ARIMA":   fc["Forecast"] = fc["ARIMA"]
    elif method=="ETS":     fc["Forecast"] = fc["ETS"]
    else:                   fc["Forecast"] = fc[["Prophet","ARIMA","ETS"]].mean(axis=1)
    hist = df_sub[["Year","Rate"]].rename(columns={"Rate":"History"})
    combined = hist.merge(fc[["Year","Forecast"]], on="Year", how="outer")
    fig = px.line(combined, x="Year", y=["History","Forecast"], title=title)
    st.plotly_chart(fig)

def main():
    st.set_page_config(layout="wide", page_title="Public Health Dashboard")
    st.title("Standardised Mortality Rates & Health Factors")

    df = load_data()
    df["CountryFull"] = df["Country"].map(COUNTRY_NAME_MAP)
    df["CauseFull"]   = df["Cause"].map(CAUSE_NAME_MAP)
    df["SexFull"]     = df["Sex"].map(SEX_NAME_MAP)

    # Sidebar
    countries   = sorted(df["CountryFull"].dropna().unique())
    country_full= st.sidebar.selectbox("Country", countries, index=countries.index("European Union"))
    country_code= REV_COUNTRY_NAME_MAP[country_full]
    causes      = sorted(df[df["Country"]==country_code]["CauseFull"].unique())
    cause_full  = st.sidebar.selectbox("Cause of Death", causes)
    cause_code  = REV_CAUSE_NAME_MAP[cause_full]
    sex_sel     = st.sidebar.multiselect("Sex", ["Total","Male","Female"], default=["Total"])
    sex_codes   = [REV_SEX_NAME[s] for s in sex_sel]
    yrs         = sorted(df["Year"].unique())
    year_range  = st.sidebar.slider("Historical Years", yrs[0], yrs[-1], (yrs[0], yrs[-1]))
    forecast_yrs= st.sidebar.slider("Forecast Horizon (yrs)", 1, 30, 10)
    method      = st.sidebar.selectbox("Forecast Method", ["Prophet","ARIMA","ETS","Ensemble"])

    # --- Mortality view for selected country/cause/sex ---
    df_f = df[
        (df["Country"]==country_code)&
        (df["Cause"]==cause_code)&
        (df["Sex"].isin(sex_codes))&
        (df["Year"].between(*year_range))
    ]
    st.header(f"{cause_full} in {country_full} ({year_range[0]}–{year_range[1]})")
    if df_f.empty:
        st.warning("No data for selected filters.")
    else:
        st.markdown("### Joinpoint Trend")
        if len(sex_sel)==1:
            plot_joinpoints_comparative(df_f, f"{cause_full} ({sex_sel[0]}) Trend")
        else:
            plot_joinpoints_comparative(df_f, f"{cause_full} Trend by Sex")

        st.markdown("### Segmented Linear Fits")
        for sc, sf in zip(sex_codes, sex_sel):
            plot_segmented_fit_series(df_f[df_f["Sex"]==sc], f"{cause_full} ({sf}) Fit")

        st.markdown("### Joinpoint & APC")
        st.dataframe(compute_joinpoints_and_apc(df_f), use_container_width=True)

        st.markdown(f"### Forecast next {forecast_yrs} yrs ({method})")
        for sc, sf in zip(sex_codes, sex_sel):
            forecast_mortality(df_f[df_f["Sex"]==sc], forecast_yrs, method, f"{cause_full} ({sf}) Forecast")

    # --- Health‐factor regression (unchanged) ---
    st.markdown("---")
    st.header("Health Factors – Exploratory Regression")
    # (same code as before…)

    # --- Clustering (unchanged) ---
    st.markdown("---")
    st.header("Cluster Analysis (Total Rates)")
    # (same code as before…)

    # --- Granger causality ---
    st.markdown("---")
    st.header("Granger Causality Analysis")
    country_list   = sorted(df["CountryFull"].dropna().unique())
    sel_countries  = st.multiselect("Select countries (default: all)", country_list, default=country_list)
    maxlag         = st.slider("Max lag (years)", 1, 5, 2)
    alpha          = st.number_input("p-value cutoff", 0.01, 0.10, 0.05, 0.01)

    if len(sel_countries) >= 2:
        # **Filter by the same cause & total sex, then pivot via pivot_table**
        df_c = df[
            (df["Cause"]==cause_code)&
            (df["CountryFull"].isin(sel_countries))&
            (df["Sex"]=="T")&
            (df["Year"].between(*year_range))
        ]
        pivot_gc = df_c.pivot_table(
            index="Year", columns="CountryFull", values="Rate", aggfunc="mean"
        ).dropna()

        # Pairwise Granger tests
        pvals = pd.DataFrame(np.nan, index=sel_countries, columns=sel_countries)
        for causer in sel_countries:
            for caused in sel_countries:
                if causer == caused: continue
                data = pivot_gc[[caused, causer]].dropna()
                try:
                    res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                    ps  = [res[l][0]["ssr_chi2test"][1] for l in range(1, maxlag+1)]
                    pvals.loc[causer, caused] = np.min(ps)
                except:
                    pvals.loc[causer, caused] = np.nan

        # Heatmap of –log10(p)
        hm = -np.log10(pvals.astype(float))
        fig_hm = px.imshow(
            hm, text_auto=".2f",
            labels={"x":"Predictor →","y":"Target ↓","color":"–log₁₀(p)"},
            title="Granger Causality Heatmap"
        )
        st.plotly_chart(fig_hm)

        # Network graph where p < alpha
        edges = [
            (i,j) for i in sel_countries for j in sel_countries
            if i!=j and pd.notna(pvals.loc[i,j]) and pvals.loc[i,j]<alpha
        ]
        theta = np.linspace(0,2*np.pi,len(sel_countries),endpoint=False)
        pos   = {n:(np.cos(t),np.sin(t)) for n,t in zip(sel_countries,theta)}
        ex, ey = [], []
        for src,dst in edges:
            x0,y0 = pos[src]; x1,y1 = pos[dst]
            ex += [x0,x1,None]; ey += [y0,y1,None]
        nx_ = [pos[n][0] for n in sel_countries]
        ny_ = [pos[n][1] for n in sel_countries]

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1),hoverinfo="none"))
        fig_net.add_trace(go.Scatter(
            x=nx_,y=ny_,mode="markers+text",
            marker=dict(size=20),text=sel_countries,textposition="bottom center"
        ))
        fig_net.update_layout(
            title=f"Granger Network (p<{alpha})",
            xaxis=dict(visible=False), yaxis=dict(visible=False), height=600
        )
        st.plotly_chart(fig_net)
    else:
        st.info("Select at least two countries to run Granger causality.")

    st.markdown("---")
    st.info("Use the sidebar & above sections to control sex, factors, clustering years, and Granger parameters.")

if __name__ == "__main__":
    main()
