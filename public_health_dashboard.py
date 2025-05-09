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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pycountry

# --------------------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas numpy plotly prophet ruptures requests statsmodels scikit-learn pycountry
#
# Dashboard for national standardised-death-rate data with joinpoints,
# forecasting (Prophet/ARIMA/ETS/Ensemble), and cluster analysis.
# --------------------------------------------------------------------------

EU_CODES = [
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE",
    "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"
]

# Disease code → full description
CAUSE_NAME_MAP = {
    "TOTAL":"Total",
    "A_B":"Certain infectious and parasitic diseases (A00-B99)",
    # … (all other mappings) …
    "U072":"COVID-19, virus not identified"
}
REV_CAUSE_NAME_MAP = {v:k for k,v in CAUSE_NAME_MAP.items()}

# Country code ↔ full name
COUNTRY_NAME_MAP = {c.alpha_2: c.name for c in pycountry.countries}
COUNTRY_NAME_MAP.update({"FX":"France (Metropolitan)","EU":"European Union","Europe":"Europe"})
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
    keys = raw["series_keys"].str.split(",", expand=True)
    keys.columns = dims
    df = pd.concat([keys, raw.drop(columns=["series_keys"])], axis=1)

    years = [c for c in df.columns if c not in dims]
    long = df.melt(id_vars=dims, value_vars=years,
                   var_name="Year", value_name="raw_rate")
    long["Year"] = long["Year"].str.strip().astype(int)
    long["Rate"] = pd.to_numeric(long["raw_rate"].str.strip().replace(":",np.nan),
                                 errors="coerce")

    units = long["unit"].unique()
    unit_val = "RT" if "RT" in units else ("NR" if "NR" in units else None)
    mask = pd.Series(True, index=long.index)
    if unit_val: mask &= (long["unit"]==unit_val)
    mask &= (long.get("freq")=="A")
    mask &= (long.get("sex")=="T")
    mask &= (long.get("age")=="TOTAL")
    if "resid" in long.columns: mask &= (long["resid"]=="TOT_IN")

    sub = long[mask].copy().rename(columns={"geo":"Region","icd10":"Cause"})
    return sub[["Region","Year","Cause","Rate"]]


@st.cache_data
def load_data() -> pd.DataFrame:
    def ld(ds):
        return load_eurostat_series(ds)\
            .rename(columns={"Region":"Country"})\
            .dropna(subset=["Rate"])
    hist = ld("hlth_cd_asdr")
    mod  = ld("hlth_cd_asdr2")
    mod  = mod[mod["Country"].str.fullmatch(r"[A-Z]{2}")]
    df   = pd.concat([hist,mod],ignore_index=True).sort_values(["Country","Cause","Year"])

    df_eu = df[df["Country"].isin(EU_CODES)]\
        .groupby(["Year","Cause"],as_index=False)["Rate"].mean()
    df_eu["Country"] = "EU"
    df_eur= df.groupby(["Year","Cause"],as_index=False)["Rate"].mean()
    df_eur["Country"] = "Europe"
    return pd.concat([df,df_eu,df_eur],ignore_index=True)


def detect_change_points(ts: pd.Series, pen:float=3)->list:
    clean = ts.dropna()
    if len(clean)<2: return []
    algo = rpt.Pelt(model="l2").fit(clean.values)
    try: return algo.predict(pen=pen)
    except BadSegmentationParameters: return []


def compute_joinpoints_and_apc(df_sub:pd.DataFrame)->pd.DataFrame:
    df_s = df_sub.sort_values("Year")
    yrs, vals = df_s["Year"].values, df_s["Rate"].values
    bkps = detect_change_points(df_s["Rate"])[:-1]
    segs = np.split(np.arange(len(yrs)), bkps) if bkps else [np.arange(len(yrs))]
    recs=[]
    for seg in segs:
        sy,ey = int(yrs[seg].min()), int(yrs[seg].max())
        seg_vals = vals[seg]
        if len(seg_vals)<2 or np.all(np.isnan(seg_vals)):
            recs.append({"start_year":sy,"end_year":ey,"slope":np.nan,"APC_pct":np.nan})
        else:
            slope = sm.OLS(seg_vals, sm.add_constant(yrs[seg])).fit().params[1]
            apc   = (slope/np.nanmean(seg_vals))*100
            recs.append({"start_year":sy,"end_year":ey,"slope":slope,"APC_pct":apc})
    return pd.DataFrame(recs)


def plot_joinpoints(df, c_code, cause_code, c_full, cause_full):
    sub = df[(df["Country"]==c_code)&(df["Cause"]==cause_code)].sort_values("Year")
    cps = detect_change_points(sub["Rate"])
    fig = px.line(sub, x="Year", y="Rate",
                  title=f"{cause_full} in {c_full}")
    for cp in cps:
        if 0<cp<len(sub): fig.add_vline(x=sub.iloc[cp]["Year"],line_dash="dash")
    st.plotly_chart(fig)


def main():
    st.set_page_config(layout="wide",page_title="Mortality Dashboard")
    st.title("Standardised Mortality Rates (1994–Present)")

    df = load_data()
    df["CountryFull"] = df["Country"].map(COUNTRY_NAME_MAP).fillna(df["Country"])
    df["CauseFull"]   = df["Cause"].map(CAUSE_NAME_MAP).fillna(df["Cause"])

    country_full = st.sidebar.selectbox("Country", sorted(df["CountryFull"].unique()))
    country_code = REV_COUNTRY_NAME_MAP[country_full]

    cause_full = st.sidebar.selectbox(
        "Cause of Death", sorted(df[df["Country"]==country_code]["CauseFull"].unique())
    )
    cause_code = REV_CAUSE_NAME_MAP[cause_full]

    yrs = sorted(df["Year"].unique())
    y0,y1 = yrs[0],yrs[-1]
    year_range = st.sidebar.slider("Historical Years", y0, y1, (y0,y1))

    st.header(f"{cause_full} in {country_full} ({year_range[0]}–{year_range[1]})")
    df_f = df[
        (df["Country"]==country_code)&
        (df["Cause"]==cause_code)&
        (df["Year"].between(*year_range))
    ]
    if df_f.empty:
        st.warning("No data for selected filters.")
    else:
        plot_joinpoints(df_f, country_code, cause_code, country_full, cause_full)

        # ---- Cluster Analysis ----
        st.markdown("---")
        st.header("Cluster Analysis")

        df_cl = df[
            (df["Cause"]==cause_code)&
            (df["Year"].between(*year_range))
        ]
        pivot = df_cl.pivot(index="Country", columns="Year", values="Rate")

        # fill missing by interpolation + ffill/bfill
        pivot = pivot.interpolate(axis=1,limit_direction="both")\
                     .ffill(axis=1).bfill(axis=1)\
                     .dropna(axis=0, how="all")

        if pivot.shape[0] < 3:
            st.warning("Not enough countries with data to perform clustering.")
        else:
            X = pivot.values
            max_k = min(10, X.shape[0]-1)

            # compute silhouette scores for k=2..max_k
            sil_scores = {}
            for k in range(2, max_k+1):
                labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
                sil_scores[k] = silhouette_score(X, labels)

            # display silhouette scores
            sil_df = pd.Series(sil_scores, name="silhouette_score").to_frame()
            st.write("Silhouette scores by number of clusters:", sil_df)
            fig_sil = px.line(
                sil_df, x=sil_df.index, y="silhouette_score",
                labels={"index":"# clusters","silhouette_score":"Silhouette"},
                title="Silhouette Score vs. # Clusters"
            )
            st.plotly_chart(fig_sil)

            # choose best k
            best_k = max(sil_scores, key=sil_scores.get)
            st.write(f"Optimal number of clusters (data-driven): **{best_k}**")

            km = KMeans(n_clusters=best_k, random_state=0).fit(X)
            clust_df = pd.DataFrame({
                "Country": pivot.index,
                "Cluster": km.labels_.astype(str)
            })
            clust_df["CountryFull"] = clust_df["Country"].map(COUNTRY_NAME_MAP)
            clust_df["iso_alpha"] = clust_df["Country"].map(
                lambda c: (pycountry.countries.get(alpha_2=c).alpha_3
                           if pycountry.countries.get(alpha_2=c) else None)
            )

            fig_map = px.choropleth(
                clust_df,
                locations="iso_alpha",
                color="Cluster",
                hover_name="CountryFull",
                locationmode="ISO-3",
                scope="europe",
                title=f"{cause_full} Mortality Clusters (k={best_k})"
            )
            st.plotly_chart(fig_map)

    st.markdown("---")
    st.info(
        "Cluster count is selected by maximizing silhouette score over k=2…"
        "min(10, #countries−1), so it adapts to your data."
    )


if __name__=="__main__":
    main()
