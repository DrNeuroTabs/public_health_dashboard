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
#  • Data-driven clustering + map
#
# Uses a single generic loader for any Eurostat SDMX-TSV dataset.
# --------------------------------------------------------------------------

EU_CODES = [
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE",
    "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"
]

SEX_NAME_MAP = {"T": "Total", "M": "Male", "F": "Female"}
REV_SEX_NAME = {v: k for k, v in SEX_NAME_MAP.items()}

# (Your full CAUSE_NAME_MAP here…)
CAUSE_NAME_MAP = {
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
REV_CAUSE_NAME_MAP = {v: k for k, v in CAUSE_NAME_MAP.items()}

# ISO α2 → full country name
COUNTRY_NAME_MAP = {c.alpha_2: c.name for c in pycountry.countries}
COUNTRY_NAME_MAP.update({
    "FX":"France (Metropolitan)",
    "EU":"European Union","Europe":"Europe"
})
REV_COUNTRY_NAME_MAP = {v:k for k,v in COUNTRY_NAME_MAP.items()}

# Health-factor dataset IDs to explore
FACTOR_IDS = {
    "BMI by citizenship":      "hlth_ehis_bm1c",
    "Phys activity by citizenship": "hlth_ehis_pe9c",
    "Fruit & veg by citizenship":   "hlth_ehis_fv3c",
    "Smoking by citizenship":       "hlth_ehis_sk1c",
    "Social support by citizenship":"hlth_ehis_ss1c",
    "Determinants (2008)":          "hlth_det_h",
    "HC exp by provider":           "hlth_sha11_hp",
    "Staff – physicians":           "hlth_rs_prs2",
    "Staff – hospital":             "hlth_rs_prshp2",
    "Staff – categories":           "hlth_rs_physcat",
    "Beds NUTS2":                   "hlth_rs_bdsrg2",
    "Beds LTC":                     "hlth_rs_bdltc",
    "Imaging devices":              "hlth_rs_medim",
    "Beds hospital":                "hlth_rs_bds2",
    "Tech resources":               "hlth_rs_tech",
    "Consultations":                "hlth_ehis_am1e",
    "Med use prescribed":           "hlth_ehis_md1e",
    "Med use non-prescribed":       "hlth_ehis_md2e",
    "Home care":                    "hlth_ehis_am7e",
    "Unmet needs":                  "hlth_ehis_un1e"
}

@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    """
    Generic loader for any Eurostat SDMX-TSV series.
    Returns standardized DataFrame with columns:
      Region, Year, (Category if present), Sex, Rate
    """
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/"
        f"sdmx/2.1/data/{dataset_id}?format=TSV&compressed=true"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep="\t", low_memory=False)

    # Split first column into dimensions
    first = raw.columns[0]
    dims = first.split("\\")[0].split(",")
    raw = raw.rename(columns={first: "series_keys"})
    keys = raw["series_keys"].str.split(",", expand=True)
    keys.columns = dims

    # Melt year columns
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

    # Filter to annual, total-age, total-sex, resid if present, correct unit
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

    # Rename geo→Region, sex→Sex; leave any other dims alone
    rename = {"geo":"Region", "sex":"Sex"}
    out = sub.rename(columns=rename)

    # Build final column list dynamically
    cols = ["Region","Year","Sex","Rate"]
    if "Category" in out.columns:
        cols.insert(2, "Category")
    return out[cols]

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Loads and concatenates historical (1994–2010) and modern (2011–present)
    mortality series, then appends EU & Europe aggregates.
    """
    def ld(id_):
        df = load_eurostat_series(id_).rename(columns={"Region":"Country", "Category":"Cause"})
        return df.dropna(subset=["Rate"])
    hist = ld("hlth_cd_asdr")
    mod  = ld("hlth_cd_asdr2")
    mod  = mod[mod["Country"].str.fullmatch(r"[A-Z]{2}")]
    df   = pd.concat([hist, mod], ignore_index=True).sort_values(
        ["Country","Cause","Sex","Year"]
    )

    # EU & Europe aggregates
    df_eu = (
        df[df["Country"].isin(EU_CODES)]
        .groupby(["Year","Cause","Sex"], as_index=False)["Rate"].mean()
    ); df_eu["Country"] = "EU"
    df_eur = (
        df.groupby(["Year","Cause","Sex"], as_index=False)["Rate"].mean()
    ); df_eur["Country"] = "Europe"

    return pd.concat([df, df_eu, df_eur], ignore_index=True)

# … all your other helper functions (detect_change_points, compute_joinpoints_and_apc, etc.) stay exactly the same …

def main():
    st.set_page_config(layout="wide", page_title="Public Health Dashboard")
    st.title("Standardised Mortality Rates & Health Factors")

    # Load mortality data
    df = load_data()
    df["CountryFull"] = df["Country"].map(COUNTRY_NAME_MAP)
    df["CauseFull"]   = df["Cause"].map(CAUSE_NAME_MAP)
    df["SexFull"]     = df["Sex"].map(SEX_NAME_MAP)

    # Sidebar selectors (Country, Cause, Sex, Years, Forecast)
    # … identical to your original …

    # Mortality plots, joinpoints, segmented fits, APC, forecasting
    # … identical to your original …

    # Health-factor regression (multivariate)
    # … identical to your original …

    # Cross-country BMI vs Mortality regression
    st.markdown("---")
    st.header("Cross-country BMI vs Mortality")

    years = sorted(df["Year"].unique())
    reg_year = st.select_slider("Select year for regression", years, value=years[-1])

    # Pull BMI (Total sex) for all countries
    bmi = (load_eurostat_series("hlth_ehis_bm1c")
           .rename(columns={"Region":"Country"})
           .query("Year == @reg_year and Sex == 'T'")
           .loc[:, ["Country","Rate"]]
           .rename(columns={"Rate":"BMI"}))

    # Pull mortality (Total sex) for same cause & year
    mort = (df.query("Year == @reg_year and Sex == 'T' and Cause == @cause_code")
            .loc[:, ["Country","Rate"]]
            .rename(columns={"Rate":"Mortality"}))

    reg_df = mort.merge(bmi, on="Country").dropna()
    if reg_df.shape[0] < 5:
        st.warning("Not enough countries with both BMI & mortality data.")
    else:
        X = sm.add_constant(reg_df["BMI"])
        y = reg_df["Mortality"]
        res = sm.OLS(y, X).fit()

        st.subheader(f"OLS Regression: Mortality ~ BMI ({reg_year})")
        st.text(res.summary())

        fig = px.scatter(
            reg_df, x="BMI", y="Mortality", text="Country",
            trendline="ols",
            labels={"BMI":"BMI (kg/m²)", "Mortality":"Mortality rate"},
            title=f"Mortality vs BMI across countries ({reg_year})"
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig)

    # Cluster analysis
    # … identical to your original …

    st.markdown("---")
    st.info("… your footer text …")

if __name__ == "__main__":
    main()
