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
#  • hlth_cd_asdr2  (2011–present NUTS2 rates, unit="NR"), filtering
#                   only the country‐level codes (length==2)
# Then it appends two aggregated series:
#  • "EU"     – simple average across the 27 EU member states
#  • "Europe" – simple average across all countries present
# Finally runs joinpoint analysis, APC calc, and forecasting.
# --------------------------------------------------------------------------

EU_CODES = [
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE",
    "IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"
]

# Map of Disease Codes → Full Descriptions
CAUSE_NAME_MAP = {
    "TOTAL":            "Total",
    "A_B":              "Certain infectious and parasitic diseases (A00-B99)",
    "A15-A19_B90":      "Tuberculosis",
    "B15-B19_B942":     "Viral hepatitis and sequelae of viral hepatitis",
    "B180-B182":        "Chronic viral hepatitis B and C",
    "B20-B24":          "Human immunodeficiency virus [HIV] disease",
    "A_B_OTH":          "Other infectious and parasitic diseases (remainder of A00-B99)",
    "C00-D48":          "Neoplasms",
    "C":                "Malignant neoplasms (C00-C97)",
    "C00-C14":          "Malignant neoplasm of lip, oral cavity, pharynx",
    "C15":              "Malignant neoplasm of oesophagus",
    "C16":              "Malignant neoplasm of stomach",
    "C18-C21":          "Malignant neoplasm of colon, rectosigmoid junction, rectum, anus and anal canal",
    "C22":              "Malignant neoplasm of liver and intrahepatic bile ducts",
    "C25":              "Malignant neoplasm of pancreas",
    "C32":              "Malignant neoplasm of larynx",
    "C33_C34":          "Malignant neoplasm of trachea, bronchus and lung",
    "C43":              "Malignant melanoma of skin",
    "C50":              "Malignant neoplasm of breast",
    "C53":              "Malignant neoplasm of cervix uteri",
    "C54_C55":          "Malignant neoplasm of other parts of uterus",
    "C56":              "Malignant neoplasm of ovary",
    "C61":              "Malignant neoplasm of prostate",
    "C64":              "Malignant neoplasm of kidney, except renal pelvis",
    "C67":              "Malignant neoplasm of bladder",
    "C70-C72":          "Malignant neoplasm of brain and central nervous system",
    "C73":              "Malignant neoplasm of thyroid gland",
    "C81-C86":          "Hodgkin disease and lymphomas",
    "C88_C90_C96":      "Other malignant neoplasm of lymphoid, haematopoietic and related tissue",
    "C91-C95":          "Leukaemia",
    "C_OTH":            "Other malignant neoplasms (remainder of C00-C97)",
    "D00-D48":          "Non-malignant neoplasms (benign and uncertain)",
    "D50-D89":          "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
    "E":                "Endocrine, nutritional and metabolic diseases (E00-E90)",
    "E10-E14":          "Diabetes mellitus",
    "E_OTH":            "Other endocrine, nutritional and metabolic diseases (remainder of E00-E90)",
    "F":                "Mental and behavioural disorders (F00-F99)",
    "F01_F03":          "Dementia",
    "F10":              "Mental and behavioural disorders due to use of alcohol",
    "TOXICO":           "Drug dependence, toxicomania (F11-F16, F18-F19)",
    "F_OTH":            "Other mental and behavioural disorders (remainder of F00-F99)",
    "G_H":              "Diseases of the nervous system and the sense organs (G00-H95)",
    "G20":              "Parkinson disease",
    "G30":              "Alzheimer disease",
    "G_H_OTH":          "Other diseases of the nervous system and the sense organs (remainder of G00-H95)",
    "I":                "Diseases of the circulatory system (I00-I99)",
    "I20-I25":          "Ischaemic heart diseases",
    "I21_I22":          "Acute myocardial infarction including subsequent myocardial infarction",
    "I20_I23-I25":      "Other ischaemic heart diseases",
    "I30-I51":          "Other heart diseases",
    "I60-I69":          "Cerebrovascular diseases",
    "I_OTH":            "Other diseases of the circulatory system (remainder of I00-I99)",
    "J":                "Diseases of the respiratory system (J00-J99)",
    "J09-J11":          "Influenza (including swine flu)",
    "J12-J18":          "Pneumonia",
    "J40-J47":          "Chronic lower respiratory diseases",
    "J45_J46":          "Asthma and status asthmaticus",
    "J40-J44_J47":      "Other lower respiratory diseases",
    "J_OTH":            "Other diseases of the respiratory system (remainder of J00-J99)",
    "K":                "Diseases of the digestive system (K00-K93)",
    "K25-K28":          "Ulcer of stomach, duodenum and jejunum",
    "K70_K73_K74":      "Chronic liver disease",
    "K_OTH":            "Other diseases of the digestive system (remainder of K00-K93)",
    "L":                "Diseases of the skin and subcutaneous tissue (L00-L99)",
    "M":                "Diseases of the musculoskeletal system and connective tissue (M00-M99)",
    "RHEUM_ARTHRO":     "Rheumatoid arthritis and arthrosis (M05-M06,M15-M19)",
    "M_OTH":            "Other diseases of the musculoskeletal system and connective tissue (remainder of M00-M99)",
    "N":                "Diseases of the genitourinary system (N00-N99)",
    "N00-N29":          "Diseases of kidney and ureter",
    "N_OTH":            "Other diseases of the genitourinary system (remainder of N00-N99)",
    "O":                "Pregnancy, childbirth and the puerperium (O00-O99)",
    "P":                "Certain conditions originating in the perinatal period (P00-P96)",
    "Q":                "Congenital malformations, deformations and chromosomal abnormalities (Q00-Q99)",
    "R":                "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified (R00-R99)",
    "R95":              "Sudden infant death syndrome",
    "R96-R99":          "Ill-defined and unknown causes of mortality",
    "R_OTH":            "Other symptoms, signs and abnormal clinical and laboratory findings (remainder of R00-R99)",
    "V01-Y89":          "External causes of morbidity and mortality (V01-Y89)",
    "ACC":              "Accidents (V01-X59, Y85, Y86)",
    "V_Y85":            "Transport accidents (V01-V99, Y85)",
    "ACC_OTH":          "Other accidents (W20-W64, W75-X39, X50-X59, Y86)",
    "W00-W19":          "Falls",
    "W65-W74":          "Accidental drowning and submersion",
    "X60-X84_Y870":     "Intentional self-harm",
    "X40-X49":          "Accidental poisoning by and exposure to noxious substances",
    "X85-Y09_Y871":     "Assault",
    "Y10-Y34_Y872":     "Event of undetermined intent",
    "V01-Y89_OTH":      "Other external causes of morbidity and mortality (remainder of V01-Y89)"
}

# Reverse map for lookup
REV_CAUSE_NAME_MAP = {v: k for k, v in CAUSE_NAME_MAP.items()}


@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/
        2.1/data/{dataset_id}?format=TSV&compressed=true"
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
    long = df.melt(
        id_vars=dims,
        value_vars=year_cols,
        var_name="Year",
        value_name="raw_rate"
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

    # EU aggregate
    df_eu = (
        df[df["Country"].isin(EU_CODES)]
        .groupby(["Year","Cause"], as_index=False)["Rate"].mean()
    )
    df_eu["Country"] = "EU"

    # Europe aggregate
    df_eur = (
        df.groupby(["Year","Cause"], as_index=False)["Rate"].mean()
    )
    df_eur["Country"] = "Europe"

    return pd.concat([df, df_eu, df_eur], ignore_index=True)


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
            recs.append({"start_year": sy, "end_year": ey, "slope": np.nan, "APC_pct": np.nan})
        else:
            slope = sm.OLS(seg_vals, sm.add_constant(yrs[seg])).fit().params[1]
            apc = (slope / np.nanmean(seg_vals)) * 100
            recs.append({"start_year": sy, "end_year": ey, "slope": slope, "APC_pct": apc})
    return pd.DataFrame(recs)


def plot_joinpoints(df: pd.DataFrame, country: str, cause: str) -> None:
    sub = df[(df["Country"] == country) & (df["Cause"] == cause)].sort_values("Year")
    cps = detect_change_points(sub["Rate"])
    fig = px.line(
        sub, x="Year", y="Rate",
        title=f"{CAUSE_NAME_MAP.get(cause, cause)} Mortality Rate in {country}"
    )
    for cp in cps:
        if 0 < cp < len(sub):
            fig.add_vline(x=sub.iloc[cp]["Year"], line_dash="dash")
    st.plotly_chart(fig)


def forecast_mortality(df_sub: pd.DataFrame, periods: int = 10) -> None:
    dfp = df_sub[["Year", "Rate"]].rename(columns={"Year": "ds", "Rate": "y"})
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

    # Build a column of full cause names
    df["CauseFull"] = df["Cause"].map(CAUSE_NAME_MAP).fillna(df["Cause"])

    countries = sorted(df["Country"].unique())
    country = st.sidebar.selectbox("Country", countries)

    causes_full = sorted(df[df["Country"] == country]["CauseFull"].unique())
    cause_full = st.sidebar.selectbox("Cause of Death", causes_full)

    # Reverse lookup code from full name, fallback to full name itself
    cause_code = REV_CAUSE_NAME_MAP.get(cause_full, cause_full)

    yrs = sorted(df["Year"].unique())
    y0, y1 = int(yrs[0]), int(yrs[-1])
    year_range = st.sidebar.slider("Year Range", y0, y1, (y0, y1))

    df_f = df[
        (df["Country"] == country) &
        (df["Cause"] == cause_code) &
        (df["Year"].between(*year_range))
    ]

    st.header(f"{cause_full} Mortality Rate in {country} ({year_range[0]}–{year_range[1]})")
    if df_f.empty:
        st.warning("No data available for selected filters.")
    else:
        plot_joinpoints(df_f, country, cause_code)
        st.markdown("### Joinpoint & Annual Percent Change (APC)")
        st.dataframe(compute_joinpoints_and_apc(df_f), use_container_width=True)
        st.markdown("### Forecast Next 10 Years")
        forecast_mortality(df_f)

    st.markdown("---")
    st.info(
        "Data combined from national (1994–2010) and NUTS2 (2011–present) series; "
        "EU and Europe aggregates appended as simple means."
    )


if __name__ == "__main__":
    main()
