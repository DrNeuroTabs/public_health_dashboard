import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
import ruptures as rpt
import requests
import gzip
from io import BytesIO, StringIO
import statsmodels.api as sm

# --------------------------------------------------------------------------
# Requirements:
# pip install streamlit pandas numpy plotly prophet ruptures requests statsmodels
# This dashboard lets you choose between Eurostat (2011–2023) and WHO (1950–2023)
# mortality data, then performs joinpoint analysis, forecasting, and APC calc.
# --------------------------------------------------------------------------

@st.cache_data
def load_eurostat() -> pd.DataFrame:
    """Download, parse, and load the Eurostat hlth_cd_aro dataset."""
    url = (
        'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/'
        'hlth_cd_aro?format=TSV&compressed=true'
    )
    resp = requests.get(url)
    resp.raise_for_status()
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        df = pd.read_csv(gz, sep='\t', low_memory=False)

    # Split the combined-key column into dimensions
    key_col = df.columns[0]  # e.g. 'freq,unit,sex,age,icd10,resid,geo\TIME_PERIOD'
    dims = key_col.split('\\')[0].split(',')
    df = df.rename(columns={key_col: 'series_keys'})
    keys_df = df['series_keys'].str.split(',', expand=True)
    keys_df.columns = dims
    df = pd.concat([keys_df, df.drop(columns=['series_keys'])], axis=1)

    # Melt year-columns into long format
    year_cols = [c for c in df.columns if c not in dims]
    df_long = df.melt(
        id_vars=dims,
        value_vars=year_cols,
        var_name='Year',
        value_name='raw_rate'
    )

    # Clean and convert types
    df_long['Year'] = df_long['Year'].str.strip().astype(int)
    df_long['Rate'] = pd.to_numeric(
        df_long['raw_rate'].str.strip().replace(':', np.nan),
        errors='coerce'
    )

    # Filter to annual normalized rates for total population
    df_f = df_long[
        (df_long['freq'] == 'A') &
        (df_long['unit'] == 'NR') &
        (df_long['sex'] == 'T') &
        (df_long['age'] == 'TOTAL') &
        (df_long['resid'] == 'TOT_IN')
    ].rename(columns={'icd10':
