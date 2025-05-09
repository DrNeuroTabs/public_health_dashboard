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

# ---------------------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas numpy plotly prophet ruptures requests statsmodels
#
# This dashboard pulls two Eurostat series:
#   • hlth_cd_asdr (standardised death rate, 1994–2010)
#   • hlth_cd_aro  (age‐standardised death rate, 2011–present)
# and concatenates them into one 1994–present table.
# ---------------------------------------------------------------------------

@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    """Download, parse, and filter any Eurostat SDMX-TSV series by common dims."""
    url = (
        f'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/'
        f'{dataset_id}?format=TSV&compressed=true'
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep='\t', low_memory=False)

    # Split the combined key column into separate dimension columns
    key_col = raw.columns[0]
    dims = key_col.split('\\')[0].split(',')
    raw = raw.rename(columns={key_col: 'series_keys'})
    keys_df = raw['series_keys'].str.split(',', expand=True)
    keys_df.columns = dims
    df = pd.concat([keys_df, raw.drop(columns=['series_keys'])], axis=1)

    # Melt year-columns into long format
    year_cols = [c for c in df.columns if c not in dims]
    long = df.melt(
        id_vars=dims,
        value_vars=year_cols,
        var_name='Year',
        value_name='raw_rate'
    )

    # Clean types
    long['Year'] = long['Year'].str.strip().astype(int)
    long['Rate'] = pd.to_numeric(
        long['raw_rate'].str.strip().replace(':', np.nan),
        errors='coerce'
    )

    # Dynamically filter on the dims that exist
    filters = {
        'freq': 'A',      # annual
        'unit': 'NR',     # normalized rate
        'sex': 'T',       # total (both sexes)
        'age': 'TOTAL',   # total ages
        'resid': 'TOT_IN' # total residents (if present)
    }
    mask = pd.Series(True, index=long.index)
    for dim, val in filters.items():
        if dim in long.columns:
            mask &= (long[dim] == val)
    df_f = long[mask]

    # Rename for clarity
    rename_map = {}
    if 'icd10' in df_f.columns:
        rename_map['icd10'] = 'Cause'
    if 'geo' in df_f.columns:
        rename_map['geo'] = 'Country'
    df_f = df_f.rename(columns=rename_map)

    return df_f[['Country', 'Year', 'Cause', 'Rate']]

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and concatenate 1994–2010 and 2011–present Eurostat series."""
    hist   = load_eurostat_series('hlth_cd_asdr')  # 1994–2010 standardized rates
    modern = load_eurostat_series('hlth_cd_aro')   # 2011–present age‐standardized rates
    df = pd.concat([hist, modern], ignore_index=True)
    return df.dropna(subset=['Rate']).sort_values(['Country', 'Cause', 'Year'])

def detect_change_points(ts: pd.Series, pen: float = 3) -> list:
    algo = rpt.Pelt(model='l2').fit(ts.values)
    return algo.predict(pen=pen)

def compute_joinpoints_and_apc(df_sub: pd.DataFrame) -> pd.DataFrame:
    df_s = df_sub.sort_values('Year')
    yrs, rts = df_s['Year'].values, df_s['Rate'].values
    bkps = detect_change_points(df_s['Rate'])[:-1]
    segs = np.split(np.arange(len(yrs)), bkps)
    recs = []
    for seg in segs:
        sy, ey = int(yrs[seg].min()), int(yrs[seg].max())
        slope = sm.OLS(rts[seg], sm.add_constant(yrs[seg])).fit().params[1]
        apc = (slope / np.nanmean(rts[seg])) * 100
        recs.append({'start_year': sy, 'end_year': ey, 'slope': slope, 'APC_pct': apc})
    return pd.DataFrame(recs)

def plot_joinpoints(df: pd.DataFrame, country: str, cause: str) -> None:
    sub = df[(df['Country']==country)&(df['Cause']==cause)].sort_values('Year')
    cps = detect_change_points(sub['Rate'])
    fig = px.line(sub, x='Year', y='Rate', title=f'{cause} Mortality in {country}')
    for idx in cps:
        if idx < len(sub):
            fig.add_vline(x=sub.iloc[idx]['Year'], line_dash='dash')
    st.plotly_chart(fig)

def forecast_mortality(df_sub: pd.DataFrame, periods: int = 10) -> None:
    dfp = df_sub[['Year','Rate']].rename(columns={'Year':'ds','Rate':'y'})
    dfp['ds'] = pd.to_datetime(dfp['ds'].astype(str), format='%Y')
    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods, freq='Y')
    fc = m.predict(future)
    st.plotly_chart(px.line(fc, x='ds', y='yhat', title='Forecasted Mortality'))

def main():
    st.set_page_config(layout='wide', page_title='1994–Present Mortality Dashboard')
    st.title('Public Health Mortality Dashboard (1994–Present)')

    df = load_data()
    countries = sorted(df['Country'].unique())
    causes    = sorted(df['Cause'].unique())
    country   = st.sidebar.selectbox('Country', countries)
    cause     = st.sidebar.selectbox('Cause of Death', causes)
    yrs       = sorted(df['Year'].unique())
    y0, y1    = int(yrs[0]), int(yrs[-1])
    yr_range  = st.sidebar.slider('Year Range', y0, y1, (y0, y1))

    df_f = df[
        (df['Country']==country) &
        (df['Cause']==cause) &
        (df['Year'].between(*yr_range))
    ]

    st.header(f'{cause} Mortality in {country} ({yr_range[0]}–{yr_range[1]})')
    if df_f.empty:
        st.warning('No data for selected filters.')
    else:
        plot_joinpoints(df_f, country, cause)
        st.markdown('### Joinpoint & Annual Percent Change (APC)')
        st.dataframe(compute_joinpoints_and_apc(df_f))
        st.markdown('### Forecast Next 10 Years')
        forecast_mortality(df_f)

    st.markdown('---')
    st.markdown('#### Explainability & Policy Insights (Coming Soon)')
    st.info('You can plug in SHAP analyses, scenario simulators, etc.')

if __name__ == '__main__':
    main()
