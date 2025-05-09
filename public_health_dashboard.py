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

# --------------------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas numpy plotly prophet ruptures requests statsmodels
#
# This dashboard downloads Eurostat mortality data from two series:
#   • hlth_cd_hist (1994–2010 historical data)
#   • hlth_cd_aro  (2011–present age-standardized rates)
#
# It parses and filters both, concatenates them, then performs joinpoint
# analysis, forecasts trends, and computes Annual Percent Change (APC).
# --------------------------------------------------------------------------

@st.cache_data
def load_eurostat_series(dataset_id: str) -> pd.DataFrame:
    """
    Generic loader for an SDMX-TSV Eurostat series.
    Applies the same parsing/filtering logic to both hlth_cd_hist and hlth_cd_aro.
    """
    url = (
        f'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/'
        f'{dataset_id}?format=TSV&compressed=true'
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # read the gzipped TSV into a DataFrame
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep='\t', low_memory=False)

    # the first column is the series key: split it into dimensions
    key_col = raw.columns[0]
    dims = key_col.split('\\')[0].split(',')
    raw = raw.rename(columns={key_col: 'series_keys'})
    keys = raw['series_keys'].str.split(',', expand=True)
    keys.columns = dims
    df = pd.concat([keys, raw.drop(columns=['series_keys'])], axis=1)

    # melt all year-columns into long form
    year_cols = [c for c in df.columns if c not in dims]
    df_long = df.melt(
        id_vars=dims,
        value_vars=year_cols,
        var_name='Year',
        value_name='raw_rate'
    )

    # clean and convert
    df_long['Year'] = df_long['Year'].str.strip().astype(int)
    df_long['Rate'] = pd.to_numeric(
        df_long['raw_rate'].str.strip().replace(':', np.nan),
        errors='coerce'
    )

    # keep only annual, normalized rates for the total population
    df_f = df_long[
        (df_long['freq'] == 'A') &    # annual
        (df_long['unit'] == 'NR') &  # normalized rate
        (df_long['sex'] == 'T') &    # both sexes
        (df_long['age'] == 'TOTAL') &
        (df_long['resid'] == 'TOT_IN')
    ].rename(columns={'icd10': 'Cause', 'geo': 'Country'})

    return df_f[['Country', 'Year', 'Cause', 'Rate']]

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and concatenate historical (1994–2010) and current (2011–present) data."""
    df_hist = load_eurostat_series('hlth_cd_hist')
    df_aro  = load_eurostat_series('hlth_cd_aro')
    # union and sort
    df_all = pd.concat([df_hist, df_aro], ignore_index=True)
    df_all = df_all.dropna(subset=['Rate'])
    df_all = df_all.sort_values(['Country', 'Cause', 'Year'])
    return df_all

def detect_change_points(ts: pd.Series, model: str = 'l2', pen: float = 3) -> list:
    algo = rpt.Pelt(model=model).fit(ts.values)
    return algo.predict(pen=pen)

def compute_joinpoints_and_apc(df_sub: pd.DataFrame,
                               model: str = 'l2',
                               pen: float = 3) -> pd.DataFrame:
    df_s = df_sub.sort_values('Year')
    yrs = df_s['Year'].astype(float).values
    rts = df_s['Rate'].values
    bkps = detect_change_points(df_s['Rate'], model=model, pen=pen)[:-1]
    segs = np.split(np.arange(len(yrs)), bkps)
    records = []
    for seg in segs:
        sy, ey = int(yrs[seg].min()), int(yrs[seg].max())
        vals = rts[seg]
        X = sm.add_constant(yrs[seg])
        fit = sm.OLS(vals, X).fit()
        slope = fit.params[1]
        apc = (slope / np.nanmean(vals)) * 100
        records.append({
            'start_year': sy,
            'end_year':   ey,
            'slope':      slope,
            'APC_pct':    apc
        })
    return pd.DataFrame(records)

def plot_joinpoints(df: pd.DataFrame, country: str, cause: str) -> None:
    sub = df[(df['Country'] == country) & (df['Cause'] == cause)].sort_values('Year')
    cps = detect_change_points(sub['Rate'])
    fig = px.line(sub, x='Year', y='Rate', title=f'{cause} Mortality in {country}')
    for idx in cps:
        if idx < len(sub):
            fig.add_vline(x=sub.iloc[idx]['Year'], line_dash='dash')
    st.plotly_chart(fig)

def forecast_mortality(df_sub: pd.DataFrame, periods: int = 10) -> None:
    dfp = df_sub[['Year', 'Rate']].rename(columns={'Year': 'ds', 'Rate': 'y'})
    dfp['ds'] = pd.to_datetime(dfp['ds'], format='%Y')
    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods, freq='Y')
    fc = m.predict(future)
    fig = px.line(fc, x='ds', y='yhat', title='Forecasted Mortality Rates')
    st.plotly_chart(fig)

def main():
    st.set_page_config(layout='wide', page_title='Public Health Dashboard')
    st.title('Public Health Mortality Dashboard (1994–Present)')

    # Load combined data
    df = load_data()

    # Sidebar filters
    countries = sorted(df['Country'].unique())
    causes    = sorted(df['Cause'].unique())
    country   = st.sidebar.selectbox('Country', countries)
    cause     = st.sidebar.selectbox('Cause of Death', causes)
    years     = sorted(df['Year'].unique())
    yr_min, yr_max = int(years[0]), int(years[-1])
    year_range = st.sidebar.slider('Year Range', yr_min, yr_max, (yr_min, yr_max))

    df_f = df[
        (df['Country'] == country) &
        (df['Cause']   == cause)   &
        (df['Year'].between(year_range[0], year_range[1]))
    ]

    # Joinpoint & APC
    st.header(f'{cause} Mortality in {country} ({year_range[0]}–{year_range[1]})')
    if not df_f.empty:
        plot_joinpoints(df_f, country, cause)
        st.markdown('### Joinpoint & Annual Percent Change (APC)')
        st.dataframe(compute_joinpoints_and_apc(df_f))
    else:
        st.warning('No data available for those filters.')

    # Forecasting
    st.markdown('### Forecast Next 10 Years')
    if not df_f.empty:
        forecast_mortality(df_f, periods=10)
    else:
        st.warning('No data available for those filters.')

    # Future enhancements placeholder
    st.markdown('---')
    st.markdown('#### Explainability & Policy Insights (Coming Soon)')
    st.info('You can integrate SHAP, policy simulators, etc.')

if __name__ == '__main__':
    main()
