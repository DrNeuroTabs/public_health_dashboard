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
# pip install streamlit pandas numpy plotly prophet ruptures requests statsmodels
# This dashboard downloads mortality data directly from Eurostat's API,
# parses and filters the data, performs joinpoint analysis, forecasts trends,
# and computes Annual Percent Change (APC) per segment.
# --------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """Download, parse, and load the Eurostat hlth_cd_aro dataset."""
    # 1) Download the compressed TSV
    url = (
        'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/'
        'hlth_cd_aro?format=TSV&compressed=true'
    )
    resp = requests.get(url)
    resp.raise_for_status()
    buf = BytesIO(resp.content)

    # 2) Read into a DataFrame
    with gzip.GzipFile(fileobj=buf) as gz:
        df = pd.read_csv(gz, sep='\t', low_memory=False)

    # 3) Split the combined-key column into separate dims
    key_col = df.columns[0]  # 'freq,unit,sex,age,icd10,resid,geo\TIME_PERIOD'
    dims_part = key_col.split('\\')[0]  # 'freq,unit,sex,age,icd10,resid,geo'
    dims = dims_part.split(',')
    df = df.rename(columns={key_col: 'series_keys'})
    keys_df = df['series_keys'].str.split(',', expand=True)
    keys_df.columns = dims
    df = pd.concat([keys_df, df.drop(columns=['series_keys'])], axis=1)

    # 4) Melt year columns into long format
    year_cols = [c for c in df.columns if c not in dims]
    df_long = df.melt(
        id_vars=dims,
        value_vars=year_cols,
        var_name='Year',
        value_name='raw_rate'
    )

    # 5) Clean and type-convert
    df_long['Year'] = df_long['Year'].str.strip().astype(int)
    df_long['Rate'] = (
        df_long['raw_rate']
        .replace(':', np.nan)
        .astype(float)
    )

    # 6) Filter to annual normalized rates for total population
    df_filtered = df_long[
        (df_long['freq'] == 'A') &
        (df_long['unit'] == 'NR') &
        (df_long['sex'] == 'T') &
        (df_long['age'] == 'TOTAL') &
        (df_long['resid'] == 'TOT_IN')
    ]

    # 7) Rename for clarity
    df_filtered = df_filtered.rename(columns={
        'icd10': 'Cause',
        'geo':   'Country'
    })

    return df_filtered[['Country', 'Year', 'Cause', 'Rate']]

def detect_change_points(ts: pd.Series, model: str = 'l2', pen: float = 3) -> list:
    """Use PELT algorithm to detect change-point indices in a series."""
    algo = rpt.Pelt(model=model).fit(ts.values)
    return algo.predict(pen=pen)

def compute_joinpoints_and_apc(df_sub: pd.DataFrame, model: str = 'l2', pen: float = 3) -> pd.DataFrame:
    """Compute segments from change-points and calculate APC for each segment."""
    df_sorted = df_sub.sort_values('Year')
    years = df_sorted['Year'].values
    rates = df_sorted['Rate'].values
    bkps = detect_change_points(pd.Series(rates), model=model, pen=pen)[:-1]
    segments = np.split(np.arange(len(years)), bkps)
    results = []
    for seg in segments:
        seg_years = years[seg]
        seg_rates = rates[seg]
        X = sm.add_constant(seg_years)
        fit = sm.OLS(seg_rates, X).fit()
        slope = fit.params[1]
        apc = (slope / np.mean(seg_rates)) * 100
        results.append({
            'start_year': int(seg_years.min()),
            'end_year':   int(seg_years.max()),
            'slope':      slope,
            'APC_pct':    apc
        })
    return pd.DataFrame(results)

def plot_joinpoints(df: pd.DataFrame, country: str, cause: str) -> None:
    sub = df[(df['Country'] == country) & (df['Cause'] == cause)].sort_values('Year')
    ts = sub['Rate']
    cps = detect_change_points(ts)
    fig = px.line(sub, x='Year', y='Rate', title=f'{cause} Mortality in {country}')
    for idx in cps:
        if idx < len(sub):
            cp_year = sub.iloc[idx]['Year']
            fig.add_vline(x=cp_year, line_dash='dash')
    st.plotly_chart(fig)

def forecast_mortality(df_sub: pd.DataFrame, periods: int = 10) -> None:
    df_prophet = df_sub[['Year', 'Rate']].rename(columns={'Year': 'ds', 'Rate': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=periods, freq='Y')
    forecast = m.predict(future)
    fig = px.line(forecast, x='ds', y='yhat', title='Forecasted Mortality Rates')
    st.plotly_chart(fig)

def main():
    st.set_page_config(layout='wide', page_title='Public Health Dashboard')
    st.title('Public Health Mortality Dashboard')

    # Load and filter data
    df = load_data()

    # Sidebar controls
    countries = sorted(df['Country'].unique())
    causes   = sorted(df['Cause'].unique())
    country_sel = st.sidebar.selectbox('Select Country', countries)
    cause_sel   = st.sidebar.selectbox('Select Cause of Death', causes)
    years = sorted(df['Year'].unique())
    yr_min, yr_max = years[0], years[-1]
    year_range = st.sidebar.slider('Year Range', int(yr_min), int(yr_max), (int(yr_min), int(yr_max)))

    df_filtered = df[
        (df['Country'] == country_sel) &
        (df['Cause'] == cause_sel)   &
        (df['Year'].between(year_range[0], year_range[1]))
    ]

    # Joinpoint plot & APC table
    st.header(f'{cause_sel} Mortality in {country_sel} ({year_range[0]}–{year_range[1]})')
    if not df_filtered.empty:
        plot_joinpoints(df, country_sel, cause_sel)
        st.markdown('### Joinpoint & Annual Percent Change (APC)')
        apc_df = compute_joinpoints_and_apc(df_filtered)
        st.dataframe(apc_df)
    else:
        st.warning('No data available for selected filters.')

    # Forecasting
    st.markdown('### Forecasting Next 10 Years')
    if not df_filtered.empty:
        forecast_mortality(df_filtered, periods=10)
    else:
        st.warning('No data available for selected filters.')

    # Future enhancements placeholder
    st.markdown('---')
    st.markdown('#### Explainability & Policy Insights (Coming Soon)')
    st.info('Integration with SHAP or policy indicator modeling can be added.')

if __name__ == '__main__':
    main()
