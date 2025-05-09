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
# performs joinpoint (change-point) analysis, forecasts trends, and computes
# Annual Percent Change (APC) per segment.
# --------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """Download and load the Eurostat mortality dataset via API."""
    url = (
        'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/'
        'hlth_cd_aro?format=TSV&compressed=true'
    )
    response = requests.get(url)
    response.raise_for_status()
    buf = BytesIO(response.content)
    with gzip.GzipFile(fileobj=buf) as gzipped:
        df = pd.read_csv(gzipped, sep='\t', low_memory=False)
    df = df[df['unit'] == 'NR']
    df = df.rename(
        columns={
            'TIME': 'Year',
            'GEO': 'Country',
            'cause': 'Cause',
            'OBS_VALUE': 'Rate'
        }
    )
    return df

def detect_change_points(ts: pd.Series, model: str = 'l2', pen: float = 3) -> list:
    """Use PELT algorithm to detect change-point indices in a series."""
    algo = rpt.Pelt(model=model).fit(ts.values)
    return algo.predict(pen=pen)

def compute_joinpoints_and_apc(df_sub: pd.DataFrame, model: str = 'l2', pen: float = 3) -> pd.DataFrame:
    """Compute segments from change-points and calculate APC for each segment."""
    df = df_sub.sort_values('Year')
    years = df['Year'].astype(float).values
    rates = df['Rate'].values
    # get break indices, drop final
    bkps = detect_change_points(pd.Series(rates), model=model, pen=pen)[:-1]
    segments = np.split(np.arange(len(years)), bkps)
    results = []
    for seg in segments:
        seg_years = years[seg]
        seg_rates = rates[seg]
        # linear fit: Rate ~ Year
        X = sm.add_constant(seg_years)
        fit = sm.OLS(seg_rates, X).fit()
        slope = fit.params[1]
        apc = (slope / np.mean(seg_rates)) * 100
        results.append({
            'start_year': int(seg_years.min()),
            'end_year': int(seg_years.max()),
            'slope': slope,
            'APC_pct': apc
        })
    return pd.DataFrame(results)

def plot_joinpoints(df: pd.DataFrame, country: str, cause: str) -> None:
    sub = df[(df['Country'] == country) & (df['Cause'] == cause)].sort_values('Year')
    ts = sub['Rate']
    cps = detect_change_points(ts)
    fig = px.line(sub, x='Year', y='Rate', title=f'{cause} Mortality in {country}')
    for idx in cps:
        if idx < len(sub):
            year_cp = sub.iloc[idx]['Year']
            fig.add_vline(x=year_cp, line_dash='dash')
    st.plotly_chart(fig)

def forecast_mortality(df_sub: pd.DataFrame, periods: int = 10) -> None:
    df_prop = df_sub[['Year', 'Rate']].rename(columns={'Year': 'ds', 'Rate': 'y'})
    df_prop['ds'] = pd.to_datetime(df_prop['ds'], format='%Y')
    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
    m.fit(df_prop)
    future = m.make_future_dataframe(periods=periods, freq='Y')
    forecast = m.predict(future)
    fig = px.line(forecast, x='ds', y='yhat', title='Forecasted Mortality Rates')
    st.plotly_chart(fig)

def main():
    st.set_page_config(layout='wide', page_title='Public Health Dashboard')
    st.title('Public Health Mortality Dashboard')

    # Load data
    df = load_data()

    # Sidebar controls
    countries = sorted(df['Country'].unique())
    causes = sorted(df['Cause'].unique())
    country_sel = st.sidebar.selectbox('Select Country', countries)
    cause_sel = st.sidebar.selectbox('Select Cause of Death', causes)
    years = sorted(df['Year'].unique())
    year_min, year_max = int(years[0]), int(years[-1])
    year_range = st.sidebar.slider('Year Range', year_min, year_max, (year_min, year_max))

    # Filter data
    df_filtered = df[
        (df['Country'] == country_sel) &
        (df['Cause'] == cause_sel) &
        (df['Year'].between(year_range[0], year_range[1]))
    ]

    # Plot joinpoints and APC table
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
