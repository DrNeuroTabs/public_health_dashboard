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

    # split the combined-key column
    key_col = df.columns[0]
    dims = key_col.split('\\')[0].split(',')
    df = df.rename(columns={key_col: 'series_keys'})
    keys_df = df['series_keys'].str.split(',', expand=True)
    keys_df.columns = dims
    df = pd.concat([keys_df, df.drop(columns=['series_keys'])], axis=1)

    # melt years
    year_cols = [c for c in df.columns if c not in dims]
    df_long = df.melt(id_vars=dims, value_vars=year_cols,
                      var_name='Year', value_name='raw_rate')

    # clean & convert
    df_long['Year'] = df_long['Year'].str.strip().astype(int)
    df_long['Rate'] = pd.to_numeric(df_long['raw_rate'].str.strip()
                                    .replace(':', np.nan), errors='coerce')

    # filter to annual, normalized, total pop
    df_f = df_long[
        (df_long['freq'] == 'A') &
        (df_long['unit'] == 'NR') &
        (df_long['sex'] == 'T') &
        (df_long['age'] == 'TOTAL') &
        (df_long['resid'] == 'TOT_IN')
    ].rename(columns={'icd10': 'Cause', 'geo': 'Country'})

    return df_f[['Country', 'Year', 'Cause', 'Rate']]


@st.cache_data
def load_who() -> pd.DataFrame:
    """Download and load WHO GHO MORT_100_1 cause-of-death dataset."""
    url = "https://ghoapi.azureedge.net/api/MORT_100_1?Format=CSV"
    resp = requests.get(url)
    resp.raise_for_status()
    txt = resp.content.decode('utf-8')
    df = pd.read_csv(StringIO(txt))

    # rename to match our schema (adjust if your column names differ)
    df = df.rename(columns={
        'SpatialDim': 'Country',
        'TimeDim':    'Year',
        'Dim1':       'Cause',
        'NumericValue': 'Rate'
    })

    # filter to both sexes & all ages if present
    if 'Dim2' in df.columns:
        df = df[df['Dim2'] == 'Both sexes']
    if 'Dim3' in df.columns:
        df = df[df['Dim3'] == 'All ages']

    df['Year'] = df['Year'].astype(int)
    df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')

    return df[['Country', 'Year', 'Cause', 'Rate']]


def detect_change_points(ts: pd.Series, model: str = 'l2', pen: float = 3) -> list:
    algo = rpt.Pelt(model=model).fit(ts.values)
    return algo.predict(pen=pen)


def compute_joinpoints_and_apc(df_sub: pd.DataFrame, model: str = 'l2', pen: float = 3) -> pd.DataFrame:
    df_s = df_sub.sort_values('Year')
    yrs  = df_s['Year'].values
    rts  = df_s['Rate'].values
    bkps = detect_change_points(pd.Series(rts), model=model, pen=pen)[:-1]
    segs = np.split(np.arange(len(yrs)), bkps)
    out  = []
    for seg in segs:
        sy, ey = yrs[seg].min(), yrs[seg].max()
        srates = rts[seg]
        X = sm.add_constant(yrs[seg])
        fit = sm.OLS(srates, X).fit()
        slope = fit.params[1]
        apc   = (slope / np.nanmean(srates)) * 100
        out.append({'start_year': int(sy), 'end_year': int(ey),
                    'slope': slope, 'APC_pct': apc})
    return pd.DataFrame(out)


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
    st.title('Public Health Mortality Dashboard')

    # Let user pick data source
    source = st.sidebar.radio(
        "Data source",
        ["Eurostat (2011–2023)", "WHO (1950–2023)"]
    )
    df = load_eurostat() if source.startswith("Eurostat") else load_who()

    # Filters
    countries = sorted(df['Country'].unique())
    causes    = sorted(df['Cause'].unique())
    c_sel = st.sidebar.selectbox('Country', countries)
    k_sel = st.sidebar.selectbox('Cause of Death', causes)
    yrs = sorted(df['Year'].unique())
    yr_min, yr_max = int(yrs[0]), int(yrs[-1])
    yr_range = st.sidebar.slider('Year Range', yr_min, yr_max, (yr_min, yr_max))

    df_f = df[
        (df['Country'] == c_sel) &
        (df['Cause']   == k_sel) &
        (df['Year'].between(yr_range[0], yr_range[1]))
    ]

    # Joinpoints & APC
    st.header(f'{k_sel} Mortality in {c_sel} ({yr_range[0]}–{yr_range[1]})')
    if not df_f.empty:
        plot_joinpoints(df, c_sel, k_sel)
        st.markdown('### Joinpoint & Annual Percent Change (APC)')
        apc_df = compute_joinpoints_and_apc(df_f)
        st.dataframe(apc_df)
    else:
        st.warning('No data for selected filters.')

    # Forecast
    st.markdown('### Forecasting Next 10 Years')
    if not df_f.empty:
        forecast_mortality(df_f, periods=10)
    else:
        st.warning('No data for selected filters.')

    st.markdown('---')
    st.markdown('#### Explainability & Policy Insights (Coming Soon)')
    st.info('You can plug in SHAP, scenario simulators, etc.')

if __name__ == '__main__':
    main()
