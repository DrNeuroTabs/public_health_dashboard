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
#
# This dashboard downloads two Bulk‐Download TSVs:
#  • hlth_cd_asdr (1994–2010 standardised death rate)
#  • hlth_cd_aro  (2011–present age‐standardised rate)
# parses, filters, concatenates them, then does joinpoints, APC, forecasting.
# --------------------------------------------------------------------------

@st.cache_data
def load_bulk(dataset_id: str) -> pd.DataFrame:
    """Fetch and parse a Eurostat Bulk‐Download TSV for the given dataset."""
    url = (
        "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/"
        f"BulkDownloadListing?file=hlth/{dataset_id}.tsv.gz"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buf = BytesIO(resp.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        raw = pd.read_csv(gz, sep='\t', low_memory=False)

    # split the composite key into dimensions
    key = raw.columns[0]
    dims = key.split('\\')[0].split(',')
    raw = raw.rename(columns={key: 'series_keys'})
    keys = raw['series_keys'].str.split(',', expand=True)
    keys.columns = dims
    df = pd.concat([keys, raw.drop(columns=['series_keys'])], axis=1)

    # melt all year-columns into long form
    years = [c for c in df.columns if c not in dims]
    long = df.melt(
        id_vars=dims,
        value_vars=years,
        var_name='Year',
        value_name='raw_rate'
    )

    # clean & convert
    long['Year'] = long['Year'].str.strip().astype(int)
    long['Rate'] = pd.to_numeric(
        long['raw_rate'].str.strip().replace(':', np.nan),
        errors='coerce'
    )

    # filter to annual, normalized total-population rate
    mask = (
        (long.get('freq') == 'A') &
        (long.get('unit') == 'NR') &
        (long.get('sex') == 'T') &
        (long.get('age') == 'TOTAL')
    )
    # some datasets include 'resid', some don't—.get() handles that
    if 'resid' in long.columns:
        mask &= (long['resid'] == 'TOT_IN')

    df_f = long[mask]
    rename = {}
    if 'icd10' in df_f.columns: rename['icd10'] = 'Cause'
    if 'geo'   in df_f.columns: rename['geo']   = 'Country'
    return df_f.rename(columns=rename)[['Country','Year','Cause','Rate']]

@st.cache_data
def load_data() -> pd.DataFrame:
    hist   = load_bulk('hlth_cd_asdr')  # 1994–2010 :contentReference[oaicite:2]{index=2}
    modern = load_bulk('hlth_cd_aro')   # 2011–present :contentReference[oaicite:3]{index=3}
    df = pd.concat([hist, modern], ignore_index=True)
    return df.dropna(subset=['Rate']).sort_values(['Country','Cause','Year'])

def detect_change_points(ts: pd.Series, pen: int = 3) -> list:
    algo = rpt.Pelt(model='l2').fit(ts.values)
    return algo.predict(pen=pen)

def compute_joinpoints_and_apc(df_sub: pd.DataFrame) -> pd.DataFrame:
    df_s = df_sub.sort_values('Year')
    yrs, vals = df_s['Year'].values, df_s['Rate'].values
    bkps = detect_change_points(df_s['Rate'])[:-1]
    segs = np.split(np.arange(len(yrs)), bkps)
    records = []
    for seg in segs:
        sy, ey = int(yrs[seg].min()), int(yrs[seg].max())
        slope = sm.OLS(vals[seg], sm.add_constant(yrs[seg])).fit().params[1]
        apc   = (slope / np.nanmean(vals[seg])) * 100
        records.append({'start_year':sy,'end_year':ey,'slope':slope,'APC_pct':apc})
    return pd.DataFrame(records)

def plot_joinpoints(df, country, cause):
    sub = df[(df['Country']==country)&(df['Cause']==cause)].sort_values('Year')
    cps = detect_change_points(sub['Rate'])
    fig = px.line(sub, x='Year', y='Rate', title=f'{cause} Mortality in {country}')
    for idx in cps:
        if idx < len(sub):
            fig.add_vline(x=sub.iloc[idx]['Year'], line_dash='dash')
    st.plotly_chart(fig)

def forecast_mortality(df_sub, periods=10):
    dfp = df_sub[['Year','Rate']].rename(columns={'Year':'ds','Rate':'y'})
    dfp['ds'] = pd.to_datetime(dfp['ds'].astype(str), format='%Y')
    m = Prophet(yearly_seasonality=False,daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods, freq='Y')
    fc = m.predict(future)
    st.plotly_chart(px.line(fc, x='ds', y='yhat', title='Forecasted Mortality'))

def main():
    st.set_page_config(layout='wide', page_title='1994–Present Dashboard')
    st.title('Public Health Mortality Dashboard (1994–Present)')

    df = load_data()
    countries = sorted(df['Country'].unique())
    causes    = sorted(df['Cause'].unique())
    c = st.sidebar.selectbox('Country', countries)
    k = st.sidebar.selectbox('Cause of Death', causes)
    yrs = sorted(df['Year'].unique())
    y0, y1 = yrs[0], yrs[-1]
    yr_range = st.sidebar.slider('Year Range', y0, y1, (y0, y1))

    sub = df[(df['Country']==c)&(df['Cause']==k)&(df['Year'].between(*yr_range))]
    st.header(f'{k} Mortality in {c} ({yr_range[0]}–{yr_range[1]})')
    if sub.empty:
        st.warning('No data for selected filters.')
    else:
        plot_joinpoints(sub, c, k)
        st.markdown('### Joinpoint & Annual Percent Change (APC)')
        st.dataframe(compute_joinpoints_and_apc(sub))
        st.markdown('### Forecast Next 10 Years')
        forecast_mortality(sub)

    st.markdown('---')
    st.markdown('#### Explainability & Policy Insights (Coming Soon)')
    st.info('Plug in SHAP analyses, scenario simulators, etc.')

if __name__ == '__main__':
    main()
