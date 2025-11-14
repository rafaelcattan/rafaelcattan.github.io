"""
US UNEMPLOYMENT RATE FORECASTING
Complete time series analysis with multiple forecasting models

Required packages:
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn tensorflow prophet deap mapie requests scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
# Data retrieval
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ML libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Prophet
from prophet import Prophet
from deap import base, creator, tools, algorithms
warnings.filterwarnings('ignore')

# Set matplotlib backend for interactive plots
import matplotlib
matplotlib.use('TkAgg')
plt.ion()

# Suppress Prophet logging
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)

# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import breaks_hansen
from statsmodels.regression.linear_model import OLS

# ML libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Prophet
from prophet import Prophet

# Genetic Algorithm
from deap import base, creator, tools, algorithms

# Conformal Prediction
try:
    from mapie.regression import MapieRegressor
    from mapie.subsample import BlockBootstrap
    MAPIE_AVAILABLE = True
except ImportError:
    try:
        from mapie.regression import MapieRegressor
        MAPIE_AVAILABLE = True
        BlockBootstrap = None
    except ImportError:
        print("⚠ MAPIE not available. Conformal prediction will be skipped.")
        MAPIE_AVAILABLE = False
        MapieRegressor = None
        BlockBootstrap = None

# Additional libraries
import requests
from scipy import stats

print("=" * 80)
print("US UNEMPLOYMENT RATE FORECASTING - COMPLETE ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. DATA RETRIEVAL
# ============================================================================

def get_us_unemployment():
    """
    Retrieve US unemployment rate from BLS API
    Series: LNS14000000 (Unemployment Rate)
    """
    print("\n1. RETRIEVING DATA FROM BLS API")
    print("-" * 80)
    
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    
    mykey = ""  # Optional: Add your BLS API key here
    
    payload = {
        "seriesid": ["LNS14000000"],
        "startyear": "2005",
        "endyear": "2025"
    }
    
    if mykey:
        payload["registrationKey"] = mykey
    
    try:
        r = requests.post(url, json=payload, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            
            if data['status'] == 'REQUEST_SUCCEEDED':
                series_data = data['Results']['series'][0]['data']
                
                dates = []
                values = []
                
                for item in series_data:
                    year = int(item['year'])
                    period = item['period']
                    
                    if period.startswith('M'):
                        month = int(period[1:])
                        date = pd.Timestamp(year=year, month=month, day=1)
                        value = float(item['value'])
                        
                        dates.append(date)
                        values.append(value)
                
                df = pd.DataFrame({
                    'year_month': dates,
                    'unemployment_rate': values
                })
                
                df = df.sort_values('year_month').reset_index(drop=True)
                
                print(f"✓ Retrieved {len(df)} data points from BLS")
                print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
                print(f"\nFirst few records:")
                print(df.head())
                print(f"\nLast few records:")
                print(df.tail())
                
                return df
            else:
                print(f"✗ BLS API error: {data.get('message', 'Unknown error')}")
        else:
            print(f"✗ HTTP error: {r.status_code}")
    
    except Exception as e:
        print(f"✗ Failed to retrieve data from BLS API: {str(e)}")
    
    return None

# Retrieve data
df = get_us_unemployment()

if df is None or len(df) < 20:
    print("\n⚠ Using synthetic data for demonstration")
    dates = pd.date_range(start='2005-01-01', end='2025-10-01', freq='MS')
    np.random.seed(42)
    trend = np.linspace(5, 8, len(dates))
    seasonal = 1.5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12)
    noise = np.random.normal(0, 0.3, len(dates))
    values = trend + seasonal + noise
    
    df = pd.DataFrame({
        'year_month': dates,
        'unemployment_rate': values
    })

# ============================================================================
# 2. INITIAL VISUALIZATION WITH CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "=" * 80)
print("2. INITIAL VISUALIZATION")
print("=" * 80)

# Calculate rolling statistics
df['rolling_mean'] = df['unemployment_rate'].rolling(window=12, center=True).mean()
df['rolling_std'] = df['unemployment_rate'].rolling(window=12, center=True).std()
df['ci_upper'] = df['rolling_mean'] + 1.96 * df['rolling_std']
df['ci_lower'] = df['rolling_mean'] - 1.96 * df['rolling_std']

fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(df['year_month'], df['unemployment_rate'], 
        linewidth=2.5, color='#2E86AB', label='Actual Unemployment Rate', alpha=0.8)

ax.plot(df['year_month'], df['rolling_mean'], 
        linewidth=2, color='#A23B72', label='12-Month Rolling Mean', 
        linestyle='--', alpha=0.7)

ax.fill_between(df['year_month'], df['ci_lower'], df['ci_upper'], 
                 alpha=0.2, color='#2E86AB', label='95% Confidence Interval')

ax.axvspan(pd.Timestamp('2007-12-01'), pd.Timestamp('2009-06-01'), 
           alpha=0.1, color='red', label='Great Recession')
ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-01'), 
           alpha=0.1, color='orange', label='COVID-19 Pandemic')

ax.set_title('US Unemployment Rate (2005-2025) with Confidence Intervals', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('us_unemployment_with_ci.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Initial visualization saved as 'us_unemployment_with_ci.png'")

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 80)

df_ts = df.set_index('year_month').copy()

print("\nBasic Statistics:")
print(df_ts['unemployment_rate'].describe())
print(f"\nMissing values: {df_ts['unemployment_rate'].isnull().sum()}")

# EDA Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Time Series
axes[0, 0].plot(df_ts.index, df_ts['unemployment_rate'], linewidth=2.5, color='#2E86AB')
axes[0, 0].set_title('US Unemployment Rate Over Time', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Date', fontsize=11)
axes[0, 0].set_ylabel('Unemployment Rate (%)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvspan(pd.Timestamp('2007-12-01'), pd.Timestamp('2009-06-01'), 
                   alpha=0.15, color='red')
axes[0, 0].axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-01'), 
                   alpha=0.15, color='orange')

# Density Plot
data_clean = df_ts['unemployment_rate'].dropna()
density = stats.gaussian_kde(data_clean)
xs = np.linspace(data_clean.min(), data_clean.max(), 200)
axes[0, 1].plot(xs, density(xs), linewidth=2.5, color='#2E86AB')
axes[0, 1].fill_between(xs, density(xs), alpha=0.3, color='#2E86AB')
axes[0, 1].plot(data_clean, np.zeros_like(data_clean), '|', color='black', 
                markersize=10, alpha=0.5, markeredgewidth=1.5)
axes[0, 1].set_title('Distribution of Unemployment Rate (Density)', 
                     fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Unemployment Rate (%)', fontsize=11)
axes[0, 1].set_ylabel('Density', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# ACF
plot_acf(df_ts['unemployment_rate'].dropna(), lags=40, ax=axes[1, 0])
axes[1, 0].set_title('Autocorrelation Function (ACF)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Lag', fontsize=11)
axes[1, 0].set_ylabel('ACF', fontsize=11)

# PACF
plot_pacf(df_ts['unemployment_rate'].dropna(), lags=40, ax=axes[1, 1])
axes[1, 1].set_title('Partial Autocorrelation Function (PACF)', 
                     fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Lag', fontsize=11)
axes[1, 1].set_ylabel('PACF', fontsize=11)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ EDA plots saved as 'eda_plots.png'")

# Seasonal Decomposition
print("\n" + "-" * 80)
print("SEASONAL DECOMPOSITION")
print("-" * 80)

decomposition = seasonal_decompose(
    df_ts['unemployment_rate'], 
    model='additive', 
    period=12
)

fig, axes = plt.subplots(4, 1, figsize=(16, 10))

decomposition.observed.plot(ax=axes[0], color='#2E86AB', linewidth=2)
axes[0].set_title('Observed', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Observed', fontsize=10)
axes[0].grid(True, alpha=0.3)

decomposition.trend.plot(ax=axes[1], color='#A23B72', linewidth=2)
axes[1].set_title('Trend', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Trend', fontsize=10)
axes[1].grid(True, alpha=0.3)

decomposition.seasonal.plot(ax=axes[2], color='#F18F01', linewidth=2)
axes[2].set_title('Seasonal', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Seasonal', fontsize=10)
axes[2].grid(True, alpha=0.3)

decomposition.resid.plot(ax=axes[3], color='#C73E1D', linewidth=1.5)
axes[3].set_title('Residual', fontsize=12, fontweight='bold')
axes[3].set_ylabel('Residual', fontsize=10)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('seasonal_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Seasonal decomposition saved as 'seasonal_decomposition.png'")

# Stationarity Test
print("\n" + "-" * 80)
print("STATIONARITY TEST (Augmented Dickey-Fuller)")
print("-" * 80)

adf_result = adfuller(df_ts['unemployment_rate'].dropna())

print(f'ADF Statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4f}')
print(f'Critical Values:')
for key, value in adf_result[4].items():
    print(f'  {key}: {value:.4f}')

if adf_result[1] < 0.05:
    print("\n✓ Series is STATIONARY (p < 0.05)")
else:
    print("\n✗ Series is NON-STATIONARY (p >= 0.05)")

# ============================================================================
# 4. STRUCTURAL BREAK ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("4. STRUCTURAL BREAK ANALYSIS")
print("=" * 80)

def chow_test(df_input, break_date, dep_var='unemployment_rate'):
    """Perform Chow test for structural break"""
    from statsmodels.formula.api import ols
    
    df_test = df_input.copy()
    df_test['time'] = np.arange(len(df_test))
    df_test['break_dummy'] = (df_test['year_month'] >= break_date).astype(int)
    df_test['time_after_break'] = df_test['time'] * df_test['break_dummy']
    
    model_full = ols(f'{dep_var} ~ time + break_dummy + time_after_break', 
                     data=df_test).fit()
    model_restricted = ols(f'{dep_var} ~ time', data=df_test).fit()
    
    rss_restricted = model_restricted.ssr
    rss_full = model_full.ssr
    n = len(df_test)
    k = model_full.df_model + 1
    
    chow_stat = ((rss_restricted - rss_full) / 2) / (rss_full / (n - k))
    
    from scipy import stats as sp_stats
    p_value = 1 - sp_stats.f.cdf(chow_stat, 2, n - k)
    
    return {
        'chow_statistic': chow_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Test known break points
break_points = {
    'Great Recession Start': '2007-12-01',
    'Great Recession End': '2009-06-01',
    'COVID-19 Pandemic': '2020-03-01'
}

print("\nCHOW TEST - Testing known economic events:")
for event, date in break_points.items():
    result = chow_test(df, pd.Timestamp(date))
    status = "✓ SIGNIFICANT" if result['significant'] else "✗ Not significant"
    print(f"\n{event} ({date}):")
    print(f"  Chow Statistic: {result['chow_statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  {status}")

# CUSUM Test
print("\n" + "-" * 80)
print("CUSUM TEST")
print("-" * 80)

df_cusum = df_ts.copy()
df_cusum['time'] = np.arange(len(df_cusum))

model = OLS(df_cusum['unemployment_rate'], 
            np.column_stack([np.ones(len(df_cusum)), df_cusum['time']])).fit()

residuals = model.resid
cusum = np.cumsum(residuals) / np.std(residuals)
cusum_sq = np.cumsum(residuals**2) / np.sum(residuals**2)

n = len(residuals)
cusum_upper = 0.948 * np.sqrt(n)
cusum_lower = -0.948 * np.sqrt(n)

cusum_breaks = (cusum > cusum_upper) | (cusum < cusum_lower)
if cusum_breaks.any():
    print("✓ CUSUM indicates structural instability")
    break_indices = np.where(cusum_breaks)[0]
    print(f"  First break detected around: {df_cusum.index[break_indices[0]]}")
else:
    print("✗ CUSUM suggests model is stable")

# Visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Time Series with Breaks
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_ts.index, df_ts['unemployment_rate'], 
         linewidth=2.5, color='#2E86AB', label='Unemployment Rate')

colors_breaks = ['red', 'orange', 'purple']
for i, (event, date) in enumerate(break_points.items()):
    ax1.axvline(pd.Timestamp(date), color=colors_breaks[i], 
                linestyle='--', linewidth=2, alpha=0.7, label=event)

ax1.set_title('US Unemployment Rate with Structural Break Points', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Unemployment Rate (%)', fontsize=11)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# CUSUM
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df_cusum.index, cusum, linewidth=2, color='#2E86AB', label='CUSUM')
ax2.axhline(cusum_upper, color='red', linestyle='--', linewidth=2, label='Upper Bound')
ax2.axhline(cusum_lower, color='red', linestyle='--', linewidth=2, label='Lower Bound')
ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax2.fill_between(df_cusum.index, cusum_lower, cusum_upper, alpha=0.1, color='gray')
ax2.set_title('CUSUM Test for Structural Stability', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('CUSUM', fontsize=10)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# CUSUM²
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(df_cusum.index, cusum_sq, linewidth=2, color='#A23B72', label='CUSUM²')
ax3.axhline(0.0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax3.axhline(1.0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
significance_lines = np.linspace(0, 1, len(df_cusum))
ax3.plot(df_cusum.index, significance_lines + 0.15, 
         color='red', linestyle='--', linewidth=2, alpha=0.5)
ax3.plot(df_cusum.index, significance_lines - 0.15, 
         color='red', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_title('CUSUM² Test for Variance Stability', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('CUSUM²', fontsize=10)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# Rolling Statistics
ax4 = fig.add_subplot(gs[2, 0])
rolling_mean = df_ts['unemployment_rate'].rolling(window=24, center=True).mean()
rolling_std = df_ts['unemployment_rate'].rolling(window=24, center=True).std()
ax4.plot(df_ts.index, rolling_mean, linewidth=2.5, color='#2E86AB', 
         label='24-Month Rolling Mean')
ax4.fill_between(df_ts.index, 
                 rolling_mean - 2*rolling_std, 
                 rolling_mean + 2*rolling_std,
                 alpha=0.2, color='#2E86AB', label='±2 Std Dev')
ax4.set_title('Rolling Mean with Stability Bands', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Unemployment Rate (%)', fontsize=10)
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)

# Residuals
ax5 = fig.add_subplot(gs[2, 1])
ax5.scatter(df_cusum.index, residuals, alpha=0.5, s=30, color='#2E86AB')
ax5.axhline(0, color='red', linestyle='--', linewidth=2)
ax5.axhline(2*np.std(residuals), color='orange', linestyle='--', 
            linewidth=1, alpha=0.5, label='±2σ')
ax5.axhline(-2*np.std(residuals), color='orange', linestyle='--', 
            linewidth=1, alpha=0.5)
ax5.set_title('Residuals from Linear Trend', fontsize=12, fontweight='bold')
ax5.set_xlabel('Date', fontsize=10)
ax5.set_ylabel('Residuals', fontsize=10)
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, alpha=0.3)

plt.savefig('structural_breaks.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Structural break analysis saved as 'structural_breaks.png'")