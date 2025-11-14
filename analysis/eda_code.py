
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
import warnings
warnings.filterwarnings('ignore')
from scipy import stats


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import requests
import pandas as pd
mykey = 'ff7095746d954b8a884141069e4216e9'

url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

payload = {
    "seriesid": ["LNS14000000"],      # Unemployment Rate (US)
    "startyear": "2005",
    "endyear": "2025",
    "registrationKey": mykey
}

r = requests.post(url, json=payload)

r.text
data = r.json()["Results"]["series"][0]["data"]
df = pd.DataFrame(data)

df["unemployment_rate"] = df["value"].astype(float)
df['year_month'] = df['year'] + '-' + df['period'].str[1:]
df['year_month'] = pd.to_datetime(df['year_month'])
        
df = df.sort_values('year_month').reset_index(drop=True)


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
