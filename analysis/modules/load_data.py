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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# FETCH DATA FROM BLS API
# ============================================================================

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

print(f"✓ Retrieved {len(df)} data points")
print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
print(f"\nData preview:")
print(df.head(10))
print(f"\nBasic statistics:")
print(df['unemployment_rate'].describe())



# ============================================================================
# PLOT WITH CONFIDENCE INTERVALS
# ============================================================================

if df is not None:
    print("\nCreating plot with confidence intervals...")
    
    # Calculate rolling statistics for confidence intervals
    df['rolling_mean'] = df['unemployment_rate'].rolling(window=12, center=True).mean()
    df['rolling_std'] = df['unemployment_rate'].rolling(window=12, center=True).std()
    
    # 95% confidence interval
    df['ci_upper'] = df['rolling_mean'] + 1.96 * df['rolling_std']
    df['ci_lower'] = df['rolling_mean'] - 1.96 * df['rolling_std']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot actual data
    ax.plot(df['year_month'], df['unemployment_rate'], 
            linewidth=2.5, color='#2E86AB', label='Actual Unemployment Rate', alpha=0.8)
    
    # Plot rolling mean
    ax.plot(df['year_month'], df['rolling_mean'], 
            linewidth=2, color='#A23B72', label='12-Month Rolling Mean', 
            linestyle='--', alpha=0.7)
    
    # Fill confidence interval
    ax.fill_between(df['year_month'], df['ci_lower'], df['ci_upper'], 
                     alpha=0.2, color='#2E86AB', label='95% Confidence Interval')
    
    # Styling
    ax.set_title('US Unemployment Rate (2005-2025) with Confidence Intervals', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Unemployment Rate (%)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotations for key events
    ax.axvspan(pd.Timestamp('2007-12-01'), pd.Timestamp('2009-06-01'), 
               alpha=0.1, color='red', label='Great Recession')
    ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-01'), 
               alpha=0.1, color='orange', label='COVID-19 Pandemic')
    
    plt.tight_layout()
    plt.savefig('us_unemployment_with_ci.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Plot saved as 'us_unemployment_with_ci.png'")
    print(f"\nKey Statistics:")
    print(f"  Mean: {df['unemployment_rate'].mean():.2f}%")
    print(f"  Std Dev: {df['unemployment_rate'].std():.2f}%")
    # print(f"  Min: {df['unemployment_rate'].min():.2f}% ({df.loc[df['unemployment_rate'].idxmin(), 'date'].strftime('%Y-%m')})")
    # print(f"  Max: {df['unemployment_rate'].max():.2f}% ({df.loc[df['unemployment_rate'].idxmax(), 'date'].strftime('%Y-%m')})")

else:
    print("Failed to retrieve data. Cannot create plot.")


# CREATOR FUNCTION TO LOAD DATAFRAME
def load_df():
    return df