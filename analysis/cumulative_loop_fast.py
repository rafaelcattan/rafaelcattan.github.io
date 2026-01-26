"""
Fast cumulative yearly training loop (limited to first 5 years) for SARIMA, Prophet, and RandomForest+MAPIE.
Outputs RMSE for each iteration.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sys

# Import required libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from mapie.regression import MapieRegressor
from mapie.subsample import BlockBootstrap
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Load the data (same as notebook)
def load_data():
    import requests
    mykey = 'ff7095746d954b8a884141069e4216e9'
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": ["LNS14000000"],
        "startyear": "2005",
        "endyear": "2026",
        "registrationKey": mykey
    }
    r = requests.post(url, json=payload)
    data = r.json()["Results"]["series"][0]["data"]
    df = pd.DataFrame(data)
    df["unemployment_rate"] = df["value"].astype(float)
    df['year_month'] = df['year'] + '-' + df['period'].str[1:]
    df['year_month'] = pd.to_datetime(df['year_month'])
    df = df.sort_values('year_month').reset_index(drop=True)
    df.set_index('year_month', inplace=True)
    return df

# Load data
print("Loading data from BLS API...")
df = load_data()
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Determine total years
total_months = len(df)
total_years = total_months // 12
print(f"Total years: {total_years}")

# Limit iterations for speed
max_iterations = total_years - 1
print(f"Running cumulative loop for first {max_iterations} years...")

# Prepare results DataFrame
results = []

# Define helper functions for each model (same as notebook)

def train_sarima(train_series, test_length):
    """Train SARIMA on train_series and forecast test_length steps."""
    # Use auto_arima to find optimal orders (as in notebook)
    auto_model = auto_arima(
        train_series,
        seasonal=True,
        m=12,
        stepwise=True,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        max_p=5,
        max_q=5,
        max_P=2,
        max_Q=2,
        max_d=2,
        max_D=1
    )
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order
    
    # Fit SARIMA
    sarima_model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_results = sarima_model.fit(disp=False)
    forecast = sarima_results.get_forecast(steps=test_length)
    pred = forecast.predicted_mean
    return pred

def train_prophet(train_df, test_dates):
    """Train Prophet on train_df and forecast for test_dates."""
    prophet_train = train_df.reset_index()[['year_month', 'unemployment_rate']]
    prophet_train.columns = ['ds', 'y']
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.fit(prophet_train)
    
    future = pd.DataFrame({'ds': test_dates})
    forecast = model.predict(future)
    pred = forecast.set_index('ds')['yhat']
    return pred

def create_lag_features(data, n_lags=12):
    """Create lag features for RF+MAPIE."""
    df_lags = pd.DataFrame()
    for i in range(1, n_lags + 1):
        df_lags[f'lag_{i}'] = data.shift(i)
    df_lags['target'] = data.values
    return df_lags.dropna()

def recursive_rf_forecast(model, last_values, n_steps, n_lags):
    """Recursive multi-step forecast for lag-based models."""
    history = list(last_values)
    preds = []
    for _ in range(n_steps):
        X = np.array(history[-n_lags:]).reshape(1, -1)
        y_pred = model.predict(X)[0]
        preds.append(y_pred)
        history.append(y_pred)
    return np.array(preds)

def train_rf_mapie(train_series, test_length, n_lags=12):
    """Train RandomForest+MAPIE on train_series and forecast test_length steps."""
    # Prepare lag features
    train_lags = create_lag_features(train_series, n_lags=n_lags)
    X_train = train_lags.iloc[:, :-1].values
    y_train = train_lags.iloc[:, -1].values
    
    # Base model
    base_model = RandomForestRegressor(
        n_estimators=500,
        random_state=42
    )
    
    # MAPIE with Block Bootstrap (time-series safe)
    mapie_model = MapieRegressor(
        estimator=base_model,
        cv=BlockBootstrap(n_resamplings=10, length=10, random_state=42),
        method="plus"
    )
    mapie_model.fit(X_train, y_train)
    
    # Last observed values from training set
    last_train_values = train_series.iloc[-n_lags:].values
    
    # Point forecast
    pred = recursive_rf_forecast(
        model=mapie_model,
        last_values=last_train_values,
        n_steps=test_length,
        n_lags=n_lags
    )
    return pred

# Main cumulative yearly loop
for year_idx in range(1, max_iterations + 1):
    # Determine cutoff date: first `year_idx` years
    start_date = df.index[0]
    cutoff_date = start_date + pd.DateOffset(years=year_idx)
    
    # Split data
    train = df.loc[df.index < cutoff_date].copy()
    test = df.loc[df.index >= cutoff_date].copy()
    
    print(f"\n--- Iteration {year_idx}: Train {len(train)} months ({year_idx} years), Test {len(test)} months ---")
    sys.stdout.flush()
    
    # SARIMA
    try:
        sarima_pred = train_sarima(train['unemployment_rate'], len(test))
        sarima_rmse = np.sqrt(mean_squared_error(test['unemployment_rate'], sarima_pred))
        print(f"  SARIMA RMSE: {sarima_rmse:.4f}")
    except Exception as e:
        print(f"  SARIMA failed: {e}")
        sarima_rmse = np.nan
    
    # Prophet
    try:
        prophet_pred = train_prophet(train, test.index)
        prophet_rmse = np.sqrt(mean_squared_error(test['unemployment_rate'], prophet_pred))
        print(f"  Prophet RMSE: {prophet_rmse:.4f}")
    except Exception as e:
        print(f"  Prophet failed: {e}")
        prophet_rmse = np.nan
    
    # RandomForest+MAPIE
    try:
        rf_pred = train_rf_mapie(train['unemployment_rate'], len(test))
        rf_rmse = np.sqrt(mean_squared_error(test['unemployment_rate'], rf_pred))
        print(f"  RF+MAPIE RMSE: {rf_rmse:.4f}")
    except Exception as e:
        print(f"  RF+MAPIE failed: {e}")
        rf_rmse = np.nan
    
    # Store results
    results.append({
        'train_years': year_idx,
        'train_months': len(train),
        'test_months': len(test),
        'SARIMA_RMSE': sarima_rmse,
        'Prophet_RMSE': prophet_rmse,
        'RF_MAPIE_RMSE': rf_rmse
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("CUMULATIVE TRAINING RMSE RESULTS (first 5 years)")
print("="*80)
print(results_df.to_string(index=False))

# Save results to CSV
out_path = Path(r'C:\Users\Dell\data_science\rafaelcattan.github.io\analysis\cumulative_rmse_fast.csv')

results_df.to_csv(out_path, index=False)

print("\nResults saved to 'analysis/cumulative_rmse_all.csv'")