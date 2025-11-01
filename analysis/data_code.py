
import matplotlib.pyplot as plt
import numpy as np


# Installation instructions
"""
Install all required libraries with:

pip install numpy pandas matplotlib seaborn statsmodels scikit-learn tensorflow prophet deap mapie requests

Or use this complete command:
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn tensorflow==2.15.0 prophet deap mapie requests

For conda users:
conda install numpy pandas matplotlib seaborn statsmodels scikit-learn tensorflow -c conda-forge
pip install prophet deap mapie requests
"""

import sys
import subprocess

def check_and_install_packages():
    """Check and install required packages"""
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'statsmodels': 'statsmodels',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'prophet': 'prophet',
        'deap': 'deap',
        'mapie': 'mapie',
        'requests': 'requests'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("=" * 80)
        print("MISSING PACKAGES DETECTED")
        print("=" * 80)
        print(f"\nThe following packages need to be installed:")
        for pkg in missing_packages:
            print(f"  • {pkg}")
        
        print("\nAttempting to install missing packages...")
        print("-" * 80)
        
        for package in missing_packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
                print(f"  Please run: pip install {package}")
        
        print("-" * 80)
        print("\nPlease restart the script after installation.\n")
        return False
    
    return True

# Check packages before importing
if not check_and_install_packages():
    print("⚠ Please install missing packages and run the script again.")
    sys.exit(1)

# Now import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Suppress Prophet plotly warning
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)

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

# Genetic Algorithm
from deap import base, creator, tools, algorithms

# Conformal Prediction
try:
    from mapie.regression import MapieRegressor
    from mapie.subsample import BlockBootstrap
    MAPIE_AVAILABLE = True
except ImportError:
    try:
        # Try older MAPIE API
        from mapie.regression import MapieRegressor
        MAPIE_AVAILABLE = True
        BlockBootstrap = None
    except ImportError:
        print("⚠ MAPIE not available. Conformal prediction will be skipped.")
        MAPIE_AVAILABLE = False
        MapieRegressor = None
        BlockBootstrap = None

# For data retrieval - using World Bank API
import requests

print("✓ All libraries imported successfully!")

print("=" * 80)
print("BRAZIL UNEMPLOYMENT RATE FORECASTING")
print("=" * 80)

# ============================================================================
# 1. DATA RETRIEVAL
# ============================================================================

def get_brazil_unemployment():
    """
    Retrieve Brazil unemployment rate from World Bank API
    Indicator: SL.UEM.TOTL.ZS (Unemployment, total % of total labor force)
    """
    print("\n1. RETRIEVING DATA FROM WORLD BANK API")
    print("-" * 80)
    
    # World Bank API endpoint
    url = "http://api.worldbank.org/v2/country/BRA/indicator/SL.UEM.TOTL.ZS"
    params = {
        'format': 'json',
        'per_page': 500,
        'date': '2000:2024'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if len(data) > 1:
            records = data[1]
            
            # Parse data
            dates = []
            values = []
            
            for record in records:
                if record['value'] is not None:
                    dates.append(record['date'])
                    values.append(record['value'])
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'unemployment_rate': values
            })
            
            df = df.sort_values('date').reset_index(drop=True)
            df.set_index('date', inplace=True)
            
            print(f"✓ Retrieved {len(df)} data points")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"\nFirst few records:")
            print(df.head())
            print(f"\nLast few records:")
            print(df.tail())
            
            return df
    
    print("✗ Failed to retrieve data from World Bank API")
    return None

# Retrieve data
df = get_brazil_unemployment()

if df is None or len(df) < 20:
    print("\n⚠ Using synthetic data for demonstration")
    # Generate synthetic monthly data
    dates = pd.date_range(start='2010-01-01', end='2024-09-01', freq='MS')
    np.random.seed(42)
    trend = np.linspace(8, 12, len(dates))
    seasonal = 2 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12)
    noise = np.random.normal(0, 0.5, len(dates))
    values = trend + seasonal + noise
    
    df = pd.DataFrame({
        'unemployment_rate': values
    }, index=dates)

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("2. TIME SERIES EDA")
print("=" * 80)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Missing values
print(f"\nMissing values: {df.isnull().sum().values[0]}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Time series plot
axes[0, 0].plot(df.index, df['unemployment_rate'], linewidth=2)
axes[0, 0].set_title('Brazil Unemployment Rate Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Unemployment Rate (%)')
axes[0, 0].grid(True, alpha=0.3)


# Distribution
from scipy import stats
data_clean = df['unemployment_rate'].dropna()
density = stats.gaussian_kde(data_clean)
xs = np.linspace(data_clean.min(), data_clean.max(), 200)
axes[0, 1].plot(xs, density(xs), linewidth=2.5, color='#2E86AB')
axes[0, 1].fill_between(xs, density(xs), alpha=0.3, color='#2E86AB')
# Add rug plot to show all individual observations
axes[0, 1].plot(data_clean, np.zeros_like(data_clean), '|', color='black', 
                markersize=10, alpha=0.5, markeredgewidth=1.5)
axes[0, 1].set_title('Distribution of Unemployment Rate (Density)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Unemployment Rate (%)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].grid(True, alpha=0.3)

# ACF
plot_acf(df['unemployment_rate'].dropna(), lags=min(40, len(df)//2), ax=axes[1, 0])
axes[1, 0].set_title('Autocorrelation Function', fontsize=12, fontweight='bold')

# PACF
plot_pacf(df['unemployment_rate'].dropna(), lags=min(40, len(df)//2), ax=axes[1, 1])
axes[1, 1].set_title('Partial Autocorrelation Function', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
plt.show()  # Display the plot
print("\n✓ EDA plots saved as 'eda_plots.png'")

# # Seasonal Decomposition
# print("\n" + "-" * 80)
# print("SEASONAL DECOMPOSITION")
# print("-" * 80)

# # Use additive model
# period = min(12, len(df) // 2)  # Monthly seasonality
# if len(df) >= 24:
#     decomposition = seasonal_decompose(df['unemployment_rate'], model='additive', period=period)
    
#     fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    
#     decomposition.observed.plot(ax=axes[0], title='Observed')
#     axes[0].set_ylabel('Observed')
#     axes[0].grid(True, alpha=0.3)
    
#     decomposition.trend.plot(ax=axes[1], title='Trend')
#     axes[1].set_ylabel('Trend')
#     axes[1].grid(True, alpha=0.3)
    
#     decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
#     axes[2].set_ylabel('Seasonal')
#     axes[2].grid(True, alpha=0.3)
    
#     decomposition.resid.plot(ax=axes[3], title='Residual')
#     axes[3].set_ylabel('Residual')
#     axes[3].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('seasonal_decomposition.png', dpi=300, bbox_inches='tight')
#     plt.show()  # Display the plot
#     print("✓ Seasonal decomposition saved as 'seasonal_decomposition.png'")

# # Augmented Dickey-Fuller Test
# print("\n" + "-" * 80)
# print("STATIONARITY TEST (Augmented Dickey-Fuller)")
# print("-" * 80)

# adf_result = adfuller(df['unemployment_rate'].dropna())
# print(f'ADF Statistic: {adf_result[0]:.4f}')
# print(f'p-value: {adf_result[1]:.4f}')
# print(f'Critical Values:')
# for key, value in adf_result[4].items():
#     print(f'  {key}: {value:.4f}')

# if adf_result[1] < 0.05:
#     print("\n✓ Series is STATIONARY (p < 0.05)")
# else:
#     print("\n✗ Series is NON-STATIONARY (p >= 0.05)")
#     print("  Consider differencing for SARIMA model")

# # ============================================================================
# # 3. TRAIN-TEST SPLIT
# # ============================================================================

# print("\n" + "=" * 80)
# print("3. TRAIN-TEST SPLIT")
# print("=" * 80)

# # Use 80% for training, 20% for testing
# train_size = int(len(df) * 0.8)
# train = df.iloc[:train_size].copy()
# test = df.iloc[train_size:].copy()

# print(f"\nTotal samples: {len(df)}")
# print(f"Training samples: {len(train)} ({len(train)/len(df)*100:.1f}%)")
# print(f"Test samples: {len(test)} ({len(test)/len(df)*100:.1f}%)")
# print(f"\nTrain period: {train.index.min()} to {train.index.max()}")
# print(f"Test period: {test.index.min()} to {test.index.max()}")

# # ============================================================================
# # 4. MODEL TRAINING
# # ============================================================================

# print("\n" + "=" * 80)
# print("4. MODEL TRAINING AND FORECASTING")
# print("=" * 80)

# results = {}

# # ----------------------------------------------------------------------------
# # 4.1 SARIMA
# # ----------------------------------------------------------------------------

# print("\n" + "-" * 80)
# print("4.1 SARIMA MODEL")
# print("-" * 80)

# try:
#     # Auto-select parameters (simplified approach)
#     # In practice, use auto_arima or grid search
#     sarima_model = SARIMAX(
#         train['unemployment_rate'],
#         order=(1, 1, 1),  # (p, d, q)
#         seasonal_order=(1, 1, 1, 12),  # (P, D, Q, s)
#         enforce_stationarity=False,
#         enforce_invertibility=False
#     )
    
#     sarima_fit = sarima_model.fit(disp=False)
#     print("✓ SARIMA model fitted successfully")
    
#     # Forecast
#     sarima_pred = sarima_fit.forecast(steps=len(test))
#     results['SARIMA'] = sarima_pred.values
    
#     print(f"SARIMA AIC: {sarima_fit.aic:.2f}")
#     print(f"SARIMA BIC: {sarima_fit.bic:.2f}")
    
# except Exception as e:
#     print(f"✗ SARIMA failed: {str(e)}")
#     results['SARIMA'] = None

# # ----------------------------------------------------------------------------
# # 4.2 PROPHET
# # ----------------------------------------------------------------------------

# print("\n" + "-" * 80)
# print("4.2 PROPHET MODEL")
# print("-" * 80)

# try:
#     # Prepare data for Prophet
#     prophet_train = train.reset_index()
#     prophet_train.columns = ['ds', 'y']
    
#     prophet_model = Prophet(
#         yearly_seasonality=True,
#         weekly_seasonality=False,
#         daily_seasonality=False,
#         seasonality_mode='additive'
#     )
    
#     prophet_model.fit(prophet_train)
#     print("✓ Prophet model fitted successfully")
    
#     # Forecast
#     future = prophet_model.make_future_dataframe(periods=len(test), freq='MS')
#     prophet_forecast = prophet_model.predict(future)
#     prophet_pred = prophet_forecast['yhat'].iloc[-len(test):].values
#     results['Prophet'] = prophet_pred
    
# except Exception as e:
#     print(f"✗ Prophet failed: {str(e)}")
#     results['Prophet'] = None

# # ----------------------------------------------------------------------------
# # 4.3 GA-LSTM
# # ----------------------------------------------------------------------------

# print("\n" + "-" * 80)
# print("4.3 GA-LSTM MODEL (Genetic Algorithm optimized LSTM)")
# print("-" * 80)

# def create_sequences(data, lookback=12):
#     """Create sequences for LSTM"""
#     X, y = [], []
#     for i in range(lookback, len(data)):
#         X.append(data[i-lookback:i])
#         y.append(data[i])
#     return np.array(X), np.array(y)

# def create_lstm_model(units, dropout_rate, learning_rate, lookback):
#     """Create LSTM model with given hyperparameters"""
#     model = Sequential([
#         LSTM(units, activation='tanh', return_sequences=True, 
#              input_shape=(lookback, 1)),
#         Dropout(dropout_rate),
#         LSTM(units // 2, activation='tanh'),
#         Dropout(dropout_rate),
#         Dense(1)
#     ])
    
#     optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
#     return model

# def evaluate_lstm(individual):
#     """Fitness function for GA"""
#     units, dropout_rate, learning_rate = individual
#     units = int(units)
#     dropout_rate = max(0.0, min(0.5, dropout_rate))
#     learning_rate = max(0.0001, min(0.01, learning_rate))
    
#     try:
#         model = create_lstm_model(units, dropout_rate, learning_rate, lookback)
        
#         # Train with early stopping
#         early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
#         history = model.fit(
#             X_train_lstm, y_train_lstm,
#             epochs=30,
#             batch_size=16,
#             verbose=0,
#             callbacks=[early_stop]
#         )
        
#         # Evaluate
#         loss = model.evaluate(X_train_lstm, y_train_lstm, verbose=0)[0]
#         return (loss,)  # Return tuple for DEAP
    
#     except Exception as e:
#         return (float('inf'),)

# try:
#     # Prepare data
#     scaler = MinMaxScaler()
#     train_scaled = scaler.fit_transform(train[['unemployment_rate']])
    
#     lookback = 12
#     X_train_lstm, y_train_lstm = create_sequences(train_scaled, lookback)
#     X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
    
#     print(f"LSTM training data shape: {X_train_lstm.shape}")
    
#     # Genetic Algorithm setup
#     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMin)
    
#     toolbox = base.Toolbox()
#     toolbox.register("attr_units", np.random.randint, 32, 129)
#     toolbox.register("attr_dropout", np.random.uniform, 0.1, 0.4)
#     toolbox.register("attr_lr", np.random.uniform, 0.0001, 0.005)
    
#     toolbox.register("individual", tools.initCycle, creator.Individual,
#                      (toolbox.attr_units, toolbox.attr_dropout, toolbox.attr_lr), n=1)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
#     toolbox.register("evaluate", evaluate_lstm)
#     toolbox.register("mate", tools.cxBlend, alpha=0.5)
#     toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
#     toolbox.register("select", tools.selTournament, tournsize=3)
    
#     # Run GA (reduced population for speed)
#     print("Running Genetic Algorithm optimization...")
#     population = toolbox.population(n=5)  # Small population for demo
#     ngen = 3  # Few generations for demo
    
#     for gen in range(ngen):
#         offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
#         fits = list(map(toolbox.evaluate, offspring))
#         for fit, ind in zip(fits, offspring):
#             ind.fitness.values = fit
#         population = toolbox.select(offspring, k=len(population))
        
#         best = tools.selBest(population, k=1)[0]
#         print(f"  Generation {gen+1}/{ngen} - Best fitness: {best.fitness.values[0]:.4f}")
    
#     # Get best hyperparameters
#     best_individual = tools.selBest(population, k=1)[0]
#     best_units = int(best_individual[0])
#     best_dropout = max(0.0, min(0.5, best_individual[1]))
#     best_lr = max(0.0001, min(0.01, best_individual[2]))
    
#     print(f"\n✓ GA optimization complete")
#     print(f"Best hyperparameters:")
#     print(f"  Units: {best_units}")
#     print(f"  Dropout: {best_dropout:.4f}")
#     print(f"  Learning rate: {best_lr:.6f}")
    
#     # Train final model
#     print("\nTraining final GA-LSTM model...")
#     final_model = create_lstm_model(best_units, best_dropout, best_lr, lookback)
#     early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
#     final_model.fit(
#         X_train_lstm, y_train_lstm,
#         epochs=50,
#         batch_size=16,
#         verbose=0,
#         callbacks=[early_stop]
#     )
    
#     print("✓ GA-LSTM model trained successfully")
    
#     # Forecast
#     lstm_predictions = []
#     current_sequence = train_scaled[-lookback:].reshape(1, lookback, 1)
    
#     for _ in range(len(test)):
#         pred = final_model.predict(current_sequence, verbose=0)
#         lstm_predictions.append(pred[0, 0])
#         current_sequence = np.append(current_sequence[:, 1:, :], 
#                                      pred.reshape(1, 1, 1), axis=1)
    
#     lstm_predictions = scaler.inverse_transform(
#         np.array(lstm_predictions).reshape(-1, 1)
#     ).flatten()
    
#     results['GA-LSTM'] = lstm_predictions
    
# except Exception as e:
#     print(f"✗ GA-LSTM failed: {str(e)}")
#     results['GA-LSTM'] = None

# # ----------------------------------------------------------------------------
# # 4.4 CONFORMAL PREDICTION
# # ----------------------------------------------------------------------------

# print("\n" + "-" * 80)
# print("4.4 CONFORMAL PREDICTION (using MAPIE)")
# print("-" * 80)

# try:
#     from sklearn.ensemble import RandomForestRegressor
    
#     if not MAPIE_AVAILABLE:
#         print("⊘ Skipping Conformal Prediction (MAPIE not installed)")
#         print("  To enable: pip install mapie --upgrade")
#         results['Conformal'] = None
#         raise ImportError("MAPIE not available")
    
#     # Create lag features
#     def create_lag_features(data, n_lags=12):
#         df_lags = pd.DataFrame()
#         for i in range(1, n_lags + 1):
#             df_lags[f'lag_{i}'] = data.shift(i)
#         df_lags['target'] = data.values
#         return df_lags.dropna()
    
#     # Prepare data
#     train_lags = create_lag_features(train['unemployment_rate'], n_lags=12)
#     X_train_cp = train_lags.iloc[:, :-1].values
#     y_train_cp = train_lags.iloc[:, -1].values
    
#     # Base model
#     base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
#     # Try different MAPIE API versions
#     try:
#         # Newer MAPIE version with BlockBootstrap
#         if BlockBootstrap is not None:
#             mapie_model = MapieRegressor(
#                 estimator=base_model,
#                 cv=BlockBootstrap(n_resamplings=10, length=10, random_state=42),
#                 method="plus"
#             )
#         else:
#             raise AttributeError("BlockBootstrap not available")
#     except (AttributeError, TypeError):
#         # Older MAPIE version - use simple CV
#         print("  Using standard cross-validation (BlockBootstrap not available)")
#         mapie_model = MapieRegressor(
#             estimator=base_model,
#             cv=5,  # Simple k-fold
#             method="plus"
#         )
    
#     mapie_model.fit(X_train_cp, y_train_cp)
#     print("✓ Conformal prediction model fitted successfully")
    
#     # Forecast with prediction intervals
#     cp_predictions = []
#     cp_lower = []
#     cp_upper = []
    
#     # Use rolling window for prediction
#     current_data = train['unemployment_rate'].values.copy()
    
#     for _ in range(len(test)):
#         # Create features from last 12 values
#         X_pred = current_data[-12:].reshape(1, -1)
        
#         # Predict with confidence interval (90%)
#         y_pred, y_pis = mapie_model.predict(X_pred, alpha=0.1)
        
#         cp_predictions.append(y_pred[0])
#         cp_lower.append(y_pis[0, 0, 0])
#         cp_upper.append(y_pis[0, 1, 0])
        
#         # Update current data
#         current_data = np.append(current_data, y_pred[0])
    
#     results['Conformal'] = np.array(cp_predictions)
#     results['Conformal_lower'] = np.array(cp_lower)
#     results['Conformal_upper'] = np.array(cp_upper)
    
#     print(f"✓ Generated predictions with 90% confidence intervals")
    
# except Exception as e:
#     print(f"✗ Conformal prediction failed: {str(e)}")
#     results['Conformal'] = None

# # ============================================================================
# # 5. EVALUATION
# # ============================================================================

# print("\n" + "=" * 80)
# print("5. MODEL EVALUATION")
# print("=" * 80)

# def calculate_metrics(y_true, y_pred, model_name):
#     """Calculate regression metrics"""
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     r2 = r2_score(y_true, y_pred)
    
#     return {
#         'Model': model_name,
#         'RMSE': rmse,
#         'MAE': mae,
#         'MAPE': mape,
#         'R²': r2
#     }

# y_true = test['unemployment_rate'].values
# metrics_list = []

# for model_name, predictions in results.items():
#     if predictions is not None and 'lower' not in model_name and 'upper' not in model_name:
#         metrics = calculate_metrics(y_true, predictions, model_name)
#         metrics_list.append(metrics)

# metrics_df = pd.DataFrame(metrics_list)
# metrics_df = metrics_df.sort_values('RMSE')

# print("\nModel Performance Metrics:")
# print(metrics_df.to_string(index=False))

# # Save metrics
# metrics_df.to_csv('model_metrics.csv', index=False)
# print("\n✓ Metrics saved to 'model_metrics.csv'")

# # ============================================================================
# # 6. VISUALIZATION
# # ============================================================================

# print("\n" + "=" * 80)
# print("6. RESULTS VISUALIZATION")
# print("=" * 80)

# fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# # Plot 1: All predictions
# ax1 = axes[0]
# ax1.plot(train.index, train['unemployment_rate'], 
#          label='Training Data', linewidth=2, color='blue', alpha=0.7)
# ax1.plot(test.index, test['unemployment_rate'], 
#          label='Actual Test Data', linewidth=2, color='black', linestyle='--')

# colors = ['red', 'green', 'orange', 'purple']
# for i, (model_name, predictions) in enumerate(results.items()):
#     if predictions is not None and 'lower' not in model_name and 'upper' not in model_name:
#         ax1.plot(test.index, predictions, 
#                 label=f'{model_name} Forecast', 
#                 linewidth=2, color=colors[i % len(colors)], alpha=0.7)

# ax1.set_title('Unemployment Rate Forecasts - All Models', fontsize=14, fontweight='bold')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Unemployment Rate (%)')
# ax1.legend(loc='best')
# ax1.grid(True, alpha=0.3)

# # Plot 2: Conformal prediction with intervals
# ax2 = axes[1]
# ax2.plot(train.index, train['unemployment_rate'], 
#          label='Training Data', linewidth=2, color='blue', alpha=0.7)
# ax2.plot(test.index, test['unemployment_rate'], 
#          label='Actual Test Data', linewidth=2, color='black', linestyle='--')

# if results.get('Conformal') is not None:
#     ax2.plot(test.index, results['Conformal'], 
#             label='Conformal Prediction', linewidth=2, color='purple')
    
#     if 'Conformal_lower' in results:
#         ax2.fill_between(test.index, 
#                         results['Conformal_lower'], 
#                         results['Conformal_upper'],
#                         alpha=0.3, color='purple', 
#                         label='90% Confidence Interval')

# ax2.set_title('Conformal Prediction with Uncertainty Intervals', 
#              fontsize=14, fontweight='bold')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Unemployment Rate (%)')
# ax2.legend(loc='best')
# ax2.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('forecast_results.png', dpi=300, bbox_inches='tight')
# plt.show()  # Display the plot
# print("\n✓ Forecast visualization saved as 'forecast_results.png'")

# # ============================================================================
# # 7. SUMMARY
# # ============================================================================

# print("\n" + "=" * 80)
# print("7. SUMMARY")
# print("=" * 80)

# print(f"\n📊 Dataset: Brazil Unemployment Rate")
# print(f"   Total periods: {len(df)}")
# print(f"   Training: {len(train)} | Test: {len(test)}")

# print(f"\n🏆 Best Model: {metrics_df.iloc[0]['Model']}")
# print(f"   RMSE: {metrics_df.iloc[0]['RMSE']:.4f}")
# print(f"   MAE: {metrics_df.iloc[0]['MAE']:.4f}")
# print(f"   MAPE: {metrics_df.iloc[0]['MAPE']:.2f}%")
# print(f"   R²: {metrics_df.iloc[0]['R²']:.4f}")

# print(f"\n📁 Generated Files:")
# print(f"   • eda_plots.png - Exploratory data analysis")
# print(f"   • seasonal_decomposition.png - Time series decomposition")
# print(f"   • forecast_results.png - Model predictions comparison")
# print(f"   • model_metrics.csv - Performance metrics table")

# print("\n" + "=" * 80)
# print("ANALYSIS COMPLETE!")
# print("=" * 80)