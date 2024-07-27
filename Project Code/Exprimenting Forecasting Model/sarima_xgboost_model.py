import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

datadir_final = 'F:/Masters/MRP/Datasets/'

df = pd.read_csv(datadir_final+"Large_cap_equities/500049.csv")


# Preprocess data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df_train = df[:-21]
df_test = df[-21:]
# Select the features for SARIMA
# features = ['Open', 'High', 'Low', 'Close', 'Volume', 'No. of Trades']  # Add or modify features as needed
features = ['Close']
data = df_train[features]

# Fit the SARIMA model for each feature
sarima_models = {}
for feature in features:
    model = SARIMAX(data[feature], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) #p=1, d=1, q=1, P=1, D=1, Q=1, s=12 for monthly seasonality
    model_fit = model.fit(disp=False)
    sarima_models[feature] = model_fit
    print(f'Model summary for {feature}:')
    print(model_fit.summary())

# Generate residuals
residuals = {}
for feature in features:
    residuals[feature] = sarima_models[feature].resid
    plt.figure(figsize=(10, 6))
    plt.plot(residuals[feature])
    plt.title(f'Residuals for {feature}')
    plt.show()

# Train xgboost model on residuals
xgboost = {}
for feature in features:
    X = np.arange(len(residuals[feature])).reshape(-1, 1)
    y = residuals[feature].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    xgboost[feature] = xgb_model

# Combine SARIMA and ML forecasts
forecast_period = 21  # active trading days in a month
final_forecasts = {}

for feature in features:
    # Forecast using SARIMA
    sarima_forecast = sarima_models[feature].get_forecast(steps=forecast_period)
    sarima_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='B')
    sarima_predicted_mean = sarima_forecast.predicted_mean

    # Forecast residuals using the XGBoost model
    future_steps = np.arange(len(data) + 1, len(data) + forecast_period + 1).reshape(-1, 1)
    ml_forecast = xgboost[feature].predict(future_steps)

    # Combine the forecasts
    combined_forecast = sarima_predicted_mean + ml_forecast
    final_forecasts[feature] = combined_forecast

    combined_forecast = np.array(combined_forecast)
    actual = np.array(df_test[feature])

    # Evaluate model
    rmse = np.sqrt(mean_squared_error(actual, combined_forecast))
    print(f'RMSE: {rmse}')

    # Mean Forecast Error (MFE)
    mfe = np.mean(actual - combined_forecast)
    print(f'MFE: {mfe}')

    # Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(actual - combined_forecast))
    print(f'MAD: {mad}')

    # Cumulative Forecast Error (CFE)
    cfe = np.sum(actual - combined_forecast)
    print(f'CFE: {cfe}')

    # Tracking Signal
    tracking_signal = cfe / mad
    print(f'Tracking Signal: {tracking_signal}')

    # Coverage Probability
    coverage_prob = np.mean((combined_forecast >= df['Low'][-21:]) & (combined_forecast <= df['High'][-21:]))
    print(f'Coverage Probability: {coverage_prob}')

    # Plot the combined forecast
    plt.figure(figsize=(10, 6))
    plt.plot(df_test.index, df_test[feature], label='Actual')
    plt.plot(df_test.index, combined_forecast, label='Forecast')
    plt.title(f'Combined Forecast for {feature}')
    plt.legend()
    plt.show()
