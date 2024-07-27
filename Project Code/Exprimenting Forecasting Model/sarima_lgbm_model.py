import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

datadir_final = 'F:/Masters/MRP/Datasets/'
df = pd.read_csv(datadir_final+"Large_cap_equities/500325.csv")

# Preprocess data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df_train = df[:-21]
df_test = df[-21:]
# Selecting the features for ARIMA
# features = ['Open', 'High', 'Low', 'Close', 'Volume', 'No. of Trades']  # we can add more features as required
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

# Generating residuals from the SARIMA
residuals = {}
for feature in features:
    residuals[feature] = sarima_models[feature].resid
    plt.figure(figsize=(10, 6))
    plt.plot(residuals[feature])
    plt.title(f'Residuals for {feature}')
    plt.show()

# Train LGBM model on residuals
lgbm = {}
for feature in features:
    X = np.arange(len(residuals[feature])).reshape(-1, 1)
    y = residuals[feature].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)
    lgbm_model.fit(X_train, y_train)

    lgbm[feature] = lgbm_model

# Combine SARIMA and LGBM forecasts
forecast_period = 132  # one months active trading days
final_forecasts = {}

for feature in features:
    # Forecast using SARIMA
    sarima_forecast = sarima_models[feature].get_forecast(steps=forecast_period)
    sarima_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='B')
    sarima_predicted_mean = sarima_forecast.predicted_mean

    # Forecast residuals using the LGBM model
    future_steps = np.arange(len(data) + 1, len(data) + forecast_period + 1).reshape(-1, 1)
    ml_forecast = lgbm[feature].predict(future_steps)

    # Combining the forecasts
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
    plt.plot(df_test.index, actual, label='Actual')
    plt.plot(sarima_index, combined_forecast, label='Forecast')
    plt.title(f'Combined Forecast for {feature}')
    plt.legend()
    plt.show()
