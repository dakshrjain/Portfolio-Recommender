import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

datadir_final = 'F:/Masters/MRP/Datasets/'

df = pd.read_csv(datadir_final+"Large_cap_equities/500002.csv")

# Preprocess data
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df.set_index('Date', inplace=True)

df_train = df[:-21]
df_test = df[-21:]
# Select the features for SARIMA
# features = ['Open', 'High', 'Low', 'Close', 'Volume', 'No. of Trades']  # Add or modify features as needed
features = ['Close']
data = df[features]

returns = df_test['Close'].pct_change().dropna()

        # Define the confidence level
confidence_level = 0.95

        # Fit an EVT model (e.g., Generalized Pareto Distribution)
tail_threshold = returns.quantile(1 - confidence_level)
tail_returns = returns[returns <= tail_threshold]

        # Fit the Generalized Pareto Distribution (GPD)
shape, loc, scale = stats.genpareto.fit(tail_returns)

        # Calculate VaR
var = stats.genpareto.ppf(1 - confidence_level, shape, loc, scale)

print(var)

# # Fit the SARIMA model for each feature
# sarima_models = {}
# for feature in features:
#     model = SARIMAX(data[feature], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) #p=1, d=1, q=1, P=1, D=1, Q=1, s=12 for monthly seasonality
#     model_fit = model.fit(disp=False)
#     sarima_models[feature] = model_fit
#     print(f'Model summary for {feature}:')
#     print(model_fit.summary())
#
# # Generating residuals and fit GARCH model
# garch_models = {}
# for feature in features:
#     residuals = sarima_models[feature].resid
#     plt.figure(figsize=(10, 6))
#     plt.plot(residuals)
#     plt.title(f'Residuals for {feature}')
#     plt.show()
#
#     # Fit GARCH model
#     garch_model = arch_model(residuals, vol='GARCH', p=1, q=1) #p=1, q=1
#     garch_fit = garch_model.fit(disp='off')
#     garch_models[feature] = garch_fit
#     print(f'GARCH model summary for {feature}:')
#     print(garch_fit.summary())
#
# # Forecasting
# forecast_period = 21  # active trading days in a month
# final_forecasts = {}
#
# for feature in features:
#     # Forecasting using SARIMA
#     sarima_forecast = sarima_models[feature].get_forecast(steps=forecast_period)
#     sarima_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='B')
#     sarima_predicted_mean = sarima_forecast.predicted_mean
#
#     # Forecasting volatility using GARCH
#     garch_forecast = garch_models[feature].forecast(horizon=forecast_period)
#     garch_predicted_volatility = np.sqrt(garch_forecast.variance.values[-1, :])
#
#     # Combine the forecasts
#     combined_forecast = sarima_predicted_mean + np.sqrt(garch_predicted_volatility)
#     final_forecasts[feature] = combined_forecast
#
#
#     combined_forecast = np.array(combined_forecast)
#     actual = np.array(df_test[feature])
#
#     # Evaluate model
#     rmse = np.sqrt(mean_squared_error(actual, combined_forecast))
#     print(f'RMSE: {rmse}')
#
#     # Mean Forecast Error (MFE)
#     mfe = np.mean(actual - combined_forecast)
#     print(f'MFE: {mfe}')
#
#     # Mean Absolute Deviation (MAD)
#     mad = np.mean(np.abs(actual - combined_forecast))
#     print(f'MAD: {mad}')
#
#     # Cumulative Forecast Error (CFE)
#     cfe = np.sum(actual - combined_forecast)
#     print(f'CFE: {cfe}')
#
#     # Tracking Signal
#     tracking_signal = cfe / mad
#     print(f'Tracking Signal: {tracking_signal}')
#
#     # Coverage Probability
#     coverage_prob = np.mean((combined_forecast >= df['Low'][-21:]) & (combined_forecast <= df['High'][-21:]))
#     print(f'Coverage Probability: {coverage_prob}')
#
#     # Plot the combined forecast
#     plt.figure(figsize=(10, 6))
#     plt.plot(df_test.index, df_test[feature], label='Actual')
#     plt.plot(df_test.index, combined_forecast, label='Forecast')
#     plt.title(f'Combined Forecast for {feature}')
#     plt.legend()
#     plt.show()