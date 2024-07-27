import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

datadir_final = 'F:/Masters/MRP/Datasets/'

df = pd.read_csv(datadir_final+"Large_cap_equities/500049.csv")

# Preprocess data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df_train = df[:-21]
df_test = df[-21:]


# Selecting the features for ARIMA
# features = ['Open', 'High', 'Low', 'Close', 'Volume', 'No. of Trades']  # we can add more features as required
features = ['Close']
data = df_train[features]

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print(f'ADF Statistic for {timeseries.name}:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')

# Check stationarity for each feature
for feature in features:
    check_stationarity(data[feature])

# Differencing to make the time series stationary
data_diff = data.diff().dropna()
for feature in features:
    check_stationarity(data_diff[feature])

# Determining ARIMA parameters using ACF and PACF plots
for feature in features:
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(data_diff[feature], ax=plt.gca(), lags=40)
    plt.subplot(122)
    plot_pacf(data_diff[feature], ax=plt.gca(), lags=40)
    plt.suptitle(f'{feature} ACF and PACF')
    plt.show()

# Fitting the ARIMA model for each feature
models = {}
for feature in features:
    model = ARIMA(data[feature], order=(1, 1, 1)) #p=1, d=1, q=1 from ACF and PACF plots
    model_fit = model.fit()
    models[feature] = model_fit
    print(f'Model summary for {feature}:')
    print(model_fit.summary())

# Forecast future values
forecast_period = 21  #21 trading days for 1 month
forecasts = {}

for feature in features:
    forecast = models[feature].forecast(steps=forecast_period)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='B')
    forecasts[feature] = forecast

    forecast = np.array(forecast)
    actual = np.array(df_test[feature])

    # Evaluate model
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    print(f'RMSE: {rmse}')

    # Mean Forecast Error (MFE)
    mfe = np.mean(actual - forecast)
    print(f'MFE: {mfe}')

    # Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(actual - forecast))
    print(f'MAD: {mad}')

    # Cumulative Forecast Error (CFE)
    cfe = np.sum(actual - forecast)
    print(f'CFE: {cfe}')

    # Tracking Signal
    tracking_signal = cfe / mad
    print(f'Tracking Signal: {tracking_signal}')

    # Coverage Probability
    coverage_prob = np.mean((forecast >= df['Low'][-21:]) & (forecast <= df['High'][-21:]))
    print(f'Coverage Probability: {coverage_prob}')

    plt.figure(figsize=(10, 6))
    plt.plot(df_test.index, actual, label='Actual')
    plt.plot(df_test.index, forecast, label='Forecast')
    plt.title(f'Forecast for {feature}')
    plt.legend()
    plt.show()