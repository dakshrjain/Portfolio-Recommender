import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def equity_forecasting(df, forecast_period):
    # Preprocess data
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df.set_index('Date', inplace=True)

    #selecting features for forecast
    features = ['Open', 'High', 'Low', 'Close']
    data = df[features]

    # Fit the SARIMA model for each feature
    sarima_models = {}
    for feature in features:
        model = SARIMAX(data[feature], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) #p=1, d=1, q=1, P=1, D=1, Q=1, s=12 for monthly seasonality
        model_fit = model.fit(disp=False)
        sarima_models[feature] = model_fit

    # Generating residuals and fit GARCH, Random Forest model and LGBM
    rf = {}
    garch_models = {}
    lgbm = {}
    for feature in features:
        residuals = sarima_models[feature].resid

        # Fit GARCH model
        garch_model = arch_model(residuals, vol='GARCH', p=1, q=1) #p=1, q=1
        garch_fit = garch_model.fit(disp='off')
        garch_models[feature] = garch_fit

        #Fitting Random Forest
        X = np.arange(len(residuals)).reshape(-1, 1)
        y = residuals.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
        rf_model = RandomForestRegressor(n_estimators=250, random_state=12)
        rf_model.fit(X_train, y_train)
        rf[feature] = rf_model

        #Fitting LGBM model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
        lgbm_model = LGBMRegressor(n_estimators=250, random_state=57, verbose=-1)
        lgbm_model.fit(X_train, y_train)
        lgbm[feature] = lgbm_model

    # Forecasting
    final_forecasts = {}
    for feature in features:
        # Forecasting using SARIMA
        sarima_forecast = sarima_models[feature].get_forecast(steps=forecast_period)
        sarima_predicted_mean = sarima_forecast.predicted_mean

        # Forecasting volatility using GARCH
        garch_forecast = garch_models[feature].forecast(horizon=forecast_period)
        garch_predicted_volatility = garch_forecast.variance.values[-1, :]

        #Forecasting using Random forest
        future_steps = np.arange(len(data) + 1, len(data) + forecast_period + 1).reshape(-1, 1)
        rf_forecast = rf[feature].predict(future_steps)
        lgbm_forecast = lgbm[feature].predict(future_steps)

        # Combine the forecasts
        combined_forecast = sarima_predicted_mean + np.sqrt(garch_predicted_volatility) + rf_forecast + lgbm_forecast
        final_forecasts[feature] = combined_forecast

    forecast_df = pd.DataFrame({
        'Open': np.array(final_forecasts['Open']),
        'High': np.array(final_forecasts['High']),
        'Low': np.array(final_forecasts['Low']),
        'Close': np.array(final_forecasts['Close'])
    }, index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='B'))

    return forecast_df
