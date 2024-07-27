import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

datadir_final = 'F:/Masters/MRP/Datasets/'

df = pd.read_csv(datadir_final+"Large_cap_equities/500049.csv")

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

df_train = df[:-21]
df_test = df[-21:]

print(len(df_test['High']))

# Select relevant columns for prediction
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'No. of Trades']  # Include additional features as needed
data = df_train[features]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), :])
        Y.append(data[i + time_step, 3])  # Predicting 'Close' price
    return np.array(X), np.array(Y)

time_step = 2000  # Look back period
X, Y = create_sequences(scaled_data, time_step)

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, len(features))))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, Y_train, batch_size=1, epochs=10)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], len(features)-1)), predictions), axis=1))[:, -1]

# Evaluate model
rmse = np.sqrt(mean_squared_error(Y_test,predictions))
print(f'RMSE: {rmse}')

# Mean Forecast Error (MFE)
mfe = np.mean(Y_test - predictions)
print(f'MFE: {mfe}')

# Mean Absolute Deviation (MAD)
mad = np.mean(np.abs(Y_test - predictions))
print(f'MAD: {mad}')

# Cumulative Forecast Error (CFE)
cfe = np.sum(Y_test - predictions)
print(f'CFE: {cfe}')

# Tracking Signal
tracking_signal = cfe / mad
print(f'Tracking Signal: {tracking_signal}')

# Coverage Probability
coverage_prob = np.mean((predictions >= X_test[2]) & (predictions <= X_test[1]))
print(f'Coverage Probability: {coverage_prob}')


# Predict future stock prices
def predict_future(model, data, time_step, future_days):
    temp_input = list(data[-time_step:])
    future_output = []
    for i in range(future_days):
        if len(temp_input) > time_step:
            temp_input.pop(0)
        temp_array = np.array(temp_input).reshape(1, time_step, len(features))
        predicted_value = model.predict(temp_array)[0, 0]
        future_output.append(predicted_value)
        temp_input.append(np.concatenate((temp_array[0, -1, :-1], [predicted_value]), axis=0))  # Append predicted value to temp_input
    return scaler.inverse_transform(np.concatenate((np.zeros((len(future_output), len(features)-1)), np.array(future_output).reshape(-1, 1)), axis=1))[:, -1]


# Predict for 1 month (assuming 21 trading days in a month)
future_predictions = predict_future(model, scaled_data, time_step, 21)
print(future_predictions)

plt.figure(figsize=(14, 7))
plt.plot(df_test.index, df_test['Close'], color='blue', label='Actual Stock Price')
plt.plot(df_test.index, future_predictions, color='orange', label='Predicted Stock Price')
plt.title('Forecast for Close')
plt.legend()
plt.show()



