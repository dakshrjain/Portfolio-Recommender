import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
datadir_final = 'F:/Masters/MRP/Datasets/'

df = pd.read_csv(datadir_final+"Large_cap_equities/500049.csv")

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'WAP', 'Volume', 'No. of Trades',
            'Total Turnover (Rs.)', 'Deliverable Quantity', '% Deli. Qty to Traded Qty',
            'Spread High-Low', 'Spread Close-Open']
target = 'Close'

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Creating lagged features for time series forecasting
time_step = 2000

X = []
y = []

for i in range(time_step, len(scaled_features)):
    X.append(scaled_features[i-time_step:i])
    y.append(df[target].values[i])

X = np.array(X)
y = np.array(y)

# print(X,y)

# Train-test split
X_train, X_test, y_train, y_test = X[:-21], X[-21:], y[:-21], y[-21:]
# Flatten the input for SVM
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Train the SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train_flattened, y_train)

# Make predictions
svm_predictions = svm_model.predict(X_test_flattened)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test,svm_predictions))
print(f'RMSE: {rmse}')

# Mean Forecast Error (MFE)
mfe = np.mean(y_test - svm_predictions)
print(f'MFE: {mfe}')

# Mean Absolute Deviation (MAD)
mad = np.mean(np.abs(y_test - svm_predictions))
print(f'MAD: {mad}')

# Cumulative Forecast Error (CFE)
cfe = np.sum(y_test - svm_predictions)
print(f'CFE: {cfe}')

# Tracking Signal
tracking_signal = cfe / mad
print(f'Tracking Signal: {tracking_signal}')

# Coverage Probability
coverage_prob = np.mean((svm_predictions >= df['Low'][-21:]) & (svm_predictions <= df['High'][-21:]))
print(f'Coverage Probability: {coverage_prob}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(y_test):], y_test, color='blue', label='Actual')
plt.plot(df.index[-len(y_test):], svm_predictions, color='orange', label='Forecast')
plt.title('Forecast for Close')
plt.legend()
plt.show()
