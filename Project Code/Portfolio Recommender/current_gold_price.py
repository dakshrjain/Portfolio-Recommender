import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

file_path = r"C:\Users\daksh\PycharmProjects\MRP\api_key_quandl.txt"
file = open(file_path, 'r')
file_content_1 = file.read()
file.close()

API_KEY = file_content_1
BASE_URL = 'https://www.quandl.com/api/v3/datasets/WGC/GOLD_DAILY_INR.json'

file_path = r"C:\Users\daksh\PycharmProjects\MRP\api_key_alphavantage.txt"
file = open(file_path, 'r')
file_content_2 = file.read()
file.close()

api_key = file_content_2
# Function to fetch current gold price in INR
def get_gold_price_inr():
    params = {
        'api_key': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    # Extract current price from the API response
    current_price = data['dataset']['data'][0][1]  # Assuming the latest data is first


    # Calculate the date range for the previous month
    today = datetime.today()
    start_date = (today - timedelta(days=63)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    # Set the Alpha Vantage API endpoint and parameters for gold price in USD
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'XAUUSD',
        'apikey': api_key
    }

    # Make the API request
    response = requests.get(url, params=params)

    # Check the status code and response content
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        print(response.text)

    # Extract and process the data
    if 'Time Series (Daily)' in data:
        time_series = data['Time Series (Daily)']
        dates = []
        gold_prices_usd = []

        for date, price_info in time_series.items():
            if start_date <= date <= end_date:
                dates.append(date)
                gold_prices_usd.append(float(price_info['4. close']))

        # Create a DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'USD_Price': gold_prices_usd
        })

        # Fetch the exchange rate (USD to INR) for the last day of the previous month
        exchange_rate_url = 'https://api.exchangerate-api.com/v4/latest/USD'
        exchange_rate_response = requests.get(exchange_rate_url)
        if exchange_rate_response.status_code == 200:
            exchange_rate_data = exchange_rate_response.json()
            exchange_rate = exchange_rate_data['rates']['INR']

            df['INR_Price'] = df['USD_Price'] * exchange_rate

            subtraction_value = df.iloc[0]['INR_Price'] - current_price
            df['INR_Price'] = df['INR_Price'] - subtraction_value
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            fig, ax = plt.subplots(figsize=(10, 6))

            # Set the background color
            fig.patch.set_facecolor('#FAC70A1A')  # Figure background color

            # Plotting
            ax.plot(df.index, df['INR_Price'], label='Gold', color='Gold')
            ax.set_title('Previous 3 months GOLD Prices')
            ax.legend()

            datadir_results = 'F:/Masters/MRP/Results/Portfolio_recommender_results/'
            plt.savefig(datadir_results + 'gold_price_chart.png')

            return current_price, fig

        else:
            print(f"Failed to fetch exchange rate: {exchange_rate_response.status_code}")
            print(exchange_rate_response.text)
    else:
        print("No data found in the response.")
