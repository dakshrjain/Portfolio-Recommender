import pandas as pd
from equity_forecasting_model import equity_forecasting

datadir_final = 'F:/Masters/MRP/Datasets/'
datadir_prediction = 'F:/Masters/MRP/Prediction_Datasets/'

metadata = pd.read_csv(datadir_final+"all_stocks_metadata.csv")

forecast_period = 1260 #5 years business days
for index, row in metadata.iterrows():
    print('Forecasting for row no.', index, ':', row['Company Name'])
    if row['Market Cap'] == 'Small':
        filename = 'Small_cap_equities/' + str(row['Bse']) + '.csv'
    elif row['Market Cap'] == 'Large':
        filename = 'Large_cap_equities/' + str(row['Bse']) + '.csv'
    elif row['Market Cap'] == 'Mid':
        filename = 'Mid_cap_equities/' + str(row['Bse']) + '.csv'

    df = pd.read_csv(datadir_final+filename)
    predictions_df = equity_forecasting(df, forecast_period)
    predictions_df.to_csv(datadir_prediction+filename)

