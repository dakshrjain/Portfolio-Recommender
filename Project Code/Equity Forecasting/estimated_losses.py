import pandas as pd
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

datadir_final = 'F:/Masters/MRP/Datasets/'
datadir_prediction = 'F:/Masters/MRP/Prediction_Datasets/'

metadata = pd.read_csv(datadir_final+"all_stocks_metadata.csv")

estimated_losses = []
for index, rows in metadata.iterrows():
    print('Calculating Estimated Losses for row no.', index, ':', rows['Company Name'])
    if rows['Market Cap'] == 'Large':
        filename = 'Large_cap_equities/' + str(rows['Bse']) + '.csv'
    elif rows['Market Cap'] == 'Mid':
        filename = 'Mid_cap_equities/' + str(rows['Bse']) + '.csv'
    elif rows['Market Cap'] == 'Small':
        filename = 'Small_cap_equities/' + str(rows['Bse']) + '.csv'
    equity = pd.read_csv(datadir_final + filename)
    equity['Date'] = pd.to_datetime(equity['Date'], format='mixed')
    equity.set_index('Date', inplace=True)

    time_loss = [rows['Bse']]
    for days in [21, 63, 252, 756, 1260]:
        equity_slice = equity[-days:]
        returns = equity_slice['Close'].pct_change().dropna()

        # Define the confidence level
        confidence_level = 0.95

        # Fit an EVT model (e.g., Generalized Pareto Distribution)
        tail_threshold = returns.quantile(1 - confidence_level)
        tail_returns = returns[returns <= tail_threshold]

        # Fit the Generalized Pareto Distribution (GPD)
        shape, loc, scale = stats.genpareto.fit(tail_returns)

        # Calculate VaR
        var = stats.genpareto.ppf(1 - confidence_level, shape, loc, scale)

        time_loss.append(var)

    estimated_losses.append(time_loss)

estimated_losses_df = pd.DataFrame(estimated_losses, columns=['BSE', '1_month', '3_month', '12_month', '36_month', '60_month'])
estimated_losses_df.to_csv(datadir_prediction+'estimated_losses.csv')
