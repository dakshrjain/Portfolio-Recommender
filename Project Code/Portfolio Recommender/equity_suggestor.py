import pandas as pd
import numpy as np
def equity_suggestion(investing_period,equity_debt_gold_split,user_invest_money):
    datadir_final = 'F:/Masters/MRP/Datasets/'
    datadir_prediction = 'F:/Masters/MRP/Prediction_Datasets/'
    filename = ''
    metadata = pd.read_csv(datadir_final+'all_stocks_metadata.csv')
    results = []
    for index,rows in metadata.iterrows():
        if rows['Market Cap'] == 'Large':
            filename = 'Large_cap_equities/'+str(rows['Bse'])+'.csv'
        elif rows['Market Cap'] == 'Mid':
            filename = 'Mid_cap_equities/'+str(rows['Bse'])+'.csv'
        elif rows['Market Cap'] == 'Small':
            filename = 'Small_cap_equities/'+str(rows['Bse'])+'.csv'
        predicted_equity = pd.read_csv(datadir_prediction+filename)
        predicted_close_price = np.round(predicted_equity.iloc[(investing_period*21)-1]['Close'],2)
        historical_equity = pd.read_csv(datadir_final+filename)
        starting_close_price = historical_equity.iloc[len(historical_equity)-1]['Close']
        if predicted_close_price > starting_close_price:
            expected_return = ((predicted_close_price - starting_close_price)/starting_close_price)*100
        else:
            if predicted_close_price < 0:
                predicted_close_price = 0
            expected_return = -((starting_close_price - predicted_close_price)/starting_close_price)*100


        results.append([rows['Company Name'], rows['Industry'], rows['Symbol'], rows['Market Cap'], rows['Bse'], starting_close_price,
                        predicted_close_price, np.round(expected_return, 2)])

    equity_results = pd.DataFrame(results, columns=['Company Name', 'Industry', 'Symbol', 'Market Cap', 'Ticker Symbol(BSE)', 'Current Close(in ₹)',
                                                    'Expected Close(in ₹)', 'Expected Return(%)'])

    expected_downside_return_df = pd.read_csv(datadir_prediction + 'estimated_losses.csv')
    column = str(investing_period)+'_month'
    equity_results['Estimated Loss(%)'] = np.round(np.abs(expected_downside_return_df[column]) * 100, 2)

    large_cap_equity_results = []
    mid_cap_equity_results = []
    small_cap_equity_results = []
    large_cap_equities = pd.DataFrame([], columns=equity_results.columns)
    small_cap_equities = pd.DataFrame([], columns=equity_results.columns)
    mid_cap_equities = pd.DataFrame([], columns=equity_results.columns)
    if equity_debt_gold_split[3] != 0:
        temp = equity_results[equity_results['Market Cap'] == 'Large'].sort_values(by=['Expected Return(%)', 'Estimated Loss(%)'], ascending=[False, True])
        for index, rows in temp.iterrows():
            if rows['Current Close(in ₹)'] < user_invest_money * (equity_debt_gold_split[3] / 100) and rows['Expected Return(%)'] > 0.1 and rows['Estimated Loss(%)'] < 25:
                large_cap_equity_results.append(rows)
        large_cap_equities = pd.DataFrame(large_cap_equity_results, columns=temp.columns)
    if equity_debt_gold_split[4] != 0:
        temp = equity_results[equity_results['Market Cap'] == 'Mid'].sort_values(by=['Expected Return(%)', 'Estimated Loss(%)'], ascending=[False, True])
        for index, rows in temp.iterrows():
            if rows['Current Close(in ₹)'] < user_invest_money * (equity_debt_gold_split[3] / 100) and rows['Expected Return(%)'] > 0.1 and rows['Estimated Loss(%)'] < 50:
                mid_cap_equity_results.append(rows)
        mid_cap_equities = pd.DataFrame(mid_cap_equity_results, columns=temp.columns)
    if equity_debt_gold_split[5] != 0:
        temp = equity_results[equity_results['Market Cap'] == 'Small'].sort_values(by=['Expected Return(%)', 'Estimated Loss(%)'], ascending=[False, True])
        for index, rows in temp.iterrows():
            if rows['Current Close(in ₹)'] < user_invest_money * (equity_debt_gold_split[3] / 100) and rows['Expected Return(%)'] > 0.1 and rows['Estimated Loss(%)'] < 75:
                small_cap_equity_results.append(rows)
        small_cap_equities = pd.DataFrame(small_cap_equity_results, columns=temp.columns)

    columns = ['Company Name', 'Industry', 'Symbol', 'Ticker Symbol(BSE)', 'Current Close(in ₹)', 'Expected Close(in ₹)', 'Expected Return(%)', 'Estimated Loss(%)']
    return large_cap_equities[columns], mid_cap_equities[columns], small_cap_equities[columns]

# print(equity_suggestion(36, [10, 20, 70, 5, 0, 0, 5, 0], 100000))