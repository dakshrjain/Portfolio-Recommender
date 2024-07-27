import numpy as np
import pandas as pd

def fill_missing_values(row):
    # Fill returns_temp_5yr
    if pd.isna(row['returns_temp_5yr']):
        if not pd.isna(row['returns_temp_3yr']):
            row['returns_temp_5yr'] = row['returns_temp_3yr']
        elif not pd.isna(row['returns_1yr']):
            row['returns_temp_3yr'] = row['returns_1yr']
            row['returns_temp_5yr'] = row['returns_1yr']
    elif pd.isna(row['returns_temp_3yr']):
        if not pd.isna(row['returns_1yr']):
            row['returns_temp_3yr'] = row['returns_1yr']
    return row


# Apply function to DataFrame
def mutual_fund_suggestion(user_risk_category, user_time_period, equity_debt_gold_split, user_invest_money):
    datadir_final = 'F:/Masters/MRP/Datasets/'
    mf = pd.read_csv(datadir_final + 'Mutual_Funds_Data/mf_3.csv')
    mf['beta'] = mf['beta'].replace('-', 1)
    mf['beta_deviation'] = 1 - mf['beta'].astype(float)

    mf['sortino'].replace('-', np.nan, inplace=True)
    mf['sortino'] = mf['sortino'].astype(float)
    mean_value = mf['sortino'].mean()
    mf['sortino'].fillna(mean_value, inplace=True)

    risk_categories = ['Low Risk', 'Low to Medium Risk', 'Medium Risk', 'Medium to High Risk', 'High Risk',
                       'Very High Risk']
    if user_risk_category == 'High Risk':
        risk_category_int = risk_categories.index(user_risk_category) + 2
    else:
        risk_category_int = risk_categories.index(user_risk_category) + 1

    mf_temp = mf[mf['risk_level'] == risk_category_int]

    mf_temp['returns_temp_3yr'] = mf_temp['returns_3yr']
    mf_temp['returns_temp_5yr'] = mf_temp['returns_5yr']

    mf_temp = mf_temp.apply(fill_missing_values, axis=1)
    risk_free_return = 0.03
    if user_time_period <= 12:
        mf_temp['downside_risk'] = round(np.abs(((mf_temp['returns_1yr']/100) - risk_free_return)/mf_temp['sortino']) * 100, 2)
        if risk_category_int in [1,2,3]:
            sorted_mf = mf_temp.sort_values(by=['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha','sd', 'beta_deviation', 'sharpe', 'rating', 'returns_1yr', 'downside_risk'], ascending=[True, False, False, False, False, True, True, False, False, False, True])
        else:
            sorted_mf = mf_temp.sort_values(by=['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha','sd', 'beta_deviation', 'sharpe', 'rating', 'returns_1yr', 'downside_risk'], ascending=[True, False, False, False, False, False, True, False, False, False, True])
    elif user_time_period == 36:
        mf_temp['downside_risk'] = round(np.abs(((mf_temp['returns_temp_3yr']/100) - risk_free_return)/mf_temp['sortino']) * 100, 2)
        if risk_category_int in [1, 2, 3]:
            sorted_mf = mf_temp.sort_values(by=['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha', 'sd', 'beta_deviation', 'sharpe', 'rating', 'returns_3yr', 'downside_risk'], ascending=[True, False, False, False, False, True, True, False, False, False, True])
        else:
            sorted_mf = mf_temp.sort_values(by=['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha', 'sd', 'beta_deviation', 'sharpe','rating', 'returns_3yr', 'downside_risk'], ascending=[True, False, False, False, False, False, True, False, False, False, True])
    elif user_time_period == 60:
        mf_temp['downside_risk'] = round(np.abs(((mf_temp['returns_temp_5yr']/100) - risk_free_return)/mf_temp['sortino']) * 100, 2)
        if risk_category_int in [1, 2, 3]:
            sorted_mf = mf_temp.sort_values(by=['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha', 'sd', 'beta_deviation', 'sharpe', 'rating', 'returns_5yr', 'downside_risk'], ascending=[True, False, False, False, False, True, True, False, False, False, True])
        else:
            sorted_mf = mf_temp.sort_values(by=['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha', 'sd', 'beta_deviation', 'sharpe','rating', 'returns_5yr', 'downside_risk'], ascending=[True, False, False, False, False, False, True, False, False, False, True])

    sorted_mf['risk_level'] = user_risk_category
    columns = ['Fund Name', 'min_sip', 'min_lumpsum', 'Expense Ratio(%)', 'Fund Size(in Cr)', 'fund_age_yr',
               'fund_manager', 'sortino', 'alpha', 'sd', 'beta', 'sharpe', 'Risk', 'Fund House', 'rating', 'category', 'Sub Category', 'Returns(1yr)',
               'Returns(3yr)', 'Returns(5yr)', 'beta_deviation', 'returns_temp_3yr', 'returns_temp_5yr', 'Expected Downside Risk(%)']
    sorted_mf.columns = columns

    debt_funds = sorted_mf[sorted_mf['category'] == 'Debt']
    debt_funds = debt_funds[debt_funds['min_sip'] <= user_invest_money*(equity_debt_gold_split[2]/100)]
    hybrid_funds = sorted_mf[sorted_mf['category'] == 'Hybrid']
    hybrid_funds = hybrid_funds[hybrid_funds['min_sip'] <= user_invest_money*(equity_debt_gold_split[6]/100)]
    equity_funds = sorted_mf[sorted_mf['category'] == 'Equity']
    equity_funds = equity_funds[equity_funds['min_sip'] <= user_invest_money*(equity_debt_gold_split[7]/100)]

    return debt_funds, hybrid_funds, equity_funds

# print(mutual_fund_suggestion('Low Risk', 12, [10, 20, 70, 5, 0, 0, 5, 0], 100000))