import pickle
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from equity_suggestor import equity_suggestion #function calculating the returns based on the requested time periods
from mutual_fund_suggestor import mutual_fund_suggestion #function searching for top mutual funds based on an individuals profile
from current_gold_price import get_gold_price_inr #Function to get the current price of gold being traded in INR
import warnings
import base64
warnings.filterwarnings("ignore")

#Loading the pickle file containing the Gradient Boosting Machine model to predict the financial risk tolerance score of an individual
with open('gbm_income.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

#Function to calculate cost_of_living
def calulate_cost_of_living(income, family_members):
    return int(income/family_members)


#Function to calculate Occupation_Category based on occupaton, imported from income_data_model.ipynb file
def get_occupation_category(occupation):
    professional = ['Managers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'HR staff', 'IT staff']
    skilled = ['Sales staff', 'Core staff', 'Security staff', 'Cooking staff', 'Private service staff', 'Secretaries', 'Realty agents']
    unskilled = ['Laborers', 'Drivers', 'Cleaning staff', 'Low-skill Laborers', 'Waiters/barmen staff']
    if occupation in professional:
        return 'Professional/Management'
    elif occupation in skilled:
        return 'Skilled Labour'
    elif occupation in unskilled:
        return 'Unskilled Labour'


#Function to calculate Education_Category based on Education level, imported from income_data_model.ipynb file
def get_education_category(education_level):
    higher = ['Higher education', 'Academic degree']
    secondary = ['Secondary / secondary special', 'Incomplete higher']
    lower = ['Lower secondary']
    if education_level in higher:
        return 'Higher Education'
    elif education_level in secondary:
        return 'Secondary Education'
    elif education_level in lower:
        return 'Lower Secondary Education'


#Function to calculate risk category, imported from income_data_model.ipynb file
def calculate_risk_category(score):
    if score >= 1250:
        return 'Very High Risk'
    elif score < 1250 and score >= 1100:
        return 'High Risk'
    elif score < 1100 and score >= 950:
        return 'Medium to High Risk'
    elif score < 950 and score >= 850:
        return 'Medium Risk'
    elif score < 850 and score >= 700:
        return 'Low to Medium Risk'
    elif score < 700:
        return 'Low Risk'


#Function to map real values to catogorical, imported from income_data_model.ipynb file
def mapper_to_categorical(data_row):
    gender = ['M', 'F']
    car = ['Y', 'N']
    prop = ['Y', 'N']
    martial = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
    house = ['House / apartment', 'With parents', 'Rented apartment', 'Municipal apartment', 'Co-op apartment', 'Office apartment']
    occupation = ['Unskilled Labour', 'Professional/Management', 'Skilled Labour']
    education = ['Higher Education', 'Secondary Education', 'Lower Secondary Education']
    return [gender.index(data_row[0]), car.index(data_row[1]), prop.index(data_row[2]),
            martial.index(data_row[3]), house.index(data_row[4]), data_row[5], data_row[6],
            data_row[7], occupation.index(data_row[8]), education.index(data_row[9])]


user_gender = 'M'
user_car = 'Y'
user_property = 'Y'
user_martial_status = 'Married'
user_house_type = 'House / apartment'
user_age = 25
user_employment_status = 1
user_annual_income = 5000000
user_occupation = 'IT staff'
user_education_level = 'Academic degree'
user_family_members = 2
user_invest_money = 100000
user_time_period = 36


#Function to get the portfolio recommendations
def get_portfolio_recommendations(user_gender, user_car, user_property, user_martial_status, user_house_type, user_age, user_employment_status, user_annual_income, user_occupation, user_education_level, user_family_members, user_invest_money, user_time_period):
    user_cost_of_living = calulate_cost_of_living(user_annual_income, user_family_members)
    user_occupation_category = get_occupation_category(user_occupation)
    user_education_category = get_education_category(user_education_level)

    #input_format = [Gender, Car_owner, Protperty_owner, House Type, Age, Employed, Cost of Living, Occupation_category, Education_category]
    user_input = [user_gender, user_car, user_property, user_martial_status, user_house_type, user_age, user_employment_status, user_cost_of_living, user_occupation_category, user_education_category]
    # print(user_input)
    user_input_categorical = mapper_to_categorical(user_input)
    user_input_categorical = np.array(user_input_categorical)
    # print(user_input_categorical)
    user_input_categorical = user_input_categorical.reshape(1, -1)

    predicted_score = gbm_model.predict(user_input_categorical)
    predicted_financial_risk_tolerance_score = np.round(predicted_score[0]).astype(int)

    user_risk_category = calculate_risk_category(predicted_financial_risk_tolerance_score)
    # print(user_risk_category)

    def equity_debt_gold_ratio(risk_category):
        if risk_category == 'Low Risk':
            return [10, 20, 70, 5, 0, 0, 5, 0] #format [equity_ratio, gold_ratio, debt ratio, Large cap equities, Mid cap, Small cap, Hybrid MF, Equity MF]
        elif risk_category == 'Low to Medium Risk':
            return [25, 15, 60, 10, 5, 0, 10, 0]
        elif risk_category == 'Medium Risk':
            return [40, 10, 50, 15, 10, 5, 10, 0]
        elif risk_category == 'Medium to High Risk':
            return [45, 10, 45, 15, 10, 10, 0, 10]
        elif risk_category == 'High Risk':
            return [65, 5, 30, 15, 20, 25, 0, 5]
        elif risk_category == 'Very High Risk':
            return [80, 5, 15, 20, 25, 30, 0, 5]


    equity_debt_gold_split = equity_debt_gold_ratio(user_risk_category)

    large_cap_equity_results, mid_cap_equity_results, small_cap_equity_results = equity_suggestion(user_time_period, equity_debt_gold_split, user_invest_money)
    # print(large_cap_equity_results[:10], mid_cap_equity_results[:10], small_cap_equity_results[:10])

    debt_funds_results, hybrid_funds_results, equity_funds_results = mutual_fund_suggestion(user_risk_category, user_time_period, equity_debt_gold_split, user_invest_money)
    # print(debt_funds_results[:10], hybrid_funds_results[:10], equity_funds_results[:10])

    current_gold_price, gold_graph = get_gold_price_inr()

    img = io.BytesIO()
    gold_graph.savefig(img, format='png')
    img.seek(0)

    investing_amount = []
    for i in equity_debt_gold_split[1:]:
        investing_amount.append(round((int(user_invest_money)*i)/100))

    portfolio_overview_chart = calculate_risk_return_investment(user_time_period, investing_amount, debt_funds_results, equity_funds_results,
                                     hybrid_funds_results, large_cap_equity_results, mid_cap_equity_results,
                                     small_cap_equity_results)
    columns_to_print = ['Fund House', 'Fund Name', 'Expense Ratio(%)', 'Fund Size(in Cr)', 'Risk', 'Sub Category',
                        'Returns(1yr)', 'Returns(3yr)', 'Returns(5yr)', 'Expected Downside Risk(%)']

    return debt_funds_results[columns_to_print], hybrid_funds_results[columns_to_print], equity_funds_results[columns_to_print], large_cap_equity_results, mid_cap_equity_results, small_cap_equity_results, current_gold_price, equity_debt_gold_split, img, portfolio_overview_chart, investing_amount


# get_portfolio_recommendations(user_gender, user_car, user_property, user_martial_status, user_house_type, user_age, user_employment_status, user_annual_income, user_occupation, user_education_level, user_family_members, user_invest_money, user_time_period)
def generate_plot_results_equities(dataset, color, filename):

    # Load the data
    data = dataset[:10]

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    color = color+'1A'
    fig.patch.set_facecolor(color)
    # Bar Chart
    sns.barplot(ax=axs[0], x='Symbol', y='Expected Return(%)', data=data, palette='viridis')
    axs[0].set_title('Expected Return by Company')
    axs[0].set_xlabel('Company Symbol')
    axs[0].set_ylabel('Expected Return (%)')
    axs[0].tick_params(axis='x', rotation=90)

    # Scatter Plot
    scatter = sns.scatterplot(ax=axs[1], x='Estimated Loss(%)', y='Expected Return(%)', hue='Symbol',
                              size='Expected Return(%)',
                              data=data, palette='viridis', sizes=(30, 300))
    axs[1].set_title('Estimated Loss vs Expected Return')
    axs[1].set_xlabel('Estimated Loss(%)')
    axs[1].set_ylabel('Expected Return (%)')
    axs[1].legend(title='Industry')

    # Group by Industry and sum the expected returns for the Pie Chart
    industry_returns = dataset[:20].groupby('Industry')['Expected Return(%)'].sum()

    # Pie Chart
    colors = sns.color_palette('viridis', len(industry_returns))
    wedges, texts, autotexts = axs[2].pie(industry_returns, labels=industry_returns.index, autopct='%1.1f%%',
                                          startangle=140, colors=colors, wedgeprops=dict(width=0.9),
                                          pctdistance=0.5, labeldistance=0.9)

    # Format the labels
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    axs[2].set_title('Distribution of Expected Returns by Industry')

    # Adjust layout
    plt.subplots_adjust(wspace=0.3)  # Adjust spacing between subplots
    plt.tight_layout()

    datadir_results = 'F:/Masters/MRP/Results/Portfolio_recommender_results/'
    plt.savefig(datadir_results + filename + '.png')

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)

    return img


def generate_plot_results_funds(dataset, color, filename):
    data = dataset[:10]

    fig, axs = plt.subplots(1, 3, figsize=(40, 6))
    color = color+'1A'
    fig.patch.set_facecolor(color)  # Set figure background color

    # Bar Chart for Expense Ratio
    sns.barplot(ax=axs[0], x='Expense Ratio(%)', y='Fund Name', data=data, palette='viridis')
    axs[0].set_title('Expense Ratio of Mutual Funds')
    axs[0].set_xlabel('Expense Ratio (%)')
    axs[0].set_ylabel('Fund Name')
    axs[0].tick_params(axis='x', rotation=45)

    # Line Plot for Returns Over Time
    for fund_name in data['Fund Name'].unique():
        fund_data = data[data['Fund Name'] == fund_name]
        axs[1].plot(['1yr', '3yr', '5yr'], [fund_data['Returns(1yr)'].values[0],
                                            fund_data['Returns(3yr)'].values[0],
                                            fund_data['Returns(5yr)'].values[0]],
                    marker='o', label=fund_name, markersize=8)

    axs[1].set_title('Returns Over Time for Different Mutual Funds', fontsize=10)
    axs[1].set_xlabel('Time Period', fontsize=10)
    axs[1].set_ylabel('Returns (%)', fontsize=10)
    axs[1].legend(title='Fund Name', fontsize=6)

    # Bar Chart for Fund Size
    sns.barplot(ax=axs[2], x='Fund Size(in Cr)', y='Fund Name', data=data, palette='viridis')
    axs[2].set_title('Fund Size of Mutual Funds')
    axs[2].set_xlabel('Fund Size (in Cr)')
    axs[2].set_ylabel('Fund Name')
    axs[2].tick_params(axis='x', rotation=45)
    axs[2].yaxis.set_label_position('right')
    axs[2].yaxis.tick_right()

    plt.subplots_adjust(wspace=0.7)  # Adjust spacing between subplots
    plt.tight_layout()

    datadir_results = 'F:/Masters/MRP/Results/Portfolio_recommender_results/'
    plt.savefig(datadir_results + filename + '.png')

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)

    return img


def calculate_risk_return_investment(user_time_period, investing_amount, debt_funds_results, equity_funds_results, hybrid_funds_results, large_cap_equity_results, mid_cap_equity_results, small_cap_equity_results):
    expected_return_investment = 0
    estimated_loss_investment = 0
    if not debt_funds_results.empty:
        estimated_risk_mean = debt_funds_results[:3]['Expected Downside Risk(%)'].mean()
        if user_time_period <= 12:
            expected_return_mean = debt_funds_results[:3]['Returns(1yr)'].mean()
        if user_time_period == 36:
            expected_return_mean = debt_funds_results[:3]['returns_temp_3yr'].mean()
        if user_time_period == 60:
            expected_return_mean = debt_funds_results[:3]['returns_temp_5yr'].mean()
        estimated_loss_investment += ((investing_amount[1] * estimated_risk_mean) / 100)
        expected_return_investment += investing_amount[1] + ((investing_amount[1] * ((expected_return_mean+6)/2)) / 100)

    if not equity_funds_results.empty:
        estimated_risk_mean = equity_funds_results[:3]['Expected Downside Risk(%)'].mean()
        if user_time_period <= 12:
            expected_return_mean = equity_funds_results[:3]['Returns(1yr)'].mean()
        if user_time_period == 36:
            expected_return_mean = equity_funds_results[:3]['returns_temp_3yr'].mean()
        if user_time_period == 60:
            expected_return_mean = equity_funds_results[:3]['returns_temp_5yr'].mean()
        estimated_loss_investment += ((investing_amount[6] * estimated_risk_mean) / 100)
        expected_return_investment += investing_amount[6] + ((investing_amount[6] * expected_return_mean) / 100)

    if not hybrid_funds_results.empty:
        estimated_risk_mean = hybrid_funds_results[:3]['Expected Downside Risk(%)'].mean()
        if user_time_period <= 12:
            expected_return_mean = hybrid_funds_results[:3]['Returns(1yr)'].mean()
        if user_time_period == 36:
            expected_return_mean = hybrid_funds_results[:3]['returns_temp_3yr'].mean()
        if user_time_period == 60:
            expected_return_mean = hybrid_funds_results[:3]['returns_temp_5yr'].mean()
        estimated_loss_investment += ((investing_amount[5] * estimated_risk_mean) / 100)
        expected_return_investment += investing_amount[5] + ((investing_amount[5] * expected_return_mean) / 100)

    if not large_cap_equity_results.empty:
        expected_return_mean = large_cap_equity_results[:3]['Expected Return(%)'].mean()
        estimated_risk_mean = large_cap_equity_results[:3]['Estimated Loss(%)'].mean()
        expected_return_investment += investing_amount[2] + ((investing_amount[2] * expected_return_mean) / 100)
        estimated_loss_investment += ((investing_amount[2] * estimated_risk_mean) / 100)

    if not mid_cap_equity_results.empty:
        expected_return_mean = mid_cap_equity_results[:3]['Expected Return(%)'].mean()
        estimated_risk_mean = mid_cap_equity_results[:3]['Estimated Loss(%)'].mean()
        expected_return_investment += investing_amount[3] + ((investing_amount[3] * expected_return_mean) / 100)
        estimated_loss_investment += ((investing_amount[3] * estimated_risk_mean) / 100)

    if not small_cap_equity_results.empty:
        expected_return_mean = small_cap_equity_results[:3]['Expected Return(%)'].mean()
        estimated_risk_mean = small_cap_equity_results[:3]['Estimated Loss(%)'].mean()
        expected_return_investment += investing_amount[4] + ((investing_amount[4] * expected_return_mean) / 100)
        estimated_loss_investment += ((investing_amount[4] * estimated_risk_mean) / 100)

    initial_investment = sum(investing_amount)
    final_value = round(expected_return_investment, 2)
    max_loss = round(sum(investing_amount) - estimated_loss_investment, 2)

    # Time period (in months)
    time_period_months = user_time_period
    months = np.arange(0, time_period_months + 1)

    # Linear interpolation from initial to final value
    projected_values = np.linspace(initial_investment, final_value, time_period_months + 1)
    min_values = np.linspace(initial_investment, max_loss, time_period_months + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(months, projected_values, label='Projected Portfolio Value', color='green')
    ax.plot(months, min_values, label='Maximum Portfolio Loss', color='red', linestyle='--')

    ax.fill_between(months, initial_investment, projected_values, where=(projected_values >= initial_investment),
                     color='#90EE90', alpha=0.5, label='Growth Range')

    # Fill between initial investment and potential min value (red)
    ax.fill_between(months, min_values, initial_investment, where=(min_values < initial_investment), color='#FF6347',
                     alpha=0.5, label='Loss Range')
    # # Annotations with arrows
    ax.text(time_period_months, projected_values[-1], f'₹{final_value}',
             ha='right', va='bottom', fontsize=13, color='green')

    ax.text(time_period_months, min_values[-1], f'₹{max_loss}',
             ha='right', va='top', fontsize=13, color='red')
    fig.patch.set_facecolor('#e6e9f0')
    ax.set_xlabel('Time Period (Months)')
    ax.set_ylabel('Portfolio Value (in ₹)')
    ax.set_title('Projected Portfolio Value and Maximum Loss Over Time (Confidence Level = 95%)')
    ax.legend()
    ax.grid(True)

    datadir_results = 'F:/Masters/MRP/Results/Portfolio_recommender_results/'
    plt.savefig(datadir_results + 'Projected_portfolio_chart.png')

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

