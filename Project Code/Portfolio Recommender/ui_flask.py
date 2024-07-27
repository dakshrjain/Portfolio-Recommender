from flask import Flask, render_template, request
import plotly.graph_objs as go
import plotly.io as pio
from portfolio_recommender import get_portfolio_recommendations, generate_plot_results_equities, generate_plot_results_funds
import pandas as pd
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve form data
    user_gender = request.form.get('gender')
    user_car = request.form.get('car')
    user_property = request.form.get('property')
    user_martial_status = request.form.get('martial_status')
    user_house_type = request.form.get('house_type')
    user_age = request.form.get('age')
    user_employment_status = request.form.get('employment_status')
    user_annual_income = request.form.get('annual_income')
    user_occupation = request.form.get('occupation')
    user_education_level = request.form.get('education')
    user_family_members = request.form.get('family_members')
    user_invest_money = request.form.get('investing_amount')
    user_time_period = request.form.get('investing_time')

    (df_debt_mutual_funds, df_hybrid_mutual_funds, df_equity_mutual_funds, df_large_cap_equities, df_mid_cap_equities, df_small_cap_equities, gold_price,
     equity_debt_gold_split, gold_graph, portfolio_overview_chart, investing_amount) = get_portfolio_recommendations(str(user_gender), str(user_car), str(user_property), str(user_martial_status), str(user_house_type),
                                                             int(user_age), int(user_employment_status), int(user_annual_income), str(user_occupation),
                                                             str(user_education_level), int(user_family_members), int(user_invest_money), int(user_time_period))

    # Data for the pie chart
    labels = ['Gold', 'Debt', 'Large cap equities', 'Mid cap equities',
              'Small cap equities', 'Hybrid Mutual Funds', 'Equity Mutual Funds']
    values = equity_debt_gold_split[1:]
    colors = ['#FAC70A', '#00FF2A', '#0074FF', '#F300FF', '#FF2300', '#D8F4CC', '#B2CCDC']
    color = colors.copy()

    filtered_indices = [i for i, v in enumerate(values) if v == 0]

    for idx in reversed(filtered_indices):
        del labels[idx]
        del values[idx]
        del colors[idx]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3,
                                 hoverinfo='label+percent', textinfo='label', textfont_size=13,
                                 marker=dict(colors=colors, line=dict(color='#000000', width=2)), pull=[0] * len(labels))])

    # Increase the size of the chart
    fig.update_layout(
        width=700,  # Increase the width
        height=700,
        paper_bgcolor='#e6e9f0',
        plot_bgcolor='#e6e9f0'
    )

    datadir_results = 'F:/Masters/MRP/Results/Portfolio_recommender_results/'
    file_path = datadir_results + 'equity_debt_gold_split_chart.png'
    pio.write_image(fig, file_path, format='png')
    # Convert the Plotly figure to HTML
    graph_html = pio.to_html(fig, full_html=False)

    graphs = []
    graphs.append(base64.b64encode(gold_graph.getvalue()).decode('utf-8'))
    graphs.append(base64.b64encode(generate_plot_results_funds(df_debt_mutual_funds, color[1], 'Debt_funds_charts').getvalue()).decode('utf-8'))
    graphs.append(base64.b64encode(generate_plot_results_equities(df_large_cap_equities, color[2], 'Large_cap_equities_charts').getvalue()).decode('utf-8'))
    graphs.append(base64.b64encode(generate_plot_results_equities(df_mid_cap_equities, color[3], 'Mid_cap_equities_charts').getvalue()).decode('utf-8'))
    graphs.append(base64.b64encode(generate_plot_results_equities(df_small_cap_equities, color[4], 'Small_cap_equities_charts').getvalue()).decode('utf-8'))
    graphs.append(base64.b64encode(generate_plot_results_funds(df_hybrid_mutual_funds, color[5], 'Hybrid_funds_charts').getvalue()).decode('utf-8'))
    graphs.append(base64.b64encode(generate_plot_results_funds(df_equity_mutual_funds, color[6], 'Equity_funds_charts').getvalue()).decode('utf-8'))

    gold_df = pd.DataFrame([gold_price], columns=['Gold Price'])
    df_debt_investments = pd.read_csv(r"F:\Masters\MRP\Results\Debt_investments.csv")


    # Saving Results
    df_debt_investments.to_csv(datadir_results + 'debt_other_investments.csv')
    df_debt_mutual_funds.to_csv(datadir_results + 'debt_mutual_funds.csv')
    df_hybrid_mutual_funds.to_csv(datadir_results + 'hybrid_mutual_funds.csv')
    df_equity_mutual_funds.to_csv(datadir_results + 'equity_mutual_funds.csv')
    df_large_cap_equities.to_csv(datadir_results + 'large_cap_equities.csv')
    df_mid_cap_equities.to_csv(datadir_results + 'mid_cap_equities.csv')
    df_small_cap_equities.to_csv(datadir_results + 'small_cap_equities.csv')
    gold_df.to_csv(datadir_results + 'current_gold_price.csv')


    debt_investments_html = df_debt_investments[:10].to_html(classes='table table-striped',
                                                          index=False) if not df_debt_investments.empty else None
    debt_mutual_funds_html = df_debt_mutual_funds[:10].to_html(classes='table table-striped',
                                                          index=False) if not df_debt_mutual_funds.empty else None
    large_cap_equities_html = df_large_cap_equities[:10].to_html(classes='table table-striped',
                                                            index=False) if not df_large_cap_equities.empty else None
    hybrid_mutual_funds_html = df_hybrid_mutual_funds[:10].to_html(classes='table table-striped',
                                                              index=False) if not df_hybrid_mutual_funds.empty else None
    mid_cap_equities_html = df_mid_cap_equities[:10].to_html(classes='table table-striped',
                                                        index=False) if not df_mid_cap_equities.empty else None
    equity_mutual_funds_html = df_equity_mutual_funds[:10].to_html(classes='table table-striped',
                                                              index=False) if not df_equity_mutual_funds.empty else None
    small_cap_equities_html = df_small_cap_equities[:10].to_html(classes='table table-striped',
                                                            index=False) if not df_small_cap_equities.empty else None

    ratio_str = str(equity_debt_gold_split[0])+' : '+str(equity_debt_gold_split[2])+' : '+str(equity_debt_gold_split[1])
    return render_template('results.html', graph_html=graph_html,
                           debt_mutual_funds_html=debt_mutual_funds_html,
                           large_cap_equities_html=large_cap_equities_html,
                           hybrid_mutual_funds_html=hybrid_mutual_funds_html,
                           mid_cap_equities_html=mid_cap_equities_html,
                           equity_mutual_funds_html=equity_mutual_funds_html,
                           small_cap_equities_html=small_cap_equities_html,
                           debt_investments_html=debt_investments_html,
                           gold_price=gold_price,
                           equity_debt_gold_split=ratio_str,
                           colors=color,
                           graphs=graphs,
                           investing_amount=investing_amount,
                           portfolio_overview_chart=portfolio_overview_chart)

if __name__ == '__main__':
    app.run(debug=True)
