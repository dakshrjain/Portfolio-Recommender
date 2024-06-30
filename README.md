# Portfolio-Recommender
In today's fast-paced and interconnected world, financial markets are complex and dynamic, posing significant challenges to investors. My Major Research Project (MRP) addresses these challenges by developing a personalized portfolio recommender based on individuals' financial risk tolerance and investment capacity.

# Aim 
Portfolio Recommender and Optimizer for new Investors based on Financial Risk Tolerance of individuals.

# Problem Statement 
In today's fast-paced and highly interconnected world, the financial markets represent a dynamic ecosystem where investors seek to make informed decisions in the pursuit of maximizing returns while managing risks effectively. But the sheer volume and complexity of financial data, coupled with the unpredictable nature of the market dynamics, pose significant challenges to investors. So there is a requirement for a personalized portfolio optimizer based on individuals' Financial Risk Tolerance level and ability to invest in these volatile and fast growing markets.

# Project Desciption
The system is largely divided into three parts, Financial Risk Assessment Model, Top Performing Assets and Tailored Investment Portfolio Recommendations
1. Financial Risk Assessment Model
   
    ● Utilizing demographic data like Income, Education, Marital Status, Children, Age, Family members and many more factors, our system will develop a Risk Tolerance score for each individual in our dataset.

    ● This score will categorize individuals into distinct risk-taking capacity groups, aiding in personalized investment recommendations.

2. Top-Performing Assets
   
    ● Our system will analyze historical data from the Bombay Stock Exchange (BSE), India's largest exchange. Using data from 2014-2020, we will assess the performance of 300 equities and approximately 2000 mutual funds across various categories.

    ● Performance evaluation will be conducted for different timeframes (1, 3, and 5 years) and across different equity and mutual funds categories like Large/Mid/Small Market Capitalization equities and Equity/Debt/Hybrid/Solution-oriented mutual funds.

3. Tailored Investment Portfolio Recommendations
   
    ● Individuals interested in investing will provide information about their basic demographics like Income, Age, Marital Status, Family Members, Property Owner, Amount to invest and time for which the investment has to be done.

    ● Based on this information, our system will calculate an individual's Financial risk tolerance score and then recommend an appropriate equity-to-debt split ratio for their investment portfolio.

      
    ● And we will go one more step deeper and recommend to them how they can split their equity part into Small/Mid/Large Cap Funds and provide the details for the top performing equities in that particular cap funds for the time the investor wants to invest their money.
   
    ● For instance, if the system suggests a 70:30 equity-to-debt split ratio, it will further advise on the allocation of large, mid, and small-cap funds within the equity portion along with the top performing equities in each market cap and recommend top performing Debt funds for the debt portion for the same timeframe the investor wants to invest.
   
# Datasets
1. Individual Income Dataset for assessing Financial Risk of individuals:

    ● https://www.kaggle.com/datasets/vishwas199728/credit-card
   
2. Mutual Funds Dataset for getting top performing funds:
   
    ● https://www.kaggle.com/datasets/ravibarnawal/mutual-fund-of-india-dataset-2023
   
    ● https://www.kaggle.com/datasets/abhigyandatta/mutual-funds-india
   
    ● https://www.kaggle.com/datasets/ravibarnawal/mutual-funds-india-detailed
   
3. Equities Dataset for getting top performing equities:
   
    For equities, I am planning to take 300 equities wherein 100 will be Large Cap, 100 will be Mid Cap and 100 will be Small Cap from the year 2014-2020.
   
    ● Large Cap (I am taking 50 stocks of “NIFTY50” index and 50 stocks of “SENSEX50” index):

     ❖ NIFTY50 : https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data?select=ADANIPORTS.csv

     ❖ SENSEX50 (I will use the official BSE Website and will download the 50 stocks by name from the Historical Data available on the website): https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.html?flag=0.
   
    ● Mid Cap (I am going to use the same BSE website and get the list of top 100 performing mid cap equities and download them by names):

   ❖ https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.html?flag=0
   
    ● Small Cap (I am going to use the same BSE website and get the list of top 100 performing small cap equities and download them by names)

   ❖ https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.html?flag=0
   

   
