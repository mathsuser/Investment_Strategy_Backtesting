#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The main function. This is where we do the calculations on a set of inputs

"""


import warnings
warnings.filterwarnings('ignore')
import utils as utils
import datetime as dt
from datetime import date, timedelta
import pandas as pd

def main():
    
    # Set the inputs:

    Size = 18 # The size of the backtesting window in years

    # Calculate the end date for the backtesting period, avoiding the last day to prevent API issues
    end_date = date.today() - timedelta(days=7) # Try to avoid the last day because of possible API issues

    # Calculate the start date by subtracting the backtesting window size from the end date
    start_date = end_date.replace(year=end_date.year - Size)

    # Define the base currency for the financial data
    base_currency = 'USD'

    # Print the calculated start and end dates for verification
    print("Start date: ", start_date)
    print("End date: ", end_date)

    # Define the inflation curve to be used in the analysis
    inflation_curve = 'CPIAUCSL'

    # List of tickers for the assets to be analyzed
    Tickers = ['SPY', 'CSSMI.SW', 'ISF.L', 'IAU', 'BTC-USD'] # 'SPY' corresponds to the S&P500 distributing index ETF and 'IAU' corresponds to the Index Physical Gold Trust ETF. Both quotes in the USD. The third ticker stands for Bitcoin.

    # List of foreign exchange tickers for currency analysis
    FX_Tickers = ['CHFUSD=X', 'GBPUSD=X']
    
    
    initial_investment = 10000  # Example initial investment

    # Initialize dictionaries to store price data, dividend data, and foreign exchange (FX) data for each ticker
    prices_dfs = {}
    dividends_dfs = {}
    FX_dfs = {}

    # Loop through each ticker in the list of Tickers to fetch historical price and dividend data
    for ticker in Tickers:
        print()  # Print a new line for better readability in the output
        print(f"Fetching data for {ticker}...\n")  # Indicate which ticker's data is being fetched
        # Fetch historical price data for the current ticker and store it in the prices_dfs dictionary
        prices_dfs[ticker] = utils.fetch_historical_data(ticker, start_date, end_date)
        # Fetch historical dividend data for the current ticker and store it in the dividends_dfs dictionary
        dividends_dfs[ticker] = utils.fetch_historical_dividends(ticker, start_date, end_date)

    # Loop through each ticker in the list of FX_Tickers to fetch foreign exchange data
    for ticker in FX_Tickers:
        print()  # Print a new line for better readability in the output
        print(f"Fetching data for {ticker}...\n")  # Indicate which FX ticker's data is being fetched
        # Fetch foreign exchange data for the current FX ticker and store it in the FX_dfs dictionary
        FX_dfs[ticker] = utils.fetch_FX_data(ticker, start_date, end_date)

    # Fetch inflation data based on the specified inflation curve and date range
    inflation_df = utils.fetch_inflation_data(inflation_curve, start_date, end_date)
    

    # Printing the first rows of the dfs

    for ticker, prices_df in prices_dfs.items():
        print(f"Prices for {ticker}:")
        print(prices_df.head())

    for ticker, dividends_df in dividends_dfs.items():
        print(f"Dividends for {ticker}:")
        print(dividends_df.head())
    for ticker, FX_df in FX_dfs.items():
        print(f"FX for {ticker}:")
        print(FX_df.head())

    print("CPI :\n")
    print(inflation_df.head())
    
    
    # Convert the prices of securitied quoted in a currency other than the base currency using the FX rates.

    # For UK securities, prices are rescaled to GBP if they are quoted in GBps before the conversion to the base currency.

    
    transformed_prices_dfs, transformed_dividends_dfs = utils.uniform_price_and_dividend_data(Tickers, prices_dfs, dividends_dfs, FX_dfs, base_currency)
    for ticker, prices_df in transformed_prices_dfs.items():
        print(f"Prices for {ticker}:")
        print(prices_df.head())
        
        
    # Calculate daily and monthly returns


    daily_returns_df, periodic_returns_df = utils.calculate_returns(Tickers, transformed_prices_dfs, transformed_dividends_dfs)
    print(daily_returns_df.head())
    print(periodic_returns_df.head())

    monthly_returns_df = periodic_returns_df.copy()
    Total_Returns_df =daily_returns_df.copy()    
    
    # Calculate inflation rates 
    
    inflation_rate_data = utils.calculate_inflation_rate(inflation_curve, inflation_df)
    print(inflation_rate_data.head())

    #We merge inflation rate with securtiries returns.

    merged_df = pd.merge(inflation_rate_data, monthly_returns_df, on='Date', how='inner')
    merged_df = merged_df.dropna(subset=['InflationRate'])
    monthly_data = merged_df.copy()
    returns_df = merged_df.copy()
    print(monthly_data.head())
    
    
    """We calculate the cumulative returns:
    - Cumulative nominal returns
    - Cumulative inflation
    - Cumlative real returns
    - Compound growth of the initial investment
    """


    returns_df = utils.calculate_real_returns(Tickers,inflation_rate_data, monthly_returns_df)


    cumulative_returns_df = utils.calculate_cumulative_returns(Tickers, returns_df)
  
    value_df = utils.calculate_value_of_investment(initial_investment, cumulative_returns_df, Tickers)

    # Plotting the results
    
    utils.plot_cumulative_returns(Tickers, cumulative_returns_df)
    
    utils.plot_investment_growth(Tickers, value_df)
    
    Tickers_no_bitcoin = Tickers.copy()
    Tickers_no_bitcoin.remove('BTC-USD')
    utils.plot_real_returns(Tickers_no_bitcoin, cumulative_returns_df)
    utils.plot_real_returns(['BTC-USD'], cumulative_returns_df)
    
    


if __name__ == "__main__":
    main()    







