#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
--------

Functions:
----------
    Fetching data:
        
        - fetch_inflation_data
        - fetch_historical_data
        - fetch_historical_dividends
        - fetch_FX_data
        
    Transforming data:
        
        - merge_price_dividends
        - get_price_and_dividend_currency
        - transform_price_currency
        - uniform_price_and_dividend_data
       
    Returns and inflation rates calculations
     
        - calculate_returns
        - calculate_inflation_rate
        - calculate_real_returns
        - calculate_cumulative_returns
        - calculate_value_of_investment
    
    Plotting and visualisation: 
        - plot_cumulative_returns
        - plot_investment_growth
        - plot_real_returns
      

"""



import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt



def fetch_inflation_data(inflation_curve, start_date, end_date):
    # Fetch data from FRED (Federal Reserve Economic Data)
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=' + inflation_curve
    CPI_data = pd.read_csv(url, index_col=0, parse_dates=True)
    CPI_data = CPI_data.loc[start_date:end_date]
    CPI_data = CPI_data.reset_index()
    inflation_df = CPI_data.copy()
    inflation_df = inflation_df.reset_index(drop=True)
    inflation_df.rename(columns={'DATE': 'Date'},inplace=True)

    return inflation_df



def fetch_historical_data(ticker, start_date, end_date):
    # Download historical data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date, actions=True)

    # Create a DataFrame with the date and adjusted closing prices
    prices_df = pd.DataFrame({'Date': data.index, 'Price': data['Adj Close']})

    # Reset the index of the DataFrame
    prices_df = prices_df.reset_index(drop=True)
    # Uncomment the line below to print the first few rows of the prices DataFrame for debugging
    # print(prices_df.head())
    return prices_df



def fetch_historical_dividends(ticker, start_date, end_date):
    # Download historical data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date, actions=True)

    # Filter the data to get only the rows where dividends are greater than 0
    dividends = data[data['Dividends'] > 0]

    # Create a DataFrame with the date and dividend amounts
    dividends_df = pd.DataFrame({'Date': dividends.index, 'Dividend': dividends['Dividends']})

    # Reset the index of the dividends DataFrame
    df_dividends = dividends_df.reset_index(drop=True)
    # Check if the dividends DataFrame is empty and print a message if so
    if dividends.empty:
        print("No dividends found for the selected period.")
    return df_dividends

# Function to fetch foreign exchange (FX) data for a given FX ticker within a specified date range
def fetch_FX_data(FX_ticker, start_date, end_date):
    # Download FX data using yfinance
    FX_data = yf.download(FX_ticker, start=start_date, end=end_date, actions=True)

    # Create a DataFrame with the date and FX rates
    FX_df = pd.DataFrame({'Date': FX_data.index, 'FX_rate': FX_data['Adj Close']})

    # Reset the index of the FX DataFrame
    FX_df = FX_df.reset_index(drop=True)
    return FX_df

# Function to merge price and dividend data for a given ticker within a specified date range
def merge_price_dividends(ticker, start_date, end_date):
    # Fetch historical price data
    price_data = fetch_historical_data(ticker, start_date, end_date)
    # Fetch historical dividend data
    dividends_data = fetch_historical_dividends(ticker, start_date, end_date)

    # If no dividends data is found, create an empty DataFrame with the appropriate columns
    if dividends_data is None:
        dividends_data = pd.DataFrame(columns=['Date', 'Dividend'])

    # Merge prices and dividends dataframes on date
    merged_df = pd.merge(price_data, dividends_data, on='Date', how='left')

    # Fill missing dividends with 0, since not every day will have a dividend
    merged_df['Dividend'] = merged_df['Dividend'].fillna(0)

def get_price_and_dividend_currency(ticker):
    # Fetch the ticker data
    stock = yf.Ticker(ticker)

    # Retrieve the stock's currency (for the price)
    price_currency = stock.info.get('currency', 'Currency not available')

    # Download the historical dividends
    dividends = stock.dividends

    # If dividends exist, fetch the currency from the 'Dividends' field
    if not dividends.empty:
        div_currency = stock.info.get('dividendCurrency', price_currency)  # Sometimes dividendCurrency is not available
    else:
        div_currency = "No dividends available"

    return price_currency, div_currency




def transform_price_currency(df, FX_df, base_currency, target_currency):
  """Transforms the price from one currency to another if different from base currency.

  Args:
    df: DataFrame containing the price data.
    FX_df: DataFrame containing the FX data for the currency transformation.
    base_currency: The original currency of the price.
    target_currency: The target currency to convert the price to.

  Returns:
    DataFrame with transformed prices if necessary, original prices otherwise.
  """

  if base_currency == target_currency:
    return df  # No transformation needed
  else:
    # Merge price and FX data on date
    merged_df = pd.merge(df, FX_df, on='Date', how='left')

    if 'Price' in df.columns:
      if target_currency == 'GBp':
        # Perform currency conversion if FX rate is available

        merged_df['Transformed Price'] = merged_df['Price'] * merged_df['FX_rate']

        merged_df['Transformed Price'] = merged_df['Transformed Price'] * 0.01

        transformed_prices_df = merged_df[['Date', 'Price', 'Transformed Price']].copy()
      else:
        # Perform currency conversion if FX rate is available
        merged_df['Transformed Price'] = merged_df['Price'] * merged_df['FX_rate']
        transformed_prices_df = merged_df[['Date', 'Price', 'Transformed Price']].copy()
    elif 'Dividend' in df.columns:
      if target_currency == 'GBp':
        # Perform currency conversion if FX rate is available
        merged_df['Transformed Dividend'] = 0.01* merged_df['Dividend'] * merged_df['FX_rate']
      else:
        # Perform currency conversion if FX rate is available
        merged_df['Transformed Dividend'] = merged_df['Dividend'] * merged_df['FX_rate']

      # Return only the transformed price, date, and keep the original price
      transformed_prices_df = merged_df[['Date', 'Dividend', 'Transformed Dividend']].copy()
    return transformed_prices_df


def uniform_price_and_dividend_data(Tickers, prices_dfs, dividends_dfs, FX_dfs, base_currency):
  transformed_prices_dfs = {}
  transformed_dividends_dfs = {}
  for ticker in Tickers:
    price_currency, div_currency = get_price_and_dividend_currency(ticker)
    if price_currency != base_currency:
      if price_currency == 'GBp':
        price_currency_bis = 'GBP'
        transformed_prices_df = transform_price_currency(prices_dfs[ticker], FX_dfs[f'{price_currency_bis}{base_currency}=X'], base_currency, price_currency)
        transformed_dividends_df = transform_price_currency(dividends_dfs[ticker], FX_dfs[f'{price_currency_bis}{base_currency}=X'], base_currency, price_currency)

        if 'Transformed Price' in transformed_prices_df.columns:
          transformed_prices_dfs[ticker] = transformed_prices_df[['Date', 'Transformed Price']].rename(columns={'Transformed Price': 'Price'})
        if 'Transformed Dividend' in transformed_dividends_df.columns:
          transformed_dividends_dfs[ticker] = transformed_dividends_df[['Date', 'Transformed Dividend']].rename(columns={'Transformed Dividend': 'Dividend'})

      else:
        transformed_prices_df = transform_price_currency(prices_dfs[ticker], FX_dfs[f'{price_currency}{base_currency}=X'], base_currency, price_currency)
        transformed_dividends_df = transform_price_currency(dividends_dfs[ticker], FX_dfs[f'{price_currency}{base_currency}=X'], base_currency, price_currency)

        if 'Transformed Price' in transformed_prices_df.columns:
          transformed_prices_dfs[ticker] = transformed_prices_df[['Date', 'Transformed Price']].rename(columns={'Transformed Price': 'Price'})
        if 'Transformed Dividend' in transformed_dividends_df.columns:
          transformed_dividends_dfs[ticker] = transformed_dividends_df[['Date', 'Transformed Dividend']].rename(columns={'Transformed Dividend': 'Dividend'})
    else:
      if price_currency == 'GBp':
        price_currency = 'GBP'
      transformed_prices_dfs[ticker] = prices_dfs[ticker]
      transformed_dividends_dfs[ticker] = dividends_dfs[ticker]
  return transformed_prices_dfs, transformed_dividends_dfs


def calculate_returns(Tickers ,prices_dfs, dividends_dfs, resampling_frequency = 'M'):

  #daily_returns = pd.DataFrame(columns =['Date'] + [f'{ticker}' for ticker in Tickers])
  daily_returns_df = pd.DataFrame()

  periodic_returns_df = pd.DataFrame(columns =[f'{ticker}' for ticker in Tickers])

  # Loop through each ticker to calculate total and monthly returns
  for ticker in Tickers:
      # Retrieve the price and dividend DataFrames for the current ticker
      prices_df = prices_dfs[ticker]
      dividends_df = dividends_dfs[ticker]


      # Merge the price and dividend DataFrames on the 'Date' column
      merged_df = pd.merge(prices_df, dividends_df, on='Date', how='left')

      # Fill any missing values in the 'Dividend' column with 0
      merged_df['Dividend'] = merged_df['Dividend'].fillna(0)

      # Calculate daily returns for the current ticker
      merged_df[f'{ticker}'] = (merged_df['Price'] + merged_df['Dividend']) / merged_df['Price'].shift(1) - 1

      # Create a DataFrame with Date and the calculated returns for this ticker
      returns_df = merged_df[['Date', f'{ticker}']].copy()

      # Merge the current ticker's returns into the overall returns DataFrame
      if daily_returns_df.empty:
          daily_returns_df = returns_df  # Initialize with the first ticker's returns
      else:
            # Merge with outer join to ensure all dates are preserved
          daily_returns_df = pd.merge(daily_returns_df, returns_df, on='Date', how='outer')


      # Calculate the periodic returns by resampling the total returns and store it in the monthly returns DataFrame
      m_returns = (1 + daily_returns_df.set_index('Date')[f'{ticker}']).resample(resampling_frequency).prod() - 1
      periodic_returns_df[f'{ticker}'] = m_returns
  # Move the index to a column
  periodic_returns_df = periodic_returns_df.reset_index()

  # Optionally, rename the 'index' column to 'Date'
  periodic_returns_df.rename(columns={'index': 'Date'}, inplace=True)


  return daily_returns_df, periodic_returns_df


def calculate_inflation_rate(inflation_ticker, inflation_df, resampling_frequency = 'M'):
  """
  This function calculates the monthly inflation rate and the cumulative inflation rate.

  """
  inflation_rate_df = inflation_df.copy()
  inflation_rate_df['Date'] = pd.to_datetime(inflation_df['Date'])
  inflation_rate_df = inflation_df.set_index('Date').resample('M').ffill().reset_index()
  inflation_rate_df['InflationRate'] = inflation_rate_df[inflation_ticker].pct_change()


  #inflation_rate_df = inflation_rate_df.dropna(subset=['InflationRate'])

  return inflation_rate_df


def calculate_real_returns(Tickers, inflation_df, returns_df):
  merged_df = pd.merge(inflation_df, returns_df, on='Date', how='inner')
  merged_df = merged_df.dropna(subset=['InflationRate'])
  returns_df = merged_df.copy()
  # Calculate real returns
  for ticker in Tickers:
    returns_df[f'RealReturn_{ticker}'] = ((1 + returns_df[f'{ticker}']) / (1 + returns_df['InflationRate'])) - 1


  return returns_df


def calculate_cumulative_returns(Tickers, returns_df, with_cash=True):
  cumulative_returns_df = returns_df.copy()

  #Calculate cumulative inflation
  cumulative_returns_df['CumulativeInflation'] = (1 + returns_df['InflationRate']).cumprod() - 1

  #Calculate cumulative nominal returns
  for ticker in Tickers:
    cumulative_returns_df[f'CumulativeNominalReturn_{ticker}'] = (1 + returns_df[f'{ticker}']).cumprod() - 1
    cumulative_returns_df[f'CumulativeRealReturn_{ticker}'] = (1 + cumulative_returns_df[f'CumulativeNominalReturn_{ticker}']) / (1 + cumulative_returns_df['CumulativeInflation']) - 1

  if with_cash ==True:
    cumulative_returns_df['CumulativeNominalReturn_Cash'] = 0
    cumulative_returns_df['CumulativeRealReturn_Cash'] = (1 + cumulative_returns_df['CumulativeNominalReturn_Cash']) / (1 + cumulative_returns_df['CumulativeInflation']) - 1


  return cumulative_returns_df


def calculate_value_of_investment(initial_investment, returns_df, tickers, with_cash=True):
  """
  This function calculates the value of an initial investment in one of the securities that we are considering.
  """
  value_df = returns_df.copy()
  for ticker in tickers:
    value_df[f'NominalValue_{ticker}'] = initial_investment * (1 + value_df[f'CumulativeNominalReturn_{ticker}'])
    value_df[f'RealValue_{ticker}'] = initial_investment * (1 + value_df[f'CumulativeRealReturn_{ticker}'])

  if with_cash==True:
    value_df['NominalValue_Cash'] = initial_investment
    value_df['RealValue_Cash'] = initial_investment * (1 + value_df['CumulativeRealReturn_Cash'])

  return value_df

def plot_cumulative_returns(Tickers, cumulative_returns_df, with_cash=True):

  """
  This function plots the cumulative returns (real and nominal) for each ticker.

  Args:
    cumulative_returns_df: DataFrame containing the cumulative returns data.

  Returns:
    None
  """
  ncols = 3
  nrows = (len(Tickers) + ncols) // ncols

  fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
  axes = axes.flatten()
  for i, ticker in enumerate(Tickers):
    ax = axes[i]
    ax.plot(cumulative_returns_df['Date'], cumulative_returns_df[f'CumulativeNominalReturn_{ticker}'], label=f'Cumulative Nominal Return {ticker}')
    ax.plot(cumulative_returns_df['Date'], cumulative_returns_df['CumulativeInflation'], label='Cumulative Inflation')
    ax.plot(cumulative_returns_df['Date'], cumulative_returns_df[f'CumulativeRealReturn_{ticker}'], label=f'Cumulative Real Return {ticker}')

    ax.set_title(f'{ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)

  if with_cash ==True:
    ax = axes[-1]
    ax.plot(cumulative_returns_df['Date'], cumulative_returns_df['CumulativeNominalReturn_Cash'], label='Cumulative Nominal Return Cash')
    ax.plot(cumulative_returns_df['Date'], cumulative_returns_df['CumulativeInflation'], label='Cumulative Inflation')
    ax.plot(cumulative_returns_df['Date'], cumulative_returns_df['CumulativeRealReturn_Cash'], label='Cumulative Real Return Cash')

    ax.set_title('Cash')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)

  plt.tight_layout()
  plt.show()




def plot_investment_growth(Tickers, value_df, with_cash=True):

  """
  This function plots the cumulative growth of an initial investment in one of the securities that we are considering.

  Args:
    value_df: DataFrame containing the value of the investment data.
    with_cash: Boolean indicating if cash should be included in the plot.

  Returns:
    None
  """
  ncols = 3
  nrows = (len(Tickers) + ncols) // ncols

  fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
  axes = axes.flatten()
  for i, ticker in enumerate(Tickers):
    ax = axes[i]
    ax.plot(value_df['Date'], value_df[f'NominalValue_{ticker}'], label=f'Nominal Value {ticker}')
    ax.plot(value_df['Date'], value_df[f'RealValue_{ticker}'], label=f'Real Value {ticker}')

    ax.set_title(f'{ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value (USD)')
    ax.legend()
    ax.grid(True)

  if with_cash ==True:
    ax = axes[-1]
    ax.plot(value_df['Date'], value_df['NominalValue_Cash'], label='Nominal Value Cash')
    ax.plot(value_df['Date'], value_df['RealValue_Cash'], label='Real Value Cash')

    ax.set_title('Cash')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value (USD)')
    ax.legend()
    ax.grid(True)

  plt.tight_layout()
  plt.show()
  
def plot_real_returns(Tickers, returns_df):

  fig, axes = plt.subplots(figsize=(15, 5))

  for i, ticker in enumerate(Tickers):
    axes.plot(returns_df['Date'], returns_df[f'CumulativeRealReturn_{ticker}'], label=f'Real Value {ticker}')


  axes.set_title('Cumulative Real Returns')
  axes.set_xlabel('Date')
  axes.set_ylabel('Real Value')
  axes.legend()
  axes.grid(True)

  plt.tight_layout()
  plt.show()  
  
  
  
  