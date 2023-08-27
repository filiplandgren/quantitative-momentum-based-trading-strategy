# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 18:30:23 2023

@author: fl5g21
"""

import numpy as np
import pandas as pd
import requests
import xlsxwriter
import math
from scipy import stats
from statistics import mean
import matplotlib.pyplot as plt

# API in the secrets file
from secrets import IEX_CLOUD_API_TOKEN

# -----------------------------
# Data Preparation
# -----------------------------

# Importing Our List of Stocks
stocks = pd.read_csv('sp_500_stocks.csv')

# Making Our First API Call
symbol = 'AAPL'
api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/stats?token={IEX_CLOUD_API_TOKEN}'
response = requests.get(api_url)
data = response.json()
year1_change_percent = data['year1ChangePercent']

# -----------------------------
# Helper Functions
# -----------------------------

# Define the 'chunks' function
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# -----------------------------
# Fetching Stock Data
# -----------------------------

# Divide the stocks into groups of 100 for batch API calls
symbol_groups = list(chunks(stocks['Ticker'], 100))
symbol_strings = [','.join(symbol_group) for symbol_group in symbol_groups]

# Columns for our final DataFrame
my_columns = ['Ticker', 'Price', 'One-Year Price Return', 'Number of Shares to Buy']
final_dataframe = pd.DataFrame(columns=my_columns)

# Loop through each batch of symbols
for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(batch_api_call_url).json()
    
    # Loop through each symbol in the batch
    for symbol in symbol_string.split(','):
        final_dataframe = final_dataframe.append(
            pd.Series([
                symbol,
                data[symbol]['quote']['latestPrice'],
                data[symbol]['stats']['year1ChangePercent'],
                'N/A'
            ],
            index=my_columns),
            ignore_index=True
        )
        
        
# -----------------------------
# Filtering and Portfolio Input
# -----------------------------

# Removing Low-Momentum Stocks
final_dataframe.sort_values('One-Year Price Return', ascending=False, inplace=True)
final_dataframe = final_dataframe[:50]
final_dataframe.reset_index(drop=True, inplace=True)

# Define a function to get the portfolio size from the user
def portfolio_input():
    global portfolio_size
    portfolio_size = input("Enter the value of your portfolio:")

    try:
        val = float(portfolio_size)
    except ValueError:
        print("That's not a number! \n Try again:")
        portfolio_size = input("Enter the value of your portfolio:")

# Get portfolio size from the user
if __name__ == "__main__":
    portfolio_input()
    print(portfolio_size)

position_size = float(portfolio_size) / len(final_dataframe.index)

# Loop through each stock and calculate the number of shares to buy
for i in range(len(final_dataframe['Ticker'])):
    final_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / final_dataframe['Price'][i])

# -----------------------------
# Momentum Strategy Building (better)
# -----------------------------

# Columns for HQM DataFrame
hqm_columns = [
    'Ticker', 'Price', 'Number of Shares to Buy', 'One-Year Price Return',
    'One-Year Return Percentile', 'Six-Month Price Return', 'Six-Month Return Percentile',
    'Three-Month Price Return', 'Three-Month Return Percentile',
    'One-Month Price Return', 'One-Month Return Percentile', 'HQM Score'
]

hqm_dataframe = pd.DataFrame(columns=hqm_columns)

# Loop through each batch of symbols
for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(batch_api_call_url).json()
    
    # Loop through each symbol in the batch
    for symbol in symbol_string.split(','):
        hqm_dataframe = hqm_dataframe.append(
            pd.Series([
                symbol, data[symbol]['quote']['latestPrice'], 'N/A',
                data[symbol]['stats']['year1ChangePercent'], 'N/A',
                data[symbol]['stats']['month6ChangePercent'], 'N/A',
                data[symbol]['stats']['month3ChangePercent'], 'N/A',
                data[symbol]['stats']['month1ChangePercent'], 'N/A', 'N/A'
            ], index=hqm_columns),
            ignore_index=True
        )


# -----------------------------
# Calculating Momentum Metrics
# -----------------------------

# Calculating Momentum Percentiles
time_periods = ['One-Year', 'Six-Month', 'Three-Month', 'One-Month']

for row in hqm_dataframe.index:
    for time_period in time_periods:
        hqm_dataframe.loc[row, f'{time_period} Return Percentile'] = stats.percentileofscore(
            hqm_dataframe[f'{time_period} Price Return'],
            hqm_dataframe.loc[row, f'{time_period} Price Return']
        ) / 100

# Calculating the HQM Score
for row in hqm_dataframe.index:
    momentum_percentiles = []
    for time_period in time_periods:
        momentum_percentiles.append(hqm_dataframe.loc[row, f'{time_period} Return Percentile'])
    hqm_dataframe.loc[row, 'HQM Score'] = mean(momentum_percentiles)

# Selecting the 50 Best Momentum Stocks
hqm_dataframe.sort_values(by='HQM Score', ascending=False, inplace=True)
hqm_dataframe = hqm_dataframe[:50]

# Calculating the Number of Shares to Buy for HQM Stocks
if __name__ == "__main__":
    portfolio_input()

position_size = float(portfolio_size) / len(hqm_dataframe.index)
for i in range(len(hqm_dataframe['Ticker'])):
    hqm_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / hqm_dataframe['Price'][i])
    
    
  
    
# -----------------------------
# Portfolio Diversification Analysis
# -----------------------------
    
  
    
  
# Calculate Cumulative Returns
def calculate_cumulative_returns(df):
    df['Cumulative Return'] = (1 + df['One-Year Price Return']).cumprod()
    return df

# Calculate Cumulative Returns for S&P 500 (as a simple example)
def calculate_benchmark_cumulative_returns(df, benchmark_returns):
    df['Benchmark Cumulative Return'] = (1 + benchmark_returns).cumprod()
    return df



# -----------------------------
# Data Visualization
# -----------------------------


# Visualize Cumulative Returns
def visualize_cumulative_returns(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Cumulative Return'], label='Portfolio')
    plt.plot(df['Benchmark Cumulative Return'], label='S&P 500')
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()




# Load S&P 500 benchmark data 
benchmark_data = pd.read_csv('sp500_returns.csv')  # Check that benchmark returns are in the csv file! If not, remember to import
benchmark_returns = benchmark_data['Return'] / 100

# Calculate cumulative returns for portfolio and benchmark
final_dataframe = calculate_cumulative_returns(final_dataframe)
final_dataframe = calculate_benchmark_cumulative_returns(final_dataframe, benchmark_returns)

# Visualize cumulative returns
visualize_cumulative_returns(final_dataframe)
    
    


returns = hqm_dataframe.set_index('Ticker').pct_change().dropna()
correlation_matrix = returns.corr()
covariance_matrix = returns.cov()
num_assets = len(hqm_dataframe)

# Define the portfolio weights
weights = np.array([1/num_assets] * num_assets)

# Calculate portfolio variance
portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))

# Calculate portfolio volatility
portfolio_volatility = np.sqrt(portfolio_variance)

# Calculate diversifiable risk (idiosyncratic risk)
diversifiable_risk = portfolio_variance - np.sum(weights**2 * covariance_matrix.values)

# Calculate non-diversifiable risk (systematic risk)
non_diversifiable_risk = portfolio_variance - diversifiable_risk

print(f"Portfolio Volatility: {portfolio_volatility:.4f}")
print(f"Diversifiable Risk: {diversifiable_risk:.4f}")
print(f"Non-Diversifiable Risk: {non_diversifiable_risk:.4f}")
    




# Visualize Correlation Matrix
plt.figure(figsize=(10, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation Matrix')
tick_labels = hqm_dataframe['Ticker'].values
plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=90)
plt.yticks(np.arange(len(tick_labels)), tick_labels)
plt.show()


# Visualize Returns Distribution
plt.figure(figsize=(10, 6))
for ticker in returns.columns:
    plt.hist(returns[ticker], bins=30, alpha=0.6, label=ticker)
plt.title('Returns Distribution')
plt.xlabel('Daily Returns')
plt.ylabel('Frequency')
plt.legend()
plt.show()
    

# -----------------------------
# Excel Output and Formatting
# -----------------------------


writer = pd.ExcelWriter('momentum_strategy.xlsx', engine='xlsxwriter')
hqm_dataframe.to_excel(writer, sheet_name='Momentum Strategy', index=False)

# Creating the Formats We'll Need For Our .xlsx File
background_color = '#0a0a23'
font_color = '#ffffff'

string_template = writer.book.add_format({
    'font_color': font_color,
    'bg_color': background_color,
    'border': 1
})

dollar_template = writer.book.add_format({
    'num_format': '$0.00',
    'font_color': font_color,
    'bg_color': background_color,
    'border': 1
})

integer_template = writer.book.add_format({
    'num_format': '0',
    'font_color': font_color,
    'bg_color': background_color,
    'border': 1
})

percent_template = writer.book.add_format({
    'num_format': '0.0%',
    'font_color': font_color,
    'bg_color': background_color,
    'border': 1
})

column_formats = { 
    'A': ['Ticker', string_template],
    'B': ['Price', dollar_template],
    'C': ['Number of Shares to Buy', integer_template],
    'D': ['One-Year Price Return', percent_template],
    'E': ['One-Year Return Percentile', percent_template],
    'F': ['Six-Month Price Return', percent_template],
    'G': ['Six-Month Return Percentile', percent_template],
    'H': ['Three-Month Price Return', percent_template],
    'I': ['Three-Month Return Percentile', percent_template],
    'J': ['One-Month Price Return', percent_template],
    'K': ['One-Month Return Percentile', percent_template],
    'L': ['HQM Score', integer_template]
}

# Apply formats to each column and write column names
for column in column_formats.keys():
    writer.sheets['Momentum Strategy'].set_column(f'{column}:{column}', 20, column_formats[column][1])
    writer.sheets['Momentum Strategy'].write(f'{column}1', column_formats[column][0], string_template)

# Saving Our Excel Output
writer.save()
