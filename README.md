# Quantitative Momentum Strategy

This project is aimed at implementing a quantitative momentum strategy for stock selection. The strategy identifies stocks with strong recent price performance and uses a ranking system to construct a portfolio of high-momentum stocks.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Performance Analysis](#performance-analysis)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

The Quantitative Momentum Strategy leverages historical price data to identify stocks with the strongest recent price performance. It then ranks these stocks based on their momentum scores and selects a portfolio of top-performing stocks for investment.

This project uses the IEX Cloud API for stock data and implements the strategy in Python.

## Getting Started

1. Clone this repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file using `pip`:
3. Obtain an IEX Cloud API token from [https://iexcloud.io/](https://iexcloud.io/) and update the `IEX_CLOUD_API_TOKEN` variable in the `secrets.py` file.

## Usage

1. Run the script by executing the main Python file:

2. Follow the prompts to enter your portfolio size and view the recommended stock purchases.

## Performance Analysis

The project includes a performance analysis section where the cumulative returns of the selected portfolio are compared to a benchmark index (S&P 500). The script calculates and visualizes the cumulative returns over time.

## Dependencies

- Python 3.x
- pandas
- requests
- xlsxwriter
- numpy
- scipy
- matplotlib

## License

This project is licensed under the [MIT License](LICENSE).


DISCLAIMER: Not financial advice, this code is for educational purposes only. I am grateful to FreeCodeCamp for offering a course that this project was a part of. 
