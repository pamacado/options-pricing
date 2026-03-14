# Options Pricing Calculator

This is a personal project I built to put into practice some finance concepts and Python development. It is a web application that calculates the theoretical premium of different types of financial options.

You can try the live web app here: https://options-pricing-web.streamlit.app/

## Overview

The application pulls real-time market data (spot price, risk-free rate, and historical volatility) using the Yahoo Finance API, but you can also input the parameters manually. 

Depending on the option style selected, the app uses different mathematical engines:

- European Options: Priced using the analytical Black-Scholes-Merton formula.
- American Options: Priced using a Cox-Ross-Rubinstein Binomial Tree to properly account for the early exercise premium.
- Asian Options: Priced via Monte Carlo simulations of Geometric Brownian Motion, calculating the payoff based on the arithmetic average.
- Lookback Options: Priced using Monte Carlo simulations to find the optimal global maximum or minimum (Fixed-Strike).

## Built with

- Python
- Streamlit (for the UI)
- NumPy & SciPy (for the numerical computing and statistical distributions)
- yfinance (for live market data retrieval)

## How to run it locally

If you want to test the code on your own machine, clone this repository and install the dependencies:

```bash
git clone https://github.com/pamacado/options-pricing.git
cd options-pricing
pip install -r requirements.txt
streamlit run web.py
