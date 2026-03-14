import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm

# Titles
st.set_page_config(page_title="Options Pricing")
st.title("Options Pricing Calculator")
st.write("Black-Scholes pricing, Monte Carlo & Binomial tree simulations")

# Black-Scholes formula for an European Call Option
def black_scholes_call(S, K, T, r, sigma):
    # T can't be zero
    if T <= 0: 
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Black-Scholes formula for an European Put Option
def black_scholes_put(S, K, T, r, sigma):
    # T can't be zero
    if T <= 0: 
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Monte Carlo simulation of geometric brownian motion for asian and lookback options
def path_simulation(S0, T, r, sigma, simulation_number, steps_number):
    dt = T / steps_number
    paths = np.zeros((steps_number + 1, simulation_number))
    paths[0] = S0 
    
    for t in range(1, steps_number + 1):
        Z = np.random.standard_normal(simulation_number)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
    return paths

# Binomial Tree for american call options
def binomial_tree_call(S, K, T, r, sigma, N):
    # N is the number of steps or depth of the tree
    dt = T / N
    # Up factor
    u = np.exp(sigma * np.sqrt(dt))
    # Down factor
    d = 1 / u
    
    # Risk neutral probability
    p = (np.exp(r * dt) - d) / (u - d) 

    # Price tree at maturity
    prices = np.zeros(N + 1)
    for i in range(N + 1):
        prices[i] = S * (u ** (N - i)) * (d ** i)

    # Option value at maturity
    option_values = np.maximum(prices - K, 0)

    # Backward induction or traveling back in time
    for step in range(N - 1, -1, -1):
        for i in range(step + 1):
            
            # Value if we wait
            wait_value = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            
            # Value if we exercise early
            current_price = S * (u ** (step - i)) * (d ** i)
            exercise_value = max(current_price - K, 0)
            
            # American option takes the maximum of both
            option_values[i] = max(wait_value, exercise_value)

    return option_values[0]

# Binomial Tree for American put option
def binomial_tree_put(S, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d) 

    # Price tree at maturity
    prices = np.zeros(N + 1)
    for i in range(N + 1):
        prices[i] = S * (u ** (N - i)) * (d ** i)

    # Option value at maturity
    option_values = np.maximum(K - prices, 0)

    # Backward induction
    for step in range(N - 1, -1, -1):
        for i in range(step + 1):
            
            # Value if we wait
            wait_value = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            
            # Value if we exercise early
            current_price = S * (u ** (step - i)) * (d ** i)
            exercise_value = max(K - current_price, 0)
            
            option_values[i] = max(wait_value, exercise_value)

    return option_values[0]

# Sidebar for user inputs
st.sidebar.header("Model Parameters")

# Ticker Input "AAPL" by default
ticker = st.sidebar.text_input("Ticker", value="AAPL")

# Price retrieval from yahoo finance
# @st.cache_data forces Streamlit to save the data in cache, so it doesn't make API calls with every mouse click
@st.cache_data
def price_retrieve(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1d")
        return float(hist['Close'].iloc[-1])
    except:
        return None

spot_price = price_retrieve(ticker)

# Retrieves r and sigma from yahoo
@st.cache_data
def market_dynamics_retrieve(ticker_symbol):
    try:
        # Risk-Free rate
        irx = yf.Ticker("^IRX")
        hist_irx = irx.history(period="1d")
        r_market = float(hist_irx['Close'].iloc[-1])
    except:
        # if yahoo fails
        r_market = 5.0
        
    try:
        # volatility
        stock = yf.Ticker(ticker_symbol)
        hist_stock = stock.history(period="1y")
        # Calculate daily percentage returns
        daily_returns = hist_stock['Close'].pct_change().dropna()
        # Annualized standard deviation (252 trading days)
        sigma_market = float(daily_returns.std() * np.sqrt(252)) * 100
    except:
        # if yahoo fails
        sigma_market = 20.0
        
    return r_market, sigma_market

# Main
if spot_price is None:
    st.error(f"Can't download {ticker} data. Check the ticker symbol.")
else:
    st.success(f"Spot price of {ticker}: ${spot_price:.2f}")

    # Option Parameters
    st.sidebar.subheader("Option")
    # Strike price 5% above spot price by default
    # Using session_state so Streamlit remembers the price and doesn't overwrite with every mouse click
    if "k_strike" not in st.session_state:
        st.session_state.k_strike = float(round(spot_price * 1.05, 2))
    K = st.sidebar.number_input("Strike Price (K)", value=st.session_state.k_strike, key="k_strike") 
    # Time to expiration in days 365 by default
    dias_T = st.sidebar.number_input("Days until Expiration (T)", min_value=1, value=365)
    
    # Market Dynamics
    st.sidebar.subheader("Market Dynamics")
    
    # Checkbox for automatic market data
    autofill = st.sidebar.checkbox("Autofill from Yahoo Finance", value=False)
    
    if autofill:
        # Retrieve data and show success message
        r_market, sigma_market = market_dynamics_retrieve(ticker)
        st.sidebar.success(f"Data Loaded")
        
        # Use the inputs with real data
        r_input = st.sidebar.number_input("Risk-Free Rate (r) %", value=float(round(r_market, 2)))
        sigma_input = st.sidebar.number_input("Volatility (σ) %", value=float(round(sigma_market, 2)))
    else:
        # Default manual inputs
        r_input = st.sidebar.number_input("Risk-Free Rate (r) %", value=5.0)
        sigma_input = st.sidebar.number_input("Volatility (σ) %", value=20.0)
    
    # rate and volatility converted from percentage to decimal
    r = r_input / 100
    sigma = sigma_input / 100
    # Convert days to years
    T = dias_T / 365.0

    # Model Selection
    st.sidebar.subheader("Model Selection")
    option_type = st.sidebar.selectbox("Option Style", 
        ["European (Black-Scholes)", "American (Binomial Tree)", "Asian (Monte Carlo)", "Lookback (Monte Carlo)"])

    # Calculate Button
    if st.button("Calculate Option Price", type="primary"):
        
       # European
        if option_type == "European (Black-Scholes)":

            # Explanation
            st.markdown("""
            The results you see below represent the Premium, which is the non-refundable fee that a client must pay today to the investment bank to enter into this contract.
            
            The Call Option gives the buyer the right, but not the obligation, to buy a stock from the bank at a predetermined fixed price, called the Strike Price ($K$), on an exact future date ($T$).
            If on that expiration day, the stock's market price ($S_T$) is higher than the Strike, the client exercises their right: they demand the bank sell it to them for $K$, immediately resell it in the open market for $S_T$, and pocket a profit (minus the premium paid today).
            If the stock drops below $K$, the client simply walks away and doesn't exercise the option. Their maximum loss is strictly limited to the premium paid for the contract.
            
            The Put Option gives the buyer the right to sell a stock to the bank at the Strike Price ($K$) on the future date ($T$).
            If the Strike ($K$) is higher than the stock price on the expiration date ($S_T$), the client buys it in the open market for $S_T$, and the bank is legally obligated to buy it from them for $K$. The client pockets the profit.
            If the stock rallies higher than $K$, the client doesn't exercise their option and only loses the premium paid.
            
            Under this specific model, the client can only settle the contract and exercise their right on the exact expiration date $T$.
            """)
            st.markdown("---")

            # Calculating both prices
            call_price = black_scholes_call(spot_price, K, T, r, sigma)
            put_price = black_scholes_put(spot_price, K, T, r, sigma)
            
            # Creating 2 columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="European Call Price", value=f"${call_price:.4f}")
            with col2:
                st.metric(label="European Put Price", value=f"${put_price:.4f}")

            st.markdown("---") 

            # More technical details
            st.info(r"""
            This valuation is computed using the analytical closed-form Black-Scholes-Merton model. The framework calculates the theoretical premium by discounting the expected payoff under a risk-neutral measure.
            
            The core equations for European Calls ($C$) and Puts ($P$) are defined as:
            
            $$C = S \cdot N(d_1) - K e^{-rT} N(d_2)$$
            
            $$P = K e^{-rT} N(-d_2) - S \cdot N(-d_1)$$
            
            A detailed explanation can be found [here](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black%E2%80%93Scholes_formula). 
            
            Some of this model flaws include that the formula strictly assumes volatility ($\sigma$) remains constant over the option's life. In live markets, implied volatility varies across different strike prices, forming a volatility smile or skew.
                    
            Also, this implementation assumes the underlying asset pays no dividends. A continuous dividend yield would mathematically decrease the call premium and increase the put premium.
                    
            Furthermore, the model assumes underlying stock prices follow a continuous geometric brownian motion. This framework ignores fat tails, meaning sudden, extreme market crashes occur more frequently in reality than the normal distribution predicts.
            """)  

        # Asian and Lookback with MonteCarlo
        elif option_type in ["Asian (Monte Carlo)", "Lookback (Monte Carlo)"]:
            
            # Explanations
            if option_type == "Asian (Monte Carlo)":
                st.markdown("""
                The results you see below represent the Premium, which is the non-refundable fee that a client must pay today to the investment bank to enter into this contract.
                
                An Asian Option is a path-dependent derivative where the payoff is determined by the average underlying price ($S_{avg}$) over a pre-set period of time ($T$), rather than the price at a specific expiration date.
                
                The Asian Call Option yields a profit if $S_{avg}$ is higher than the Strike Price ($K$). The client receives the difference between $S_{avg}$ and $K$. If the average price $S_{avg}$ remains below $K$, the option expires worthless, and the client only loses the premium.
                
                The Asian Put Option yields a profit if $K$ is higher than $S_{avg}$. The client receives the difference between $K$ and $S_{avg}$. If the average price remains above $K$, the option expires worthless.
                """)
            
            elif option_type == "Lookback (Monte Carlo)":
                st.markdown("""
                The results you see below represent the Premium, which is the non-refundable fee that a client must pay today to the investment bank to enter into this contract.
                
                A Lookback Option is a path-dependent derivative that allows the holder to "look back" over the life of the contract to determine the payoff based on the optimal price ($S_{max}$ or $S_{min}$) reached by the underlying asset.
                
                The Lookback Call Option yields a profit if the maximum price achieved by the stock during the contract's life ($S_{max}$) is higher than the Strike Price ($K$). The client receives the difference between $S_{max}$ and $K$. The option expires worthless with the client losing the premium, only if the stock never exceeds $K$ at any point.
                
                The Lookback Put Option yields a profit if the Strike Price ($K$) is higher than the minimum price achieved by the stock ($S_{min}$) over the contract's life. The client receives the difference between $K$ and $S_{min}$. The option expires worthless with the client losing the premium, only if the stock never drops below $K$.
                """)
            
            st.markdown("---")
            
            # Generating the paths
            with st.spinner('Simulating 5000 paths'): 
                simulation_number = 5000
                steps = max(1, int(dias_T))
                
                # Drawing the 5000 paths
                paths = path_simulation(spot_price, T, r, sigma, simulation_number, steps)
                
                # Math for Asian
                if option_type == "Asian (Monte Carlo)":
                    mean_prices = np.mean(paths, axis=0)
                    pay_call = np.maximum(mean_prices - K, 0)
                    call_price = np.mean(pay_call) * np.exp(-r * T)
                    pay_put = np.maximum(K - mean_prices, 0)
                    put_price = np.mean(pay_put) * np.exp(-r * T)
                
                # Math for Lookback
                elif option_type == "Lookback (Monte Carlo)":
                    max_prices = np.max(paths, axis=0)
                    pay_call = np.maximum(max_prices - K, 0)
                    call_price = np.mean(pay_call) * np.exp(-r * T)
                    min_prices = np.min(paths, axis=0)
                    pay_put = np.maximum(K - min_prices, 0)
                    put_price = np.mean(pay_put) * np.exp(-r * T)

            # Result columns and content
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"{option_type.split(' ')[0]} Call Price", value=f"${call_price:.4f}")
            with col2:
                st.metric(label=f"{option_type.split(' ')[0]} Put Price", value=f"${put_price:.4f}")
            
            st.markdown("---")
            
            # More technical details
            if option_type == "Asian (Monte Carlo)":
                st.info(r"""
                This valuation is computed using a Monte Carlo simulation of a geometric brownian motion (GBM). Because Asian options are strongly path-dependent (the payoff depends on the entire price history, not just the final price), closed-form analytical solutions are highly complex or non-existent for arithmetic averages.
                
                The algorithm simulates thousands (in our case 5000) of possible future price paths using the discrete-time GBM equation:
                
                $$S_{t+\Delta t} = S_t \exp\left( \left(r - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z \right)$$
                
                Where $Z \sim N(0,1)$ is a standard normal random variable. 
                
                For each path, we compute the arithmetic average price $\bar{S}$:
                
                $$\bar{S} = \frac{1}{N} \sum_{i=1}^{N} S_i$$
                
                The final premium is the discounted expected value of the payoffs across all simulated paths:
                
                $$C = e^{-rT} \mathbb{E}[\max(\bar{S} - K, 0)]$$
                
                Some of this model flaws include problems like computational intensity. Monte Carlo is a brute-force numerical method. Generating tens of thousands of paths requires significant CPU time, making it unfeasible for ultra-low-latency high-frequency trading.
                        
                The result is only an approximation. According to the central limit theorem, the pricing error converges at a rate of $O(1/\sqrt{M})$, where $M$ is the number of simulations. To halve the error, the computational workload must be quadrupled.
                
                Also, the accuracy of the simulation relies entirely on the quality of the underlying Random Number Generator (RNG) and its ability to avoid unwanted correlations in the $Z$ variables.
                """)
            
            elif option_type == "Lookback (Monte Carlo)":
                st.info(r"""
                This valuation is computed using Monte Carlo simulation of a geometric brownian motion (GBM). Lookback options are extreme examples of path-dependent derivatives, as their payoff relies entirely on the absolute global maximum or minimum achieved during the asset's trajectory.
                
                The algorithm simulates discrete price paths. For a path with $N$ time steps, we identify the extrema:
                
                $$S_{max} = \max_{0 \le i \le N} S_i \quad \text{and} \quad S_{min} = \min_{0 \le i \le N} S_i$$
                
                Based on our implementation of a **Fixed-Strike Lookback**, the discounted expected payoffs are calculated as:
                
                $$C = e^{-rT} \mathbb{E}[\max(S_{max} - K, 0)]$$
                
                $$P = e^{-rT} \mathbb{E}[\max(K - S_{min}, 0)]$$
                
                Some of this model flaws include discrete sampling bias. This is a critical flaw when simulating Lookbacks. The simulation only checks the price at discrete intervals (e.g., daily steps). However, the true continuous maximum between two days could be significantly higher. Discretely simulated Lookback Calls systematically underprice the continuous theoretical value. Advanced production models require brownian bridge interpolations to correct this continuous-time bias.
                        
                Because the payoff depends on a single extreme point rather than an average, Lookbacks are intensely sensitive to the log-normal distribution assumption. Real-world fat tails or sudden market jumps (jump-diffusion) drastically alter the actual maximums and minimums, making the standard GBM assumption precarious.
                        
                Similar to the Asian option, reaching a statistically significant expected value requires a massive number of paths, with the standard error decreasing slowly at a rate of $O(1/\sqrt{M})$.
                """)
                
            # Drawing the graph with 10 paths
            st.subheader("Stochastic Simulation of 10 (of 5000) sample paths")
            df_paths = pd.DataFrame(paths[:, :10])
            st.line_chart(df_paths)
                
        # American
        elif option_type == "American (Binomial Tree)":

            # Explanation
            st.markdown("""
            The results you see below represent the Premium, which is the non-refundable fee that a client must pay today to the investment bank to enter into this contract.
            
            An American Option is a financial contract that provides the buyer with greater flexibility than a European option. The key distinction is the early exercise feature: the client can exercise their right at any point during the life of the contract, up to and including the expiration date ($T$).
            
            The American Call Option gives the buyer the right, but not the obligation, to buy a stock from the bank at a predetermined fixed price, called the Strike Price ($K$). If at any time before or on expiration, the stock's current market price ($S_t$) is higher than $K$, the client can exercise their right. They demand the bank to sell it to them for $K$, immediately resell it in the open marketfor $S_t$, and pocket the profit. If the stock never rises above $K$, the client simply walks away, and their maximum loss is limited to the premium paid.
            
            The American Put Option gives the buyer the right to sell a stock to the bank at the Strike Price ($K$). If at any time $K$ is higher than the current stock price, the client can buy the stock in the open market, and the bank is legally obligated to buy it from them for $K$, allowing the client to pocket the profit. If the stock remains higher than $K$, the client doesn't exercise their option.
            """)
            st.markdown("---")

            with st.spinner('Building binomial tree'):
                steps = 200 
                
                # Calculating both prices
                call_price = binomial_tree_call(spot_price, K, T, r, sigma, steps)
                put_price = binomial_tree_put(spot_price, K, T, r, sigma, steps)
                
                # Creating 2 columns
                col1, col2 = st.columns(2)
                
                # Column 1 content
                with col1:
                    st.metric(label="American Call Price", value=f"${call_price:.4f}")

                #Column 2 content    
                with col2:
                    st.metric(label="American Put Price", value=f"${put_price:.4f}")
                    
                st.markdown("---")
                
                # more technical details
                st.info(r"""
                This valuation is computed using the Cox-Ross-Rubinstein Binomial Tree model. Unlike closed-form analytical solutions, this is a discrete-time lattice model that builds a pricing tree mapping possible future asset prices.
                
                The algorithm relies on backward induction, calculating the option's payoff at maturity and steps backward in time. At every single node, it compares the continuation value (the discounted expected future value) against the intrinsic value (the profit from exercising early).
                
                The core mechanics are defined by the up ($u$) and down ($d$) magnitude factors, and the risk-neutral probability ($p$) of an up-move:
                
                $$u = e^{\sigma\sqrt{\Delta t}} \quad \text{and} \quad d = \frac{1}{u}$$
                
                $$p = \frac{e^{r\Delta t} - d}{u - d}$$
                
                At each node $i$ during step $t$, the option value $V$ takes the maximum of immediate exercise or the discounted expectation:
                
                $$V_{t,i} = \max\left( \text{Intrinsic Value}, \ e^{-r\Delta t}[p \cdot V_{t+1, i+1} + (1-p) \cdot V_{t+1, i}] \right)$$
                        
                More details can be found [here](https://en.wikipedia.org/wiki/Binomial_options_pricing_model#Method)
                
                Some of this model flaws include problems with computational complexity. The tree size grows quadratically ($O(N^2)$). High precision requires thousands of time steps, which becomes computationally heavy and slow compared to instantaneous analytical formulas.
                        
                Also, market prices move continuously, but this model forces them into discrete jumps. Using too few steps can lead to oscillation errors where the calculated price bounces around the true theoretical value.
                
                And with this model we assumme constant volatility and risk-free rates across all time steps and nodes. Incorporating real-world local volatility surfaces requires significantly more advanced frameworks, such as finite difference methods.
                """)