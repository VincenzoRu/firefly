import yfinance as yf
import matplotlib.pyplot as plt
from random import uniform, random
from math import sqrt, exp


# RETRIEVE S&P500 DATA
def get_financial_data(ticker, start_date, end_date, interval):
    try:
        data = ticker.history(start=start_date, end=end_date, interval=interval)
    except Exception as e:
        print(f"Error retrieving financial data: {e}")
        return None
    #print(data.head())
    #print(type(data))
    return data


# CALCULATE MA
def calculate_ma(data, period):
    data[f'{period}-day MA'] = data['Close'].rolling(window=period).mean()
    data['MA_signal'] = 0
    data.loc[data['Close'] > data[f'{period}-day MA'], 'MA_signal'] = 1  
    data.loc[data['Close'] < data[f'{period}-day MA'], 'MA_signal'] = -1 
    return data


# CALCULATE MACD
def calculate_macd(data, short_period, long_period, signal_period):
    data['EMA_short'] = data['Close'].ewm(span=short_period, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_period, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal']
    data['MACD_signal'] = 0
    data.loc[data['MACD'] > data['Signal'], 'MACD_signal'] = 1  
    data.loc[data['MACD'] < data['Signal'], 'MACD_signal'] = -1 
    return data


# CALCULATE TRB
def calculate_trb(data, period):
    data['TRB_High'] = data['Close'].shift(1).rolling(window=period).max() #shift(1) excludes today's closing price
    data['TRB_Low'] = data['Close'].shift(1).rolling(window=period).min()
    data['TRB_signal'] = 0
    data.loc[data['Close'] > data['TRB_High'], 'TRB_signal'] = 1  
    data.loc[data['Close'] < data['TRB_Low'], 'TRB_signal'] = -1 
    return data


# EXCESS RETURN
"""
def calculate_excess_return(data, signals, transaction_cost=0.0025):
    assert len(data) == len(signals), "Signals length must match the number of data points."

    #print(f"Signals (sample): {signals[:10]}")  # Debug signal variation

    total_return = 0
    for i in range(1, len(data)):
        
        if signals[i] == 1:  # Buy signal
            total_return += (data['Close'].iloc[i] - data['Close'].iloc[i - 1]) / data['Close'].iloc[i - 1]
        elif signals[i] == -1:  # Sell signal
            total_return -= (data['Close'].iloc[i] - data['Close'].iloc[i - 1]) / data['Close'].iloc[i - 1]
        total_return -= transaction_cost  # Subtract transaction cost

    #print("total_return:",total_return)

    # Calculate B&H return
    buy_and_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
    
    #print("bh:",buy_and_hold_return)

    return total_return - buy_and_hold_return
"""
    
def calculate_excess_return_new(data, signals, transaction_cost=0.0005):
    """
    Calculate the excess return based on trading signals and transaction costs.
    Signals:
    - > 0.5: Buy
    - < -0.5: Sell
    - -0.5 <= Signal <= 0.5: Hold (No Action)
    """
    assert len(data) == len(signals), "Signals length must match the number of data points!"

    total_return = 0
    for i in range(1, len(data)):
        if signals[i] > 0.5:  # Buy signal
            total_return += (data['Close'].iloc[i] - data['Close'].iloc[i - 1]) / data['Close'].iloc[i - 1]
        elif signals[i] < -0.5:  # Sell signal
            total_return -= (data['Close'].iloc[i] - data['Close'].iloc[i - 1]) / data['Close'].iloc[i - 1]
        # Hold signal (no change in total_return for -0.5 <= signals[i] <= 0.5)
        total_return -= transaction_cost  # Subtract transaction cost for each decision (buy/sell)

     # Calculate B&H return
    buy_and_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
    print("bh:",buy_and_hold_return)
    return total_return - buy_and_hold_return



# FIREFLY ALGORITHM
def firefly_algorithm(data, ma_signals, macd_signals, trb_signals, max_generations, n_fireflies, alpha, beta0, gamma):
    """
    Firefly Algorithm to maximize excess return.
    """
    # Step 1 (Initialization): randomness
    dim = 3  
    bounds = [0, 1] 
    fireflies = [[uniform(bounds[0], bounds[1]) for _ in range(dim)] for _ in range(n_fireflies)]
    intensities = []
    #print("Fireflies:",fireflies)

    # Step 2 (Compute the brightness of firefly):
    all_signals = list(zip(ma_signals, macd_signals, trb_signals))
    #print("All Signals Sample:", all_signals[:10])

    #print("MA Signals Sample:", ma_signals)
    #print("MACD Signals Sample:", macd_signals)
    #print("TRB Signals Sample:", trb_signals)

    for firefly in fireflies:
      
        combined_signals = [(firefly[0] * s[0] + firefly[1] * s[1] + firefly[2] * s[2]) / sum(firefly) for s in all_signals]

        excess_return = calculate_excess_return_new(data, combined_signals)
        
        intensities.append(excess_return)

    #print("Combined signals:", combined_signals)
    #print("Intensity Initia:", intensities)

    # Step 3 & 4 (obtain the current global best and rank the fireflies):
    for gen in range(max_generations):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if intensities[j] > intensities[i]:  # Firefly j is brighter than firefly i
                    # Calculate distance
                    distance = sqrt(sum((fireflies[i][k] - fireflies[j][k]) ** 2 for k in range(dim)))
                    # Update attractiveness and move firefly i towards firefly j
                    beta = beta0 * exp(-gamma * distance ** 2)
                    for d in range(dim):
                        #old_weight = fireflies[i][d]  # Store old weight for debuging
                        fireflies[i][d] += beta * (fireflies[j][d] - fireflies[i][d]) + alpha * (random() - 0.5)
                        fireflies[i][d] = max(bounds[0], min(bounds[1], fireflies[i][d]))  # Enforce bounds
                        
                    # Update intensity
                    combined_signals = [
                        fireflies[i][0] * s[0] + fireflies[i][1] * s[1] + fireflies[i][2] * s[2] for s in all_signals
                    ]
                    intensities[i] = calculate_excess_return_new(data, combined_signals)

        # Find the best solution in the current generation
        best_index = intensities.index(max(intensities))

    # Return the best solution
    best_index = intensities.index(max(intensities))
    return fireflies[best_index], intensities[best_index]

# VALIDATE RESULTS FOR DEBUGGING
def debug_results(data):
    """
    plt.plot(data['Close'], label="Close Price")
    plt.plot(data['50-day MA'], label="50-day MA")
    plt.legend()
    plt.show()"""

    """
    print(training_data[['MACD', 'Signal', 'Histogram']][10:50])
    # Create subplots with relative heights
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot Close Price (Top Panel)
    axs[0].plot(data['Close'], label="Close Price", color='blue')
    axs[0].set_title("Close Price and MACD")
    axs[0].legend()
    
    # Plot MACD, Signal, and Histogram (Bottom Panel)
    axs[1].plot(data['MACD'], label="MACD Line", color='red')
    axs[1].plot(data['Signal'], label="Signal Line", color='green')
    axs[1].bar(data.index, data['Histogram'], label="Histogram", color='gray', alpha=0.5)
    axs[1].axhline(0, color='black', linewidth=0.5, linestyle='--')  # Zero line
    axs[1].legend()
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()"""

    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['TRB_High'], label='TRB High', color='green', linestyle='--')
    plt.plot(data['TRB_Low'], label='TRB Low', color='red', linestyle='--')
    plt.title("TRB (Trading Range Breakout)")
    plt.legend()
    plt.show()"""


if __name__ == "__main__":
    ticker = yf.Ticker("^GSPC") # S&P500
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    interval = "1d"
    training_data = get_financial_data(ticker, start_date, end_date, interval)
    print(f"Collected {len(training_data)} data points.")
    
    # Calculate indicators
    training_data = calculate_ma(training_data, period=50)
    training_data = calculate_macd(training_data, short_period=12, long_period=26, signal_period=9)
    training_data = calculate_trb(training_data, period=20)

    # Firefly parameters
    max_generations = 50
    n_fireflies = 20
    alpha = 0.2
    beta0 = 1
    gamma = 1

    ma_signals = training_data['MA_signal'].tolist()
    macd_signals = training_data['MACD_signal'].tolist() 
    trb_signals = training_data['TRB_signal'].tolist()  

    best_weights, best_excess_return = firefly_algorithm(training_data, ma_signals, macd_signals, trb_signals, max_generations, n_fireflies, alpha, beta0, gamma)
    print("\nFinal Best Weights:", best_weights)
    print("Final Best Excess Return:", best_excess_return)

    # Debugging
    #training_data.to_csv('output.csv', index=True)
    #debug_results(training_data)