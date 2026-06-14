import numpy as np
import pandas as pd

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    middle = data['close'].rolling(period).mean()
    std = data['close'].rolling(period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return upper, middle, lower

def bollinger_rsi_strategy(
    data: pd.DataFrame,
    bb_period: int = 20,
    bb_std_dev: float = 2.0,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    stop_loss_pct: float = 0.05,
    max_holding_days: int = 10
) -> pd.DataFrame:
    """
    Bollinger Bands + RSI Trading Strategy
    
    Theory:
    - Bollinger Bands identify overbought/oversold conditions based on price deviation from moving average
    - RSI confirms momentum and identifies potential reversal points
    - Combined signals provide stronger entry/exit points
    
    Entry Logic:
    - LONG: Price touches lower Bollinger Band AND RSI < oversold threshold
    - SHORT: Price touches upper Bollinger Band AND RSI > overbought threshold
    
    Exit Logic:
    - LONG: Price reaches middle band OR RSI > 50 OR stop loss
    - SHORT: Price reaches middle band OR RSI < 50 OR stop loss
    
    Parameters:
    - data: DataFrame with OHLCV data
    - bb_period: Period for Bollinger Bands calculation
    - bb_std_dev: Standard deviation multiplier for Bollinger Bands
    - rsi_period: Period for RSI calculation
    - rsi_oversold: RSI threshold for oversold condition (buy signal)
    - rsi_overbought: RSI threshold for overbought condition (sell signal)
    - stop_loss_pct: Stop loss percentage
    - max_holding_days: Maximum days to hold a position
    
    Returns:
    - DataFrame with trade information
    """
    # Ensure datetime index
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    
    # Calculate indicators
    data['rsi'] = calculate_rsi(data, rsi_period)
    data['bb_upper'], data['bb_middle'], data['bb_lower'] = calculate_bollinger_bands(
        data, bb_period, bb_std_dev
    )
    
    # Remove NaN values
    data = data.dropna()
    
    trades = []
    position = None
    entry_price = None
    entry_date = None
    
    for i in range(len(data) - 1):
        current_row = data.iloc[i]
        next_row = data.iloc[i + 1]
        current_date = current_row.name
        current_price = current_row['close']
        current_rsi = current_row['rsi']
        current_bb_upper = current_row['bb_upper']
        current_bb_lower = current_row['bb_lower']
        current_bb_middle = current_row['bb_middle']
        
        # Position Management
        if position is not None:
            holding_days = (current_date - entry_date).days
            
            # Exit conditions for LONG position
            if position == "long":
                exit_reason = "target"
                
                # Exit if price reaches middle band
                if current_price >= current_bb_middle:
                    exit_reason = "middle_band"
                # Exit if RSI becomes overbought
                elif current_rsi >= rsi_overbought:
                    exit_reason = "rsi_overbought"
                # Exit if stop loss hit
                elif current_price <= entry_price * (1 - stop_loss_pct):
                    exit_reason = "stop_loss"
                # Exit if max holding days reached
                elif holding_days >= max_holding_days:
                    exit_reason = "time_exit"
                else:
                    exit_reason = None
                
                if exit_reason:
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "profit_loss": current_price - entry_price,
                        "trade_type": "long",
                        "holding_days": holding_days,
                        "exit_reason": exit_reason
                    })
                    position = None
            
            # Exit conditions for SHORT position
            elif position == "short":
                exit_reason = "target"
                
                # Exit if price reaches middle band
                if current_price <= current_bb_middle:
                    exit_reason = "middle_band"
                # Exit if RSI becomes oversold
                elif current_rsi <= rsi_oversold:
                    exit_reason = "rsi_oversold"
                # Exit if stop loss hit
                elif current_price >= entry_price * (1 + stop_loss_pct):
                    exit_reason = "stop_loss"
                # Exit if max holding days reached
                elif holding_days >= max_holding_days:
                    exit_reason = "time_exit"
                else:
                    exit_reason = None
                
                if exit_reason:
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "profit_loss": entry_price - current_price,
                        "trade_type": "short",
                        "holding_days": holding_days,
                        "exit_reason": exit_reason
                    })
                    position = None
        
        # Entry conditions (only if no position)
        if position is None:
            # LONG entry: Price touches lower BB AND RSI oversold
            if (current_price <= current_bb_lower and current_rsi <= rsi_oversold):
                position = "long"
                entry_price = next_row['close']  # Enter at next day's open/close
                entry_date = next_row.name
            
            # SHORT entry: Price touches upper BB AND RSI overbought
            elif (current_price >= current_bb_upper and current_rsi >= rsi_overbought):
                position = "short"
                entry_price = next_row['close']  # Enter at next day's open/close
                entry_date = next_row.name
    
    # Handle any open position at the end
    if position is not None:
        last_row = data.iloc[-1]
        exit_date = last_row.name
        exit_price = last_row['close']
        holding_days = (exit_date - entry_date).days
        
        if position == "long":
            profit_loss = exit_price - entry_price
        else:  # short
            profit_loss = entry_price - exit_price
        
        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_loss": profit_loss,
            "trade_type": position,
            "holding_days": holding_days,
            "exit_reason": "end_of_data"
        })
    
    # Create and return trade DataFrame
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["entry_date"] = pd.to_datetime(trade_df["entry_date"])
        trade_df["exit_date"] = pd.to_datetime(trade_df["exit_date"])
        return trade_df
    else:
        return pd.DataFrame(columns=["entry_date", "exit_date", "entry_price", "exit_price",
                                   "profit_loss", "trade_type", "holding_days", "exit_reason"])

def run_bollinger_rsi_if_valid(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Wrapper function to run Bollinger Bands + RSI strategy with validation
    """
    if data is None or data.empty:
        print("No data provided")
        return pd.DataFrame()
    
    if len(data) < 50:  # Need enough data for indicators
        print("Insufficient data for indicator calculation")
        return pd.DataFrame()
    
    print("Running Bollinger Bands + RSI Strategy")
    return bollinger_rsi_strategy(data, **kwargs) 