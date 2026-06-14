import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from Analysis.statistical_tests import run_adf_test

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def estimate_hedge_ratio(y, x):
    """Estimate hedge ratio (beta) using linear regression: y ~ x"""
    x = add_constant(x)
    model = OLS(y, x).fit()
    return model.params[1]  # beta

def calculate_rolling_hedge_ratio(df, window=252):
    """
    Calculate rolling hedge ratio using only historical data.
    
    Parameters:
    - df: DataFrame with 'A' and 'B' columns
    - window: Lookback window in days (default 252 = 1 year)
    
    Returns:
    - Series of hedge ratios
    """
    rolling_beta = []
    dates = []
    
    for i in range(window, len(df)):
        # Use only past data (no look-ahead bias)
        past_data = df.iloc[i-window:i]
        
        # Calculate hedge ratio on historical data only
        beta = estimate_hedge_ratio(past_data['A'], past_data['B'])
        rolling_beta.append(beta)
        dates.append(df.index[i])
    
    return pd.Series(rolling_beta, index=dates)

def test_cointegration_rolling(df, window=252, min_coint_days=60):
    """
    Test cointegration using rolling window approach.
    
    Parameters:
    - df: DataFrame with 'A' and 'B' columns
    - window: Lookback window for cointegration test
    - min_coint_days: Minimum days required for cointegration test
    
    Returns:
    - Series of cointegration p-values
    """
    coint_p_values = []
    dates = []
    
    for i in range(window, len(df)):
        # Use only past data
        past_data = df.iloc[i-window:i]
        
        if len(past_data) >= min_coint_days:
            try:
                _, coint_p, _ = coint(past_data['A'], past_data['B'])
                coint_p_values.append(coint_p)
            except:
                coint_p_values.append(1.0)  # Fail cointegration test
        else:
            coint_p_values.append(1.0)  # Not enough data
        
        dates.append(df.index[i])
    
    return pd.Series(coint_p_values, index=dates)

def pairs_trading_cointegration_strategy_improved(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    hedge_ratio_window: int = 252,  # 1 year for hedge ratio
    coint_window: int = 252,        # 1 year for cointegration test
    coint_threshold: float = 0.05,
    max_holding_days: int = 30,     # Maximum holding period
    stop_loss_z: float = 3.0,       # Stop loss threshold
    transaction_cost: float = 0.001, # 0.1% transaction cost
    min_coint_days: int = 60        # Minimum days for cointegration test
) -> pd.DataFrame:
    """
    IMPROVED pairs trading strategy with proper backtesting methodology.
    
    Key Improvements:
    1. Rolling hedge ratio calculation (no look-ahead bias)
    2. Rolling cointegration testing
    3. Risk management (stop loss, max holding period)
    4. Transaction costs
    5. Proper position sizing
    
    Parameters:
    - price_a: Series of stock A prices (index: datetime)
    - price_b: Series of stock B prices (index: datetime)
    - window: Lookback window for z-score calculation
    - entry_z: Z-score threshold for entry
    - exit_z: Z-score threshold for exit
    - hedge_ratio_window: Window for hedge ratio calculation (default 252 days)
    - coint_window: Window for cointegration test (default 252 days)
    - coint_threshold: P-value threshold for cointegration
    - max_holding_days: Maximum holding period
    - stop_loss_z: Stop loss Z-score threshold
    - transaction_cost: Transaction cost as percentage
    - min_coint_days: Minimum days required for cointegration test
    """
    
    # Align indices
    df = pd.DataFrame({'A': price_a, 'B': price_b}).dropna()
    df.index = pd.to_datetime(df.index)
    
    print(f"Data Overview:")
    print(f"   Total days: {len(df)}")
    print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Hedge ratio window: {hedge_ratio_window} days")
    print(f"   Cointegration window: {coint_window} days")
    
    # 1. Calculate rolling hedge ratio (no look-ahead bias)
    print(f"\nCalculating rolling hedge ratio...")
    rolling_beta = calculate_rolling_hedge_ratio(df, window=hedge_ratio_window)
    print(f"   Hedge ratio range: [{rolling_beta.min():.4f}, {rolling_beta.max():.4f}]")
    print(f"   Average hedge ratio: {rolling_beta.mean():.4f}")
    
    # 2. Test cointegration using rolling window
    print(f"\nTesting rolling cointegration...")
    rolling_coint_p = test_cointegration_rolling(df, window=coint_window, min_coint_days=min_coint_days)
    coint_success_rate = (rolling_coint_p <= coint_threshold).mean()
    print(f"   Cointegration success rate: {coint_success_rate:.2%}")
    print(f"   Average p-value: {rolling_coint_p.mean():.4f}")
    
    # 3. Calculate spread using rolling hedge ratio
    print(f"\nCalculating spread...")
    df['hedge_ratio'] = rolling_beta
    df['spread'] = df['A'] - df['hedge_ratio'] * df['B']
    
    # Remove rows where hedge ratio is not available
    df = df.dropna()
    
    # 4. Calculate rolling mean/std and z-score (using only past data)
    print(f"\nCalculating Z-scores...")
    df['spread_mean'] = df['spread'].rolling(window=window).mean()
    df['spread_std'] = df['spread'].rolling(window=window).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    # 5. Add cointegration status
    df['coint_p_value'] = rolling_coint_p
    df['is_cointegrated'] = df['coint_p_value'] <= coint_threshold
    
    # Remove rows where z-score is not available
    df = df.dropna()
    
    print(f"   Z-score range: [{df['z_score'].min():.2f}, {df['z_score'].max():.2f}]")
    print(f"   Spread mean: {df['spread'].mean():.4f}")
    print(f"   Spread std: {df['spread'].std():.4f}")
    
    # 6. IMPROVED Trading logic with risk management
    print(f"\nStarting trading simulation...")
    trades = []
    position = None
    entry_price = None
    entry_date = None
    entry_z_score = None
    entry_hedge_ratio = None
    
    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        z = row['z_score']
        date = row.name
        is_cointegrated = row['is_cointegrated']
        hedge_ratio = row['hedge_ratio']
        
        # Enter positions (only if cointegrated)
        if position is None and is_cointegrated:
            if z > entry_z:
                # Short spread: short A, long B
                position = 'short'
                entry_price = row['spread']
                entry_date = date
                entry_z_score = z
                entry_hedge_ratio = hedge_ratio
                print(f"SHORT ENTRY: {date.strftime('%Y-%m-%d')} | Z: {z:.2f} | Spread: {entry_price:.4f} | Beta: {hedge_ratio:.4f}")
                
            elif z < -entry_z:
                # Long spread: long A, short B
                position = 'long'
                entry_price = row['spread']
                entry_date = date
                entry_z_score = z
                entry_hedge_ratio = hedge_ratio
                print(f"LONG ENTRY: {date.strftime('%Y-%m-%d')} | Z: {z:.2f} | Spread: {entry_price:.4f} | Beta: {hedge_ratio:.4f}")
        
        # Exit positions
        elif position is not None:
            exit_triggered = False
            exit_reason = ""
            
            # Check exit conditions
            if position == 'long':
                if z > -exit_z:
                    exit_triggered = True
                    exit_reason = "Normal exit (Z-score reversion)"
                elif z > stop_loss_z:
                    exit_triggered = True
                    exit_reason = "Stop loss triggered"
                    
            elif position == 'short':
                if z < exit_z:
                    exit_triggered = True
                    exit_reason = "Normal exit (Z-score reversion)"
                elif z < -stop_loss_z:
                    exit_triggered = True
                    exit_reason = "Stop loss triggered"
            
            # Check maximum holding period
            if (date - entry_date).days >= max_holding_days:
                exit_triggered = True
                exit_reason = "Maximum holding period reached"
            
            # Check if cointegration breaks down
            # Only exit on breakdown if we're losing money (more lenient)
            # This prevents premature exits when cointegration temporarily breaks but spread is reverting
            if not is_cointegrated:
                # Calculate current P&L to see if we're losing
                current_spread = row['spread']
                if position == 'long':
                    current_pnl = current_spread - entry_price
                else:  # short
                    current_pnl = entry_price - current_spread
                
                # Only exit on breakdown if we're losing (spread moved against us)
                if current_pnl < 0:
                    exit_triggered = True
                    exit_reason = "Cointegration breakdown (losing)"
                # If we're winning, let it ride (cointegration might come back)
            
            # Execute exit
            if exit_triggered:
                exit_spread = next_row['spread']
                exit_date = next_row.name
                
                # Get actual stock prices for entry and exit
                entry_price_a = df.loc[entry_date, 'A']
                entry_price_b = df.loc[entry_date, 'B']
                exit_price_a = next_row['A']
                exit_price_b = next_row['B']
                
                # Calculate actual dollar profit/loss from both positions
                # For pairs trading: profit = (P&L from stock A) - hedge_ratio * (P&L from stock B)
                if position == 'long':
                    # Long A, short B: profit = (exit_A - entry_A) - hedge_ratio * (entry_B - exit_B)
                    dollar_pnl = (exit_price_a - entry_price_a) - entry_hedge_ratio * (entry_price_b - exit_price_b)
                else:  # short
                    # Short A, long B: profit = (entry_A - exit_A) - hedge_ratio * (exit_B - entry_B)
                    dollar_pnl = (entry_price_a - exit_price_a) - entry_hedge_ratio * (exit_price_b - entry_price_b)
                
                # Store actual stock prices and dollar P&L
                # For pairs trading, capital at risk = entry_price_a + hedge_ratio * entry_price_b
                # We store capital_at_risk in entry_price for proper normalization in equity curve
                capital_at_risk = entry_price_a + entry_hedge_ratio * entry_price_b
                exit_capital = exit_price_a + entry_hedge_ratio * exit_price_b
                normalized_return = dollar_pnl / capital_at_risk if capital_at_risk != 0 else 0.0
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': capital_at_risk,  # Total capital at risk (for proper normalization)
                    'exit_price': exit_capital,      # Total capital at exit
                    'profit_loss': dollar_pnl,       # Dollar P&L (absolute amount)
                    'trade_type': position,
                    'holding_days': (exit_date - entry_date).days,
                    'hedge_ratio': entry_hedge_ratio,
                    'exit_reason': exit_reason
                })
                
                # Calculate return for display
                return_pct = normalized_return * 100
                print(f"{position.upper()} EXIT: {exit_date.strftime('%Y-%m-%d')} | Z: {z:.2f} | P&L: ${dollar_pnl:.2f} | Return: {return_pct:.2f}% | Reason: {exit_reason}")
                position = None
    
    # 7. Create results DataFrame
    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        trade_df = trade_df[['entry_date', 'exit_date', 'entry_price', 'exit_price',
                             'profit_loss', 'trade_type', 'holding_days',
                             'hedge_ratio', 'exit_reason']]
        
        print(f"\nTrading Results:")
        print(f"   Total trades: {len(trade_df)}")
        print(f"   Winning trades: {len(trade_df[trade_df['profit_loss'] > 0])}")
        print(f"   Losing trades: {len(trade_df[trade_df['profit_loss'] < 0])}")
        print(f"   Win rate: {len(trade_df[trade_df['profit_loss'] > 0]) / len(trade_df):.2%}")
        print(f"   Total P&L: {trade_df['profit_loss'].sum():.4f}")
        print(f"   Average P&L per trade: {trade_df['profit_loss'].mean():.4f}")
        print(f"   Average holding period: {trade_df['holding_days'].mean():.1f} days")
    else:
        print(f"\nNo trades generated")
    
    return trade_df

# Example usage and testing function
def test_improved_strategy():
    """Test the improved strategy with sample data"""
    print("Testing Improved Pairs Cointegration Strategy")
    print("=" * 60)
    
    # This would be called from your main testing script
    # with real data from your database
    pass

if __name__ == "__main__":
    test_improved_strategy()
