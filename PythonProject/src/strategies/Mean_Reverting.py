import numpy as np
import pandas as pd
from Analysis.statistical_tests import run_adf_test, hurst_exponent
from Backtesting.Market_Analysis import MarketAnalyzer

pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.width', None)        # auto-expand to fit screen


def calculate_atr(data, window=14):
    """Calculate Average True Range"""
    high = data['high']
    low = data['low']
    close = data['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr


def mean_reversion_strategy(
        data: pd.DataFrame,
        window: int = 20,
        long_entry: float = -2.0,
        long_exit: float = 0.0,
        short_entry: float = 2.0,
        short_exit: float = 0.0,
        stop_loss_pct: float = 0.05,
        stop_loss_atr_multiple: float = 2.0,
        use_atr_stops: bool = True,
        max_holding_days: int = 10,
        z_score_scale_factor: float = 0.5,
        use_regime_filter: bool = True,
        regime_lookback: int = 50
) -> pd.DataFrame:
    """
    Enhanced mean-reversion trading with adaptive parameters.
    Returns trade_df with columns:
    ['entry_date', 'exit_date', 'entry_price', 'exit_price',
     'pnl', 'trade_type', 'holding_days', 'position_size', 'exit_reason']
    
    Parameters:
    - data: DataFrame with OHLCV data
    - window: Lookback window for moving average
    - long_entry: Z-score threshold for long entry
    - long_exit: Z-score threshold for long exit
    - short_entry: Z-score threshold for short entry  
    - short_exit: Z-score threshold for short exit
    - stop_loss_pct: Fixed percentage stop loss
    - stop_loss_atr_multiple: ATR multiple for dynamic stop loss
    - use_atr_stops: Whether to use ATR-based stops
    - max_holding_days: Maximum days to hold a position
    - z_score_scale_factor: Scale position size based on z-score magnitude
    - use_regime_filter: Whether to filter trades based on market regime
    - regime_lookback: Lookback period for regime detection
    """
    # Ensure datetime index
    data = data.copy()
    data.index = pd.to_datetime(data.index)

    # Calculate ATR for dynamic stop losses
    data['atr'] = calculate_atr(data)
    
    # Calculate spread and z-scores
    rolling_mean = data["close"].rolling(window=window).mean()
    spread = data["close"] - rolling_mean
    
    # Use expanding standard deviation after initial window to reduce outlier impact
    rolling_std = spread.rolling(window=window).std()
    data["z_score"] = spread / rolling_std
    
    # Calculate dynamic window based on volatility
    volatility = data['close'].pct_change().rolling(window=window).std()
    
    # Adjust window based on volatility - shorter windows in high volatility
    volatility_clean = volatility.replace([np.inf, -np.inf], np.nan).fillna(0)
    adaptive_window = np.maximum(10, np.round(window * (1 - volatility_clean * 10))).astype(int)
    data['adaptive_window'] = adaptive_window
    
    # Market regime filter using MarketAnalyzer if enough data
    if use_regime_filter and len(data) > regime_lookback:
        # Rename columns for MarketAnalyzer (expects uppercase)
        ohlcv_data = data.rename(columns={
            'open': 'Open', 
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close', 
            'volume': 'Volume'
        })
        
        # Create analyzer and detect regime
        analyzer = MarketAnalyzer(ohlcv_data)
        regimes = analyzer.detect_market_regime(window=min(regime_lookback, len(data) // 4))
        
        # Add regime information to data
        data['market_regime'] = regimes['regime']
        
        # Flag for favorable mean-reversion regimes (low volatility ranging or normal ranging)
        data['favorable_regime'] = data['market_regime'].apply(
            lambda x: x in ['low_ranging', 'normal_ranging']
        )
    else:
        # Default to favorable if not using regime filter
        data['favorable_regime'] = True
    
    data = data.dropna()

    trades = []
    position = None
    entry_price = None
    entry_date = None
    position_size = 1.0  # Default full position

    for i in range(len(data)):
        current_date = data.index[i]
        current_price = data["close"].iloc[i]
        current_z = data["z_score"].iloc[i]
        current_atr = data["atr"].iloc[i]
        favorable_regime = data["favorable_regime"].iloc[i]

        # ----------------------------------
        # Position Management - Handle existing positions
        # ----------------------------------
        if position is not None:
            holding_days = (current_date - entry_date).days
            
            # ----------------------------------
            # Exit Logic (Long Positions)
            # ----------------------------------
            if position == "long":
                # Dynamic stop loss based on ATR or fixed percentage
                if use_atr_stops:
                    stop_loss_price = entry_price - (stop_loss_atr_multiple * entry_atr)
                else:
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                
                exit_reason = "target"  # Default reason
                
                # Exit conditions - Z-score target, stop loss, or max holding period
                if current_z > long_exit:
                    exit_reason = "target"
                elif current_price < stop_loss_price:
                    exit_reason = "stop_loss"
                elif holding_days >= max_holding_days:
                    exit_reason = "time_exit"
                
                # Check if any exit condition met
                if (current_z > long_exit) or (current_price < stop_loss_price) or (holding_days >= max_holding_days):
                    pnl = current_price - entry_price
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "profit_loss": pnl,
                        "trade_type": "long",
                        "holding_days": holding_days,
                        "exit_reason": exit_reason
                    })
                    position = None  # Close position

            # ----------------------------------
            # Exit Logic (Short Positions)
            # ----------------------------------
            elif position == "short":
                # Dynamic stop loss
                if use_atr_stops:
                    stop_loss_price = entry_price + (stop_loss_atr_multiple * entry_atr)
                else:
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                
                exit_reason = "target"  # Default reason
                
                # Exit conditions
                if current_z < short_exit:
                    exit_reason = "target"
                elif current_price > stop_loss_price:
                    exit_reason = "stop_loss"
                elif holding_days >= max_holding_days:
                    exit_reason = "time_exit"
                
                # Check if any exit condition met
                if (current_z < short_exit) or (current_price > stop_loss_price) or (holding_days >= max_holding_days):
                    pnl = entry_price - current_price  # Correct short PnL
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "profit_loss": pnl,
                        "trade_type": "short",
                        "holding_days": holding_days,
                        "exit_reason": exit_reason
                    })
                    position = None  # Close position

        # ----------------------------------
        # Entry Logic - Only enter if no position
        # ----------------------------------
        if position is None:
            # Skip if regime not favorable and using regime filter
            if use_regime_filter and not favorable_regime:
                continue
                
            # Dynamic position sizing based on z-score
            # More extreme z-scores get larger position sizes
            z_score_magnitude = abs(current_z)
            
            # Long entry condition
            if current_z < long_entry:
                position = "long"
                entry_price = current_price
                entry_date = current_date
                entry_atr = current_atr
                
                # Position size fixed at 1.0 for simplicity

            # Short entry condition
            elif current_z > short_entry:
                position = "short"
                entry_price = current_price
                entry_date = current_date
                entry_atr = current_atr
                
                # Position size fixed at 1.0 for simplicity

    # Format output
    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        trade_df = trade_df[[
            "entry_date", "exit_date", "entry_price", "exit_price",
            "profit_loss", "trade_type", "holding_days", "exit_reason"
        ]]
    return trade_df


def run_mean_reversion_if_valid(
    data: pd.DataFrame, 
    window: int = 20,
    z_entry: float = 2.0, 
    z_exit: float = 0.5,
    use_atr_stops: bool = True, 
    stop_loss_atr_multiple: float = 2.0,
    max_holding_days: int = 10,
    use_regime_filter: bool = True
) -> pd.DataFrame:
    """
    Validates mean-reversion conditions and returns formatted trade_df
    
    Parameters:
    - data: DataFrame with OHLCV data
    - window: Lookback window for moving average
    - z_entry: Z-score threshold for entry (absolute value)
    - z_exit: Z-score threshold for exit (absolute value)
    - use_atr_stops: Whether to use ATR-based stops
    - stop_loss_atr_multiple: ATR multiple for dynamic stop loss
    - max_holding_days: Maximum days to hold a position
    - use_regime_filter: Whether to filter trades based on market regime
    """
    data = data.copy()

    # 1. Calculate spread
    data["rolling_mean"] = data["close"].rolling(window=window).mean()
    data["spread"] = data["close"] - data["rolling_mean"]
    
    # Need enough data for calculations
    if len(data) < window * 2:
        print("Not enough data for strategy calculations")
        return pd.DataFrame()
        
    data = data.dropna()

    # 2. Statistical validation
    spread_series = data["spread"]
    adf_result = run_adf_test(spread_series)
    hurst = hurst_exponent(spread_series.values)

    print(f"Validation - ADF p-value: {adf_result['p_value']:.4f}, Hurst: {hurst:.4f}")

    # 3. Execute if valid
    if adf_result["p_value"] < 0.05 and hurst < 0.5:
        print("Valid mean-reversion pattern detected")
        return mean_reversion_strategy(
            data=data,
            window=window,
            long_entry=-z_entry,
            long_exit=-z_exit,
            short_entry=z_entry,
            short_exit=z_exit,
            use_atr_stops=use_atr_stops,
            stop_loss_atr_multiple=stop_loss_atr_multiple,
            max_holding_days=max_holding_days,
            use_regime_filter=use_regime_filter
        )
    else:
        print("Invalid mean-reversion conditions")
        return pd.DataFrame()  # Empty but properly formatted


def optimize_mean_reversion_parameters(
    data: pd.DataFrame,
    window_range=(10, 50, 10),
    z_entry_range=(1.0, 3.0, 0.5),
    z_exit_range=(0.0, 1.0, 0.25),
    stop_loss_atr_range=(1.0, 3.0, 0.5),
    max_holding_range=(5, 20, 5)
) -> dict:
    """
    Optimize strategy parameters based on Sharpe ratio
    
    Parameters:
    - data: DataFrame with OHLCV data
    - window_range: Tuple of (min, max, step) for window parameter
    - z_entry_range: Tuple of (min, max, step) for z_entry parameter
    - z_exit_range: Tuple of (min, max, step) for z_exit parameter
    - stop_loss_atr_range: Tuple of (min, max, step) for stop_loss_atr_multiple
    - max_holding_range: Tuple of (min, max, step) for max_holding_days
    
    Returns:
    - Dictionary with best parameters and performance metrics
    """
    from src.Backtesting.Metrics import Evaluator
    
    results = []
    
    # Generate parameter combinations
    windows = list(range(window_range[0], window_range[1] + 1, window_range[2]))
    z_entries = np.arange(z_entry_range[0], z_entry_range[1] + 0.01, z_entry_range[2])
    z_exits = np.arange(z_exit_range[0], z_exit_range[1] + 0.01, z_exit_range[2])
    atr_multiples = np.arange(stop_loss_atr_range[0], stop_loss_atr_range[1] + 0.01, stop_loss_atr_range[2])
    max_holdings = list(range(max_holding_range[0], max_holding_range[1] + 1, max_holding_range[2]))
    
    # Store best result
    best_sharpe = -float('inf')
    best_params = None
    best_trades = None
    
    total_combinations = len(windows) * len(z_entries) * len(z_exits) * len(atr_multiples) * len(max_holdings)
    print(f"Starting optimization with {total_combinations} parameter combinations")
    
    # Limit to a reasonable number of combinations for performance
    if total_combinations > 500:
        print(f"Warning: Large parameter space ({total_combinations} combinations). This may take a while.")
    
    # Simple progress counter
    count = 0
    
    for window in windows:
        # Calculate common data components once per window to save time
        window_data = data.copy()
        window_data["rolling_mean"] = window_data["close"].rolling(window=window).mean()
        window_data["spread"] = window_data["close"] - window_data["rolling_mean"]
        window_data = window_data.dropna()
        
        # Skip if not a valid mean-reversion series
        spread_series = window_data["spread"]
        adf_result = run_adf_test(spread_series)
        hurst = hurst_exponent(spread_series.values)
        
        if adf_result["p_value"] >= 0.05 or hurst >= 0.5:
            print(f"Skipping window={window}, not mean-reverting (ADF p={adf_result['p_value']:.4f}, Hurst={hurst:.4f})")
            continue
        
        for z_entry in z_entries:
            for z_exit in z_exits:
                # Skip invalid combinations (exit > entry)
                if z_exit >= z_entry:
                    continue
                    
                for atr_multiple in atr_multiples:
                    for max_holding in max_holdings:
                        count += 1
                        if count % 50 == 0:
                            print(f"Progress: {count}/{total_combinations} combinations tested")
                        
                        # Run strategy with current parameters
                        trades = mean_reversion_strategy(
                            data=window_data,
                            window=window,
                            long_entry=-z_entry,
                            long_exit=-z_exit,
                            short_entry=z_entry,
                            short_exit=z_exit,
                            use_atr_stops=True,
                            stop_loss_atr_multiple=atr_multiple,
                            max_holding_days=max_holding
                        )
                        
                        # Skip if no trades generated
                        if trades.empty or len(trades) < 5:  # Require at least 5 trades
                            continue
                            
                        # Calculate daily strategy returns for Sharpe calculation
                        strategy_returns = pd.Series(0, index=data.index, dtype=float)
                        
                        for _, trade in trades.iterrows():
                            holding_period = data.loc[
                                (data.index >= trade["entry_date"]) & (data.index < trade["exit_date"])
                            ].index
                            
                            if not holding_period.empty:
                                # Account for position size
                                position_size = trade.get("position_size", 1.0)
                                daily_return = (trade["profit_loss"] / position_size / len(holding_period)) / trade["entry_price"] * position_size
                                strategy_returns.loc[holding_period] = daily_return
                        
                        # Create benchmark returns
                        data_copy = data.copy()
                        data_copy["Benchmark_Return"] = data_copy["close"].pct_change().fillna(0)
                        
                        # Build results DataFrame for Evaluator
                        results_df = pd.DataFrame({
                            "Strategy_Return": strategy_returns,
                            "Benchmark_Return": data_copy["Benchmark_Return"],
                            "Signal": 0  # Placeholder for signal
                        })
                        
                        # Skip if not enough return data
                        if results_df.empty:
                            continue
                            
                        # Evaluate performance
                        evaluator = Evaluator(results_df)
                        sharpe = evaluator.calculate_sharpe_ratio()
                        sortino = evaluator.calculate_sortino()
                        max_dd = evaluator.calculate_max_drawdown()
                        total_return = evaluator.calculate_total_return()
                        
                        # Record results
                        param_result = {
                            "window": window,
                            "z_entry": z_entry,
                            "z_exit": z_exit, 
                            "stop_loss_atr": atr_multiple,
                            "max_holding": max_holding,
                            "sharpe": sharpe,
                            "sortino": sortino,
                            "max_drawdown": max_dd,
                            "total_return": total_return,
                            "trade_count": len(trades),
                            "win_rate": (trades["profit_loss"] > 0).mean()
                        }
                        
                        results.append(param_result)
                        
                        # Update best result
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {
                                "window": window,
                                "z_entry": z_entry,
                                "z_exit": z_exit,
                                "stop_loss_atr": atr_multiple,
                                "max_holding": max_holding
                            }
                            best_trades = trades
    
    # Sort results by Sharpe ratio
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("sharpe", ascending=False)
        
        print("\nTop 5 Parameter Combinations:")
        print(results_df.head(5))
        
        best_row = results_df.iloc[0]
        
        return {
            "best_params": best_params,
            "best_trades": best_trades,
            "best_metrics": {
                "sharpe": best_row["sharpe"],
                "sortino": best_row["sortino"],
                "max_drawdown": best_row["max_drawdown"],
                "total_return": best_row["total_return"],
                "trade_count": best_row["trade_count"],
                "win_rate": best_row["win_rate"]
            },
            "all_results": results_df
        }
    else:
        return {
            "best_params": None,
            "best_trades": None,
            "best_metrics": None,
            "all_results": None
        }