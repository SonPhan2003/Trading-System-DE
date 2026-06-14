import warnings
import pandas as pd
import numpy as np
import streamlit as st

from Data.Loader import load_stock_data

# Strategy imports with safe fallbacks
try:
    from strategies.Mean_Reverting import run_mean_reversion_if_valid
except Exception as e:
    run_mean_reversion_if_valid = None
    MEAN_REV_IMPORT_ERR = e
else:
    MEAN_REV_IMPORT_ERR = None

try:
    from strategies.Bollinger_RSI_strategy import run_bollinger_rsi_if_valid
except Exception as e:
    run_bollinger_rsi_if_valid = None
    BB_RSI_IMPORT_ERR = e
else:
    BB_RSI_IMPORT_ERR = None

try:
    from strategies.Pairs_Cointegration import (
        pairs_trading_cointegration_strategy_improved,
    )
except Exception as e:
    pairs_trading_cointegration_strategy_improved = None
    PAIRS_IMPORT_ERR = e
else:
    PAIRS_IMPORT_ERR = None

try:
    from strategies.LSTM_strategy import lstm_predict_on_dates
except Exception as e:
    lstm_predict_on_dates = None
    LSTM_IMPORT_ERR = e
else:
    LSTM_IMPORT_ERR = None

try:
    from strategies.Random_Forest_strategy import random_forest_strategy
except Exception as e:
    random_forest_strategy = None
    RF_IMPORT_ERR = e
else:
    RF_IMPORT_ERR = None

from Backtesting.Metrics import Evaluator


def convert_trades_to_daily_returns(trades: pd.DataFrame, price_index: pd.DatetimeIndex, initial_capital: float = None) -> pd.Series:
    """Create a daily return series from trade logs.
    Apply the full trade P&L on the exit date.
    Assumes non-overlapping trades (as in our strategies).
    
    For pairs trading, if initial_capital is provided, returns are normalized by initial_capital
    instead of each trade's capital at risk, which provides a more accurate equity curve.
    """
    returns = pd.Series(0.0, index=price_index)
    if trades is None or trades.empty:
        return returns

    # For pairs trading, use a fixed initial capital if provided
    # This ensures returns are normalized consistently for equity curve calculation
    if initial_capital is None and 'hedge_ratio' in trades.columns:
        # Pairs trading detected - use first trade's capital at risk as initial capital
        # This provides consistent normalization: all returns are relative to initial capital
        # The equity curve will show cumulative returns on the initial capital allocation
        if len(trades) > 0:
            initial_capital = float(trades.iloc[0]['entry_price'])
        else:
            initial_capital = 10000.0  # Fallback
    
    for _, t in trades.iterrows():
        exit_date = pd.to_datetime(t["exit_date"])
        if exit_date in returns.index:
            profit_loss = float(t.get("profit_loss", 0.0))
            
            if initial_capital is not None and initial_capital > 0:
                # Normalize by fixed initial capital (for pairs trading)
                returns.loc[exit_date] += profit_loss / initial_capital
            else:
                # Normalize by entry price (for single-stock strategies)
                entry_price = float(t.get("entry_price", np.nan))
                if np.isfinite(entry_price) and entry_price != 0.0:
                    returns.loc[exit_date] += profit_loss / entry_price
                else:
                    returns.loc[exit_date] += profit_loss
    return returns.fillna(0.0)


def evaluate_strategy_from_trades(trades: pd.DataFrame, benchmark_close: pd.Series) -> tuple[pd.DataFrame, Evaluator]:
    strategy_returns = convert_trades_to_daily_returns(trades, benchmark_close.index)
    benchmark_returns = benchmark_close.pct_change().fillna(0.0)
    results_df = pd.DataFrame(
        {
            "Strategy_Return": strategy_returns,
            "Benchmark_Return": benchmark_returns,
            "Signal": 0,
        },
        index=benchmark_close.index,
    )
    evaluator = Evaluator(results_df)
    return results_df, evaluator


def run_mean_reversion(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    if run_mean_reversion_if_valid is None:
        raise ImportError(MEAN_REV_IMPORT_ERR)
    df = load_stock_data(symbol)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    trades = run_mean_reversion_if_valid(df)
    return trades


def run_bollinger_rsi(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    if run_bollinger_rsi_if_valid is None:
        raise ImportError(BB_RSI_IMPORT_ERR)
    df = load_stock_data(symbol)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    trades = run_bollinger_rsi_if_valid(df)
    return trades


def run_pairs_cointegration(
    symbol_a: str,
    symbol_b: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    window: int = 60,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
    hedge_ratio_window: int = 252,
    coint_window: int = 252,
    coint_threshold: float = 0.05,
    max_holding_days: int = 30,
    stop_loss_z: float = 3.5,
    min_coint_days: int = 60,
):
    if pairs_trading_cointegration_strategy_improved is None:
        raise ImportError(PAIRS_IMPORT_ERR)
    df_a = load_stock_data(symbol_a)
    df_b = load_stock_data(symbol_b)
    df_a = df_a[(df_a.index >= start_date) & (df_a.index <= end_date)]
    df_b = df_b[(df_b.index >= start_date) & (df_b.index <= end_date)]
    price_a = df_a["close"]
    price_b = df_b["close"]
    trades = pairs_trading_cointegration_strategy_improved(
        price_a=price_a,
        price_b=price_b,
        window=window,
        entry_z=entry_z,
        exit_z=exit_z,
        hedge_ratio_window=hedge_ratio_window,
        coint_window=coint_window,
        coint_threshold=coint_threshold,
        max_holding_days=max_holding_days,
        stop_loss_z=stop_loss_z,
        min_coint_days=min_coint_days,
    )
    return trades


def run_lstm(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    if lstm_predict_on_dates is None:
        raise ImportError(LSTM_IMPORT_ERR)
    # Relaxed parameters to generate more trades
    trades = lstm_predict_on_dates(
        symbol, start_date, end_date,
        threshold=0.0015,  # Lower threshold for more signals (default was 0.001)
        max_holding_days=20,  # Longer holding period (default was 15)
        stop_loss_pct=0.04,  # Wider stop loss (default was 0.03)
        take_profit_pct=0.06  # Higher take profit (default was 0.05)
    )
    return trades


def run_random_forest(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    if random_forest_strategy is None:
        raise ImportError(RF_IMPORT_ERR)
    df = load_stock_data(symbol)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    trades, _ = random_forest_strategy(df)
    return trades


def show_metrics_block(name: str, trades: pd.DataFrame, benchmark_close: pd.Series):
    st.subheader(f"📊 {name} Strategy Analysis")
    
    # Trade information
    trade_count = 0 if trades is None or trades.empty else len(trades)
    st.write(f"**Total Trades Generated:** {trade_count}")
    
    if trade_count == 0:
        st.warning("⚠️ **No trades generated!** This strategy didn't find any trading opportunities.")
        st.info("💡 This could be due to:")
        st.info("• Strategy parameters being too strict")
        st.info("• Market conditions not meeting strategy criteria")
        st.info("• Data issues or insufficient lookback period")
    else:
        st.subheader("📋 Recent Trades")
        st.dataframe(trades.tail(10))
    
    # Calculate metrics
    results_df, evaluator = evaluate_strategy_from_trades(trades, benchmark_close)
    summary = evaluator.generate_summary()
    
    # Display metrics table
    st.subheader("📈 Performance Metrics")
    st.dataframe(pd.DataFrame(summary, index=[name]).T.rename(columns={name: "Value"}))
    
    # Create comprehensive charts
    strat_cum = (1 + results_df["Strategy_Return"]).cumprod()
    bench_cum = (1 + results_df["Benchmark_Return"]).cumprod()
    
    # 1. Equity Curve Comparison
    st.subheader("📈 Equity Curve Comparison")
    eq_df = pd.DataFrame({
        name: strat_cum,
        "Benchmark": bench_cum,
    })
    st.line_chart(eq_df)
    final_strat_return = (strat_cum.iloc[-1] - 1) * 100
    final_bench_return = (bench_cum.iloc[-1] - 1) * 100
    if final_strat_return > final_bench_return:
        st.caption(f"✅ Strategy outperformed benchmark: {final_strat_return:.2f}% vs {final_bench_return:.2f}%. Higher line = better performance.")
    else:
        st.caption(f"⚠️ Strategy underperformed benchmark: {final_strat_return:.2f}% vs {final_bench_return:.2f}%. Higher line = better performance.")
    
    # 2. Drawdown Analysis
    st.subheader("📉 Drawdown Analysis")
    strat_dd = strat_cum / strat_cum.cummax() - 1
    bench_dd = bench_cum / bench_cum.cummax() - 1
    dd_df = pd.DataFrame({f"{name} Drawdown": strat_dd, "Benchmark Drawdown": bench_dd})
    st.area_chart(dd_df)
    max_dd_strat = strat_dd.min() * 100
    max_dd_bench = bench_dd.min() * 100
    if max_dd_strat > max_dd_bench:
        st.caption(f"✅ Strategy has better risk control: {max_dd_strat:.2f}% max drawdown vs {max_dd_bench:.2f}%. Closer to 0% = better.")
    else:
        st.caption(f"⚠️ Strategy has higher risk: {max_dd_strat:.2f}% max drawdown vs {max_dd_bench:.2f}%. Closer to 0% = better.")
    
    # 3. Rolling Sharpe Ratio
    st.subheader("⚡ Rolling Sharpe Ratio (30-day window)")
    rolling_sharpe = results_df["Strategy_Return"].rolling(30).mean() / results_df["Strategy_Return"].rolling(30).std() * np.sqrt(252)
    bench_rolling_sharpe = results_df["Benchmark_Return"].rolling(30).mean() / results_df["Benchmark_Return"].rolling(30).std() * np.sqrt(252)
    sharpe_df = pd.DataFrame({
        f"{name} Sharpe": rolling_sharpe,
        "Benchmark Sharpe": bench_rolling_sharpe
    })
    st.line_chart(sharpe_df)
    avg_sharpe_strat = rolling_sharpe.mean()
    avg_sharpe_bench = bench_rolling_sharpe.mean()
    if avg_sharpe_strat >= 1.0:
        st.caption(f"✅ Strategy Sharpe ratio: {avg_sharpe_strat:.2f} vs benchmark {avg_sharpe_bench:.2f}. ≥1.0 = good, ≥2.0 = excellent. Higher = better risk-adjusted returns.")
    else:
        st.caption(f"⚠️ Strategy Sharpe ratio: {avg_sharpe_strat:.2f} vs benchmark {avg_sharpe_bench:.2f}. ≥1.0 = good, ≥2.0 = excellent. Higher = better risk-adjusted returns.")
    
    # 4. Rolling Volatility
    st.subheader("📊 Rolling Volatility (30-day window)")
    rolling_vol = results_df["Strategy_Return"].rolling(30).std() * np.sqrt(252) * 100
    bench_rolling_vol = results_df["Benchmark_Return"].rolling(30).std() * np.sqrt(252) * 100
    vol_df = pd.DataFrame({
        f"{name} Volatility (%)": rolling_vol,
        "Benchmark Volatility (%)": bench_rolling_vol
    })
    st.line_chart(vol_df)
    avg_vol_strat = rolling_vol.mean()
    avg_vol_bench = bench_rolling_vol.mean()
    if avg_vol_strat < avg_vol_bench:
        st.caption(f"✅ Strategy volatility: {avg_vol_strat:.2f}% vs benchmark {avg_vol_bench:.2f}%. Lower volatility with same/higher returns = better.")
    else:
        st.caption(f"ℹ️ Strategy volatility: {avg_vol_strat:.2f}% vs benchmark {avg_vol_bench:.2f}%. Lower volatility with same/higher returns = better.")
    
    # 5. Monthly Returns Heatmap
    st.subheader("📅 Monthly Returns Comparison")
    monthly_strat = results_df["Strategy_Return"].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_bench = results_df["Benchmark_Return"].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_df = pd.DataFrame({
        f"{name} (%)": monthly_strat,
        "Benchmark (%)": monthly_bench
    })
    st.line_chart(monthly_df)
    positive_months = (monthly_strat > 0).sum()
    total_months = len(monthly_strat)
    win_rate = (positive_months / total_months) * 100 if total_months > 0 else 0
    st.caption(f"Strategy had {positive_months}/{total_months} positive months ({win_rate:.1f}% win rate). More positive months = more consistent performance.")
    


def app():
    st.title("Strategy Runner and Comparison")

    # Sidebar inputs
    st.sidebar.header("Inputs")
    symbol = st.sidebar.text_input("Primary symbol", value="AMZN")
    symbol_b = st.sidebar.text_input("Pair symbol (for Pairs Cointegration)", value="MSFT")
    start_date = pd.to_datetime(st.sidebar.date_input("Start date", value=pd.to_datetime("2021-01-01")).strftime("%Y-%m-%d"))
    end_date = pd.to_datetime(st.sidebar.date_input("End date", value=pd.to_datetime("2023-12-31")).strftime("%Y-%m-%d"))

    mode = st.sidebar.radio("Mode", ["Run single", "Compare all"]) 
    single_choice = st.sidebar.selectbox(
        "Strategy (single mode)",
        [
            "Mean Reverting",
            "Bollinger + RSI",
            "Pairs Cointegration",
            "LSTM",
            "Random Forest",
        ],
    )

    # Benchmark - always use SPY
    st.sidebar.markdown("---")
    bench_symbol = "SPY"  # Always use SPY as benchmark

    # Advanced parameters for Pairs Cointegration
    pairs_expander = st.sidebar.expander("Pairs Cointegration Settings", expanded=False)
    pc_window = int(pairs_expander.number_input("Z-score window (days)", min_value=20, max_value=252, value=60, step=5))
    pc_entry_z = float(pairs_expander.number_input("Entry Z-score threshold", min_value=0.5, max_value=3.5, value=1.5, step=0.1))
    pc_exit_z = float(pairs_expander.number_input("Exit Z-score threshold", min_value=0.05, max_value=1.0, value=0.3, step=0.05))
    pc_coint_threshold = float(pairs_expander.number_input("Cointegration p-value threshold", min_value=0.01, max_value=0.5, value=0.05, step=0.01))
    pc_hedge_window = int(pairs_expander.number_input("Hedge ratio window (days)", min_value=60, max_value=252, value=252, step=10))
    pc_coint_window = int(pairs_expander.number_input("Cointegration test window (days)", min_value=60, max_value=252, value=252, step=10))
    pc_min_coint_days = int(pairs_expander.number_input("Minimum cointegration days", min_value=20, max_value=120, value=60, step=5))
    pc_max_holding = int(pairs_expander.number_input("Max holding days", min_value=5, max_value=90, value=30, step=5))
    pc_stop_loss_z = float(pairs_expander.number_input("Stop loss Z-score", min_value=1.0, max_value=6.0, value=3.5, step=0.1))

    pair_params = {
        "window": pc_window,
        "entry_z": pc_entry_z,
        "exit_z": pc_exit_z,
        "hedge_ratio_window": pc_hedge_window,
        "coint_window": pc_coint_window,
        "coint_threshold": pc_coint_threshold,
        "max_holding_days": pc_max_holding,
        "stop_loss_z": pc_stop_loss_z,
        "min_coint_days": pc_min_coint_days,
    }

    if MEAN_REV_IMPORT_ERR:
        st.sidebar.warning(f"Mean_Reverting import issue: {MEAN_REV_IMPORT_ERR}")
    if BB_RSI_IMPORT_ERR:
        st.sidebar.warning(f"Bollinger_RSI import issue: {BB_RSI_IMPORT_ERR}")
    if PAIRS_IMPORT_ERR:
        st.sidebar.warning(f"Pairs_Cointegration import issue: {PAIRS_IMPORT_ERR}")
    if LSTM_IMPORT_ERR:
        st.sidebar.warning(f"LSTM import issue: {LSTM_IMPORT_ERR}")
    if RF_IMPORT_ERR:
        st.sidebar.warning(f"Random Forest import issue: {RF_IMPORT_ERR}")

    if st.sidebar.button("Run"):
        # Load benchmark (market index like SPY, QQQ, etc.)
        bench_df = load_stock_data(bench_symbol)
        bench_df = bench_df[(bench_df.index >= start_date) & (bench_df.index <= end_date)]
        if bench_df.empty:
            st.error("No data for benchmark symbol in selected range.")
            return
        benchmark_close = bench_df["close"]
        
        # Debug: Show benchmark info
        st.sidebar.info(f"Benchmark: {bench_symbol}")
        st.sidebar.info(f"Benchmark data points: {len(benchmark_close)}")
        st.sidebar.info(f"Benchmark range: {benchmark_close.index[0]} to {benchmark_close.index[-1]}")
        st.sidebar.info(f"Benchmark return: {((benchmark_close.iloc[-1] / benchmark_close.iloc[0]) - 1) * 100:.2f}%")

        if mode == "Run single":
            try:
                if single_choice == "Mean Reverting":
                    trades = run_mean_reversion(symbol, start_date, end_date)
                    show_metrics_block("Mean Reverting", trades, benchmark_close)
                elif single_choice == "Bollinger + RSI":
                    trades = run_bollinger_rsi(symbol, start_date, end_date)
                    show_metrics_block("Bollinger + RSI", trades, benchmark_close)
                elif single_choice == "Pairs Cointegration":
                    trades = run_pairs_cointegration(
                        symbol,
                        symbol_b,
                        start_date,
                        end_date,
                        **pair_params,
                    )
                    show_metrics_block("Pairs Cointegration", trades, benchmark_close)
                elif single_choice == "LSTM":
                    trades = run_lstm(symbol, start_date, end_date)
                    show_metrics_block("LSTM", trades, benchmark_close)
                elif single_choice == "Random Forest":
                    trades = run_random_forest(symbol, start_date, end_date)
                    show_metrics_block("Random Forest", trades, benchmark_close)
            except Exception as e:
                st.exception(e)

        else:  # Compare all
            results = []
            # Mean Reverting
            if run_mean_reversion_if_valid is not None:
                try:
                    mr_trades = run_mean_reversion(symbol, start_date, end_date)
                    mr_results, mr_eval = evaluate_strategy_from_trades(mr_trades, benchmark_close)
                    results.append(("Mean Reverting", mr_results, mr_eval))
                except Exception as e:
                    st.warning(f"Mean Reverting failed: {e}")
            # Bollinger + RSI
            if run_bollinger_rsi_if_valid is not None:
                try:
                    bb_trades = run_bollinger_rsi(symbol, start_date, end_date)
                    bb_results, bb_eval = evaluate_strategy_from_trades(bb_trades, benchmark_close)
                    results.append(("Bollinger + RSI", bb_results, bb_eval))
                except Exception as e:
                    st.warning(f"Bollinger + RSI failed: {e}")
            # Pairs Cointegration
            if pairs_trading_cointegration_strategy_improved is not None:
                try:
                    pc_trades = run_pairs_cointegration(
                        symbol,
                        symbol_b,
                        start_date,
                        end_date,
                        **pair_params,
                    )
                    pc_results, pc_eval = evaluate_strategy_from_trades(pc_trades, benchmark_close)
                    results.append(("Pairs Cointegration", pc_results, pc_eval))
                except Exception as e:
                    st.warning(f"Pairs Cointegration failed: {e}")
            # LSTM
            if lstm_predict_on_dates is not None:
                try:
                    ls_trades = run_lstm(symbol, start_date, end_date)
                    ls_results, ls_eval = evaluate_strategy_from_trades(ls_trades, benchmark_close)
                    results.append(("LSTM", ls_results, ls_eval))
                except Exception as e:
                    st.warning(f"LSTM failed: {e}")
            # Random Forest
            if random_forest_strategy is not None:
                try:
                    rf_trades = run_random_forest(symbol, start_date, end_date)
                    rf_results, rf_eval = evaluate_strategy_from_trades(rf_trades, benchmark_close)
                    results.append(("Random Forest", rf_results, rf_eval))
                except Exception as e:
                    st.warning(f"Random Forest failed: {e}")

            # Summary table
            if results:
                summaries = {}
                for name, _res, evalr in results:
                    summaries[name] = evalr.generate_summary()
                summary_df = pd.DataFrame(summaries).T
                
                st.subheader("📊 Strategy Comparison Summary")
                st.dataframe(summary_df)
                
                # Create comprehensive comparison charts
                st.subheader("📈 Equity Curves Comparison")
                eq = pd.concat(
                    [
                        (1 + r["Strategy_Return"]).cumprod().rename(n)
                        for (n, r, _e) in results
                    ],
                    axis=1,
                )
                # Add benchmark to equity curve
                bench_cum = (1 + results[0][1]["Benchmark_Return"]).cumprod()
                eq["Benchmark"] = bench_cum
                st.line_chart(eq)
                st.caption("Cumulative returns comparison. Higher is better.")
                
                # Drawdown comparison
                st.subheader("📉 Drawdown Comparison")
                dd_comparison = pd.DataFrame()
                for name, res, _eval in results:
                    strat_cum = (1 + res["Strategy_Return"]).cumprod()
                    dd_comparison[f"{name} Drawdown"] = strat_cum / strat_cum.cummax() - 1
                # Add benchmark drawdown
                dd_comparison["Benchmark Drawdown"] = bench_cum / bench_cum.cummax() - 1
                st.area_chart(dd_comparison)
                st.caption("Drawdown comparison. Closer to 0 is better.")
                
                # Sharpe Ratio comparison
                st.subheader("⚡ Sharpe Ratio Comparison")
                sharpe_data = {}
                for name, res, evalr in results:
                    summary = evalr.generate_summary()
                    sharpe_data[name] = summary.get('Sharpe Ratio', 0)
                # Add benchmark Sharpe
                bench_sharpe = results[0][1]["Benchmark_Return"].mean() / results[0][1]["Benchmark_Return"].std() * np.sqrt(252)
                sharpe_data["Benchmark"] = bench_sharpe
                sharpe_df = pd.DataFrame(list(sharpe_data.items()), columns=['Strategy', 'Sharpe Ratio'])
                sharpe_df = sharpe_df.set_index('Strategy')
                st.bar_chart(sharpe_df)
                st.caption("Risk-adjusted returns. Higher is better. Above 1.0 is good.")
                
                # Volatility comparison
                st.subheader("📊 Volatility Comparison")
                vol_data = {}
                for name, res, evalr in results:
                    summary = evalr.generate_summary()
                    # Use 'Annualized Volatility' key instead of 'Volatility'
                    vol_data[name] = summary.get('Annualized Volatility', 0) * 100
                # Add benchmark volatility
                bench_vol = results[0][1]["Benchmark_Return"].std() * np.sqrt(252) * 100
                vol_data["Benchmark"] = bench_vol
                
                # Debug: Show actual volatility values
                st.write("**Actual Volatility Values:**")
                for name, vol in vol_data.items():
                    st.write(f"{name}: {vol:.4f}%")
                
                vol_df = pd.DataFrame(list(vol_data.items()), columns=['Strategy', 'Volatility (%)'])
                vol_df = vol_df.set_index('Strategy')
                st.bar_chart(vol_df)
                st.caption("Annualized volatility. Lower is generally better for same returns.")
                
                # Debug: Show which strategies have very low volatility
                low_vol_strategies = [name for name, vol in vol_data.items() if vol < 0.1 and name != "Benchmark"]
                if low_vol_strategies:
                    st.warning(f"⚠️ These strategies have very low volatility: {', '.join(low_vol_strategies)}")
                    st.info("💡 Very low volatility might indicate few trades or very consistent returns.")
                
                # Debug: Show Sortino ratio details
                st.subheader("🔍 Sortino Ratio Analysis")
                for name, res, evalr in results:
                    summary = evalr.generate_summary()
                    sortino = summary.get('Sortino Ratio', 0)
                    returns = res["Strategy_Return"]
                    negative_returns = returns[returns < 0]
                    st.write(f"**{name}:**")
                    st.write(f"  - Sortino Ratio: {sortino:.4f}")
                    st.write(f"  - Total returns: {len(returns)}")
                    st.write(f"  - Negative returns: {len(negative_returns)}")
                    if len(negative_returns) > 0:
                        st.write(f"  - Downside deviation: {negative_returns.std() * np.sqrt(252):.6f}")
                    else:
                        st.write(f"  - No negative returns (perfect downside performance)")
                    st.write("")
                
                # Max Drawdown comparison
                st.subheader("📉 Maximum Drawdown Comparison")
                mdd_data = {}
                for name, res, evalr in results:
                    summary = evalr.generate_summary()
                    mdd_data[name] = summary.get('Max Drawdown', 0) * 100
                # Add benchmark max drawdown
                bench_mdd = (bench_cum / bench_cum.cummax() - 1).min() * 100
                mdd_data["Benchmark"] = bench_mdd
                mdd_df = pd.DataFrame(list(mdd_data.items()), columns=['Strategy', 'Max Drawdown (%)'])
                mdd_df = mdd_df.set_index('Strategy')
                st.bar_chart(mdd_df)
                st.caption("Maximum loss from peak. Lower (less negative) is better.")
                
                # Total Return comparison
                st.subheader("💰 Total Return Comparison")
                return_data = {}
                for name, res, evalr in results:
                    summary = evalr.generate_summary()
                    return_data[name] = summary.get('Total Return', 0) * 100
                # Add benchmark total return
                bench_return = (bench_cum.iloc[-1] - 1) * 100
                return_data["Benchmark"] = bench_return
                return_df = pd.DataFrame(list(return_data.items()), columns=['Strategy', 'Total Return (%)'])
                return_df = return_df.set_index('Strategy')
                st.bar_chart(return_df)
                st.caption("Total return over the period. Higher is better.")
                
                # Risk-Return Scatter Plot
                st.subheader("🎯 Risk-Return Scatter Plot")
                scatter_data = []
                for name, res, evalr in results:
                    summary = evalr.generate_summary()
                    scatter_data.append({
                        'Strategy': name,
                        'Return (%)': summary.get('Total Return', 0) * 100,
                        'Volatility (%)': summary.get('Volatility', 0) * 100,
                        'Sharpe': summary.get('Sharpe Ratio', 0)
                    })
                # Add benchmark
                scatter_data.append({
                    'Strategy': 'Benchmark',
                    'Return (%)': bench_return,
                    'Volatility (%)': bench_vol,
                    'Sharpe': bench_sharpe
                })
                scatter_df = pd.DataFrame(scatter_data)
                
                # Create scatter plot using plotly
                import plotly.express as px
                fig = px.scatter(scatter_df, x='Volatility (%)', y='Return (%)', 
                               text='Strategy', title='Risk vs Return Analysis',
                               hover_data=['Sharpe'])
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Risk-return analysis. Top-left quadrant (high return, low risk) is ideal.")
                
            else:
                st.info("No strategies ran successfully.")


if __name__ == "__main__":
    app()


