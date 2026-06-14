import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Evaluator:
    def __init__(self, results_df):
        """
        Initialize with a results DataFrame containing:
        - Date (datetime index)
        - Strategy_Return (daily returns)
        - Benchmark_Return (e.g., Buy & Hold)
        """
        self.results = results_df

    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        excess_returns = self.results['Strategy_Return'] - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_max_drawdown(self):
        cumulative = (1 + self.results['Strategy_Return']).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def calculate_drawdown_duration(self):
        """Calculate the maximum drawdown duration in trading days"""
        cumulative = (1 + self.results['Strategy_Return']).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        
        # No drawdown case
        if not is_drawdown.any():
            return 0
        
        # Get start of each drawdown
        drawdown_start = (is_drawdown & ~is_drawdown.shift(1).fillna(False))
        drawdown_end = (~is_drawdown & is_drawdown.shift(1).fillna(False))
        
        # If we're still in a drawdown at the end of the data
        if is_drawdown.iloc[-1]:
            drawdown_end.iloc[-1] = True
        
        # Find the longest drawdown duration
        max_duration = 0
        for start_idx in drawdown_start[drawdown_start].index:
            end_idx = drawdown_end[drawdown_end.index > start_idx].index
            if len(end_idx) > 0:
                duration = (end_idx[0] - start_idx).days
                max_duration = max(max_duration, duration)
        
        return max_duration

    def profit_factor(self):
        returns = self.results['Strategy_Return']
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0]).sum()
        return gross_profit / gross_loss if gross_loss != 0 else np.inf

    def calculate_annualized_return(self):
        total_return = (1 + self.results['Strategy_Return']).prod() - 1
        return (1 + total_return) ** (252 / len(self.results)) - 1

    def calculate_sortino(self, risk_free_rate=0.02):
        returns = self.results['Strategy_Return']
        downside_returns = returns[returns < 0]
        
        # Handle edge cases
        if len(downside_returns) == 0:
            # No negative returns - strategy is perfect from downside perspective
            # Return a high but reasonable value instead of infinity
            excess_return = returns.mean() * 252 - risk_free_rate
            return 10.0 if excess_return > 0 else 0.0
        
        downside_dev = downside_returns.std() * np.sqrt(252)
        excess_return = returns.mean() * 252 - risk_free_rate
        
        # Handle very small downside deviation
        if downside_dev < 1e-6:  # Very small threshold
            return 10.0 if excess_return > 0 else 0.0
        
        sortino_ratio = excess_return / downside_dev
        
        # Cap the Sortino ratio to prevent extreme values
        return min(sortino_ratio, 10.0) if sortino_ratio > 0 else max(sortino_ratio, -10.0)

    def calculate_calmar_ratio(self):
        """Calculate Calmar Ratio: Annualized Return / Max Drawdown"""
        return self.calculate_annualized_return() / abs(self.calculate_max_drawdown())
    
    def calculate_omega_ratio(self, threshold=0):
        """
        Calculate Omega Ratio: probability-weighted ratio of gains versus losses
        for all returns falling above and below a threshold
        """
        returns = self.results['Strategy_Return']
        returns_above = returns[returns > threshold]
        returns_below = returns[returns <= threshold]
        
        if len(returns_below) == 0 or abs(returns_below.sum()) == 0:
            return np.inf
            
        return returns_above.sum() / abs(returns_below.sum())
    
    def calculate_alpha_beta(self):
        """Calculate Alpha and Beta relative to the benchmark"""
        x = self.results['Benchmark_Return']
        y = self.results['Strategy_Return']
        
        beta, alpha, _, _, _ = stats.linregress(x, y)
        # Convert alpha to annualized
        alpha = alpha * 252
        
        return alpha, beta
    
    def calculate_information_ratio(self):
        """Calculate Information Ratio: Excess returns over benchmark / Tracking error"""
        excess_returns = self.results['Strategy_Return'] - self.results['Benchmark_Return']
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return np.nan
            
        return excess_returns.mean() * 252 / tracking_error
    
    def calculate_avg_gain_loss(self):
        """Calculate average gain and average loss"""
        returns = self.results['Strategy_Return']
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        avg_gain = gains.mean() if len(gains) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        return avg_gain, avg_loss
    
    def calculate_gain_loss_ratio(self):
        """Calculate ratio of average gain to average loss"""
        avg_gain, avg_loss = self.calculate_avg_gain_loss()
        
        if avg_loss == 0:
            return np.inf
            
        return abs(avg_gain / avg_loss)
    
    def calculate_volatility(self):
        """Calculate annualized volatility"""
        return self.results['Strategy_Return'].std() * np.sqrt(252)
    
    def calculate_downside_deviation(self, threshold=0):
        """Calculate downside deviation below a threshold"""
        returns = self.results['Strategy_Return']
        downside_returns = returns[returns < threshold]
        
        if len(downside_returns) == 0:
            return 0
            
        return downside_returns.std() * np.sqrt(252)
    
    def rolling_sharpe(self, window=63):
        """Calculate rolling Sharpe ratio (default: 3-month window)"""
        excess_returns = self.results['Strategy_Return'] - 0.02/252  # Using 2% risk-free rate
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std()
        
        return np.sqrt(252) * rolling_mean / rolling_std

    def calculate_total_return(self):
        returns = self.results['Strategy_Return']
        return (1 + returns).prod() - 1

    def count_trades(self):
        return (self.results['Signal'].diff() != 0).sum() // 2  # Round-trip trades

    def generate_summary(self):
        alpha, beta = self.calculate_alpha_beta()
        avg_gain, avg_loss = self.calculate_avg_gain_loss()
        
        return {
            'Total Return': self.calculate_total_return(),
            'Sharpe Ratio': self.calculate_sharpe_ratio(),
            'Sortino Ratio': self.calculate_sortino(),
            'Max Drawdown': self.calculate_max_drawdown(),
            'Max Drawdown Duration (Days)': self.calculate_drawdown_duration(),
            'Annualized Return': self.calculate_annualized_return(),
            'Annualized Volatility': self.calculate_volatility(),
            'Win Rate': (self.results['Strategy_Return'] > 0).mean(),
            'Profit Factor': self.profit_factor(),
            'Calmar Ratio': self.calculate_calmar_ratio(),
            'Omega Ratio': self.calculate_omega_ratio(),
            'Alpha': alpha,
            'Beta': beta,
            'Information Ratio': self.calculate_information_ratio(),
            'Average Gain': avg_gain,
            'Average Loss': avg_loss,
            'Gain/Loss Ratio': self.calculate_gain_loss_ratio(),
        }

    def plot_equity_curve(self, save_path=None):
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, (1 + self.results['Strategy_Return']).cumprod(), label='Strategy')
        plt.plot(self.results.index, (1 + self.results['Benchmark_Return']).cumprod(), label='Benchmark')
        plt.title('Equity Curve')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_drawdown(self, save_path=None):
        cumulative = (1 + self.results['Strategy_Return']).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        plt.figure(figsize=(12, 6))
        plt.fill_between(self.results.index, drawdown, 0, alpha=0.3)
        plt.title('Strategy Drawdown')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_rolling_sharpe(self, window=63, save_path=None):
        """Plot rolling Sharpe ratio"""
        rolling_sharpe = self.rolling_sharpe(window)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index[window-1:], rolling_sharpe[window-1:])
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.title(f'{window}-day Rolling Sharpe Ratio')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_return_distribution(self, save_path=None):
        """Plot return distribution with normal curve overlay"""
        returns = self.results['Strategy_Return']
        
        plt.figure(figsize=(12, 6))
        plt.hist(returns, bins=50, alpha=0.6, density=True)
        
        # Plot normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        plt.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 'r-')
        
        plt.title('Return Distribution')
        if save_path:
            plt.savefig(save_path)
        plt.show()