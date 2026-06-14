import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

class TradeEvaluator:
    def __init__(self, trade_df: pd.DataFrame):
        """
        Initializes with a trade log DataFrame containing:
        - entry_date
        - exit_date
        - entry_price
        - exit_price
        - profit_loss
        - trade_type (long or short)
        """
        self.trades = trade_df.copy()
        self.trades['duration'] = (self.trades['exit_date'] - self.trades['entry_date']).dt.days
        
        # Calculate percentage returns for each trade
        self.trades['return_pct'] = self.trades.apply(
            lambda x: (x['exit_price'] / x['entry_price'] - 1) * (1 if x['trade_type'] == 'long' else -1),
            axis=1
        )

    def count_trades(self):
        return len(self.trades)

    def win_rate(self):
        return (self.trades['profit_loss'] > 0).mean()

    def average_trade_duration(self):
        return self.trades['duration'].mean()

    def profit_factor(self):
        gross_profit = self.trades[self.trades['profit_loss'] > 0]['profit_loss'].sum()
        gross_loss = abs(self.trades[self.trades['profit_loss'] < 0]['profit_loss'].sum())
        return gross_profit / gross_loss if gross_loss > 0 else np.inf

    def best_trade(self):
        return self.trades['profit_loss'].max()

    def worst_trade(self):
        return self.trades['profit_loss'].min()
        
    def avg_win_loss(self):
        """Calculate average win and average loss"""
        wins = self.trades[self.trades['profit_loss'] > 0]
        losses = self.trades[self.trades['profit_loss'] < 0]
        
        avg_win = wins['profit_loss'].mean() if len(wins) > 0 else 0
        avg_loss = losses['profit_loss'].mean() if len(losses) > 0 else 0
        
        return avg_win, avg_loss
    
    def win_loss_ratio(self):
        """Calculate ratio of average win to average loss"""
        avg_win, avg_loss = self.avg_win_loss()
        
        if avg_loss == 0:
            return np.inf
            
        return abs(avg_win / avg_loss)
    
    def expectancy(self):
        """Calculate expectancy: (Win Rate × Average Win) - (Loss Rate × Average Loss)"""
        win_rate = self.win_rate()
        loss_rate = 1 - win_rate
        avg_win, avg_loss = self.avg_win_loss()
        
        return (win_rate * avg_win) - (loss_rate * abs(avg_loss))
    
    def consecutive_wins_losses(self):
        """Calculate maximum consecutive wins and losses"""
        is_win = self.trades['profit_loss'] > 0
        
        # Count consecutive wins
        win_streaks = []
        current_streak = 0
        
        for win in is_win:
            if win:
                current_streak += 1
            else:
                if current_streak > 0:
                    win_streaks.append(current_streak)
                current_streak = 0
                
        if current_streak > 0:
            win_streaks.append(current_streak)
            
        # Count consecutive losses
        loss_streaks = []
        current_streak = 0
        
        for win in is_win:
            if not win:
                current_streak += 1
            else:
                if current_streak > 0:
                    loss_streaks.append(current_streak)
                current_streak = 0
                
        if current_streak > 0:
            loss_streaks.append(current_streak)
            
        max_consec_wins = max(win_streaks) if win_streaks else 0
        max_consec_losses = max(loss_streaks) if loss_streaks else 0
        
        return max_consec_wins, max_consec_losses
    
    def time_analysis(self):
        """Analyze trade performance by day of week"""
        self.trades['day_of_week'] = self.trades['entry_date'].dt.day_name()
        
        day_performance = {}
        days = self.trades['day_of_week'].unique()
        
        for day in days:
            day_trades = self.trades[self.trades['day_of_week'] == day]
            day_performance[day] = {
                'count': len(day_trades),
                'win_rate': (day_trades['profit_loss'] > 0).mean(),
                'avg_profit': day_trades['profit_loss'].mean()
            }
            
        return day_performance
    
    def monthly_analysis(self):
        """Analyze trade performance by month"""
        self.trades['month'] = self.trades['entry_date'].dt.month_name()
        
        month_performance = {}
        months = self.trades['month'].unique()
        
        for month in months:
            month_trades = self.trades[self.trades['month'] == month]
            month_performance[month] = {
                'count': len(month_trades),
                'win_rate': (month_trades['profit_loss'] > 0).mean(),
                'avg_profit': month_trades['profit_loss'].mean()
            }
            
        return month_performance
    
    def holding_time_analysis(self):
        """Analyze performance based on holding time"""
        # Create bins for trade duration
        bins = [0, 1, 5, 10, 20, 30, float('inf')]
        labels = ['1 day', '2-5 days', '6-10 days', '11-20 days', '21-30 days', '>30 days']
        
        self.trades['duration_group'] = pd.cut(self.trades['duration'], bins=bins, labels=labels)
        
        duration_performance = {}
        groups = self.trades['duration_group'].unique()
        
        for group in groups:
            group_trades = self.trades[self.trades['duration_group'] == group]
            duration_performance[group] = {
                'count': len(group_trades),
                'win_rate': (group_trades['profit_loss'] > 0).mean(),
                'avg_profit': group_trades['profit_loss'].mean()
            }
            
        return duration_performance
    
    def mae_mfe_analysis(self):
        """
        This would require intraday data for Maximum Adverse Excursion
        and Maximum Favorable Excursion analysis.
        
        Placeholder for when such data is available.
        """
        return "Requires intraday price data for implementation"
    
    def risk_reward_ratio(self):
        """Calculate average risk/reward ratio if stop loss and take profit are recorded"""
        if 'stop_loss' in self.trades.columns and 'take_profit' in self.trades.columns:
            self.trades['risk'] = abs(self.trades['entry_price'] - self.trades['stop_loss'])
            self.trades['reward'] = abs(self.trades['take_profit'] - self.trades['entry_price'])
            
            return (self.trades['reward'] / self.trades['risk']).mean()
        else:
            return "Requires stop_loss and take_profit data"
    
    def calculate_r_multiples(self):
        """Calculate R-multiples if initial risk is recorded"""
        if 'initial_risk' in self.trades.columns:
            self.trades['r_multiple'] = self.trades['profit_loss'] / self.trades['initial_risk']
            return self.trades['r_multiple'].mean()
        else:
            return "Requires initial_risk data"
            
    def optimal_position_sizing(self, target_risk_pct=0.02):
        """
        Calculate optimal position sizing based on trade history
        
        Parameters:
        target_risk_pct (float): Target risk percentage per trade (e.g., 2%)
        
        Returns:
        Dictionary with optimal position sizing stats
        """
        if 'return_pct' not in self.trades.columns:
            return "Requires return_pct data"
            
        avg_loss_pct = abs(self.trades.loc[self.trades['return_pct'] < 0, 'return_pct'].mean())
        
        if np.isnan(avg_loss_pct) or avg_loss_pct == 0:
            return {"error": "No losing trades or zero average loss percentage"}
            
        kelly_percentage = self.win_rate() - ((1 - self.win_rate()) / (abs(self.trades[self.trades['return_pct'] > 0]['return_pct'].mean()) / avg_loss_pct))
        
        optimal_risk_per_trade = min(target_risk_pct, kelly_percentage / 2)  # Half Kelly is often used in practice
        
        return {
            "Kelly Percentage": kelly_percentage,
            "Half Kelly (Conservative)": kelly_percentage / 2,
            "Recommended Risk Per Trade": optimal_risk_per_trade,
            "Position Size Divisor": avg_loss_pct / optimal_risk_per_trade
        }

    def long_short_stats(self):
        long_trades = self.trades[self.trades['trade_type'] == 'long']
        short_trades = self.trades[self.trades['trade_type'] == 'short']
        return {
            'Long Win Rate': (long_trades['profit_loss'] > 0).mean() if len(long_trades) > 0 else np.nan,
            'Short Win Rate': (short_trades['profit_loss'] > 0).mean() if len(short_trades) > 0 else np.nan,
            'Long Count': len(long_trades),
            'Short Count': len(short_trades),
            'Long Avg Profit': long_trades['profit_loss'].mean() if len(long_trades) > 0 else np.nan,
            'Short Avg Profit': short_trades['profit_loss'].mean() if len(short_trades) > 0 else np.nan,
        }

    def generate_summary(self):
        avg_win, avg_loss = self.avg_win_loss()
        max_consecutive_wins, max_consecutive_losses = self.consecutive_wins_losses()
        
        summary = {
            'Trade Count': self.count_trades(),
            'Win Rate': self.win_rate(),
            'Profit Factor': self.profit_factor(),
            'Average Trade Duration': self.average_trade_duration(),
            'Best Trade': self.best_trade(),
            'Worst Trade': self.worst_trade(),
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Win/Loss Ratio': self.win_loss_ratio(),
            'Expectancy': self.expectancy(),
            'Max Consecutive Wins': max_consecutive_wins,
            'Max Consecutive Losses': max_consecutive_losses,
        }
        
        # Add long/short stats
        summary.update(self.long_short_stats())
        
        return summary
        
    def plot_profit_distribution(self, save_path=None):
        """Plot profit/loss distribution"""
        plt.figure(figsize=(12, 6))
        plt.hist(self.trades['profit_loss'], bins=50, alpha=0.6)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Profit/Loss Distribution')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Frequency')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_trade_durations(self, save_path=None):
        """Plot trade durations"""
        plt.figure(figsize=(12, 6))
        plt.hist(self.trades['duration'], bins=30, alpha=0.6)
        plt.title('Trade Duration Distribution')
        plt.xlabel('Duration (days)')
        plt.ylabel('Frequency')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_monthly_performance(self, save_path=None):
        """Plot monthly performance"""
        monthly = self.monthly_analysis()
        months = list(monthly.keys())
        win_rates = [monthly[m]['win_rate'] for m in months]
        avg_profits = [monthly[m]['avg_profit'] for m in months]
        
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Plot win rate
        ax1.bar(months, win_rates, alpha=0.6, color='blue')
        ax1.set_ylabel('Win Rate', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticklabels(months, rotation=45)
        
        # Plot average profit on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(months, avg_profits, 'r-')
        ax2.set_ylabel('Average Profit', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Monthly Performance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
