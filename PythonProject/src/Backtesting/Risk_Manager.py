import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RiskManager:
    def __init__(self, account_size=100000):
        """
        Initialize the risk manager with starting account size
        
        Parameters:
        account_size (float): Starting account/portfolio size
        """
        self.account_size = account_size
        
    def calculate_position_size(self, entry_price, stop_loss, risk_per_trade=0.01):
        """
        Calculate the optimal position size based on risk per trade
        
        Parameters:
        entry_price (float): Planned entry price
        stop_loss (float): Planned stop loss price
        risk_per_trade (float): Percentage of account to risk per trade (e.g. 0.01 = 1%)
        
        Returns:
        dict: Position sizing information
        """
        if entry_price <= 0 or stop_loss <= 0:
            return {"error": "Prices must be positive"}
        
        risk_amount = self.account_size * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return {"error": "Stop loss cannot be equal to entry price"}
            
        num_shares = int(risk_amount / risk_per_share)
        total_cost = num_shares * entry_price
        
        return {
            "risk_amount": risk_amount,
            "risk_per_share": risk_per_share,
            "shares": num_shares,
            "total_cost": total_cost,
            "account_percentage": total_cost / self.account_size
        }
    
    def position_sizer_fixed_risk(self, price, atr, risk_multiple=2, max_risk_pct=0.02):
        """
        Calculate position size using ATR for stop loss placement
        
        Parameters:
        price (float): Current price
        atr (float): Average True Range value
        risk_multiple (float): Multiple of ATR to use for stop loss distance (e.g. 2 = 2 * ATR)
        max_risk_pct (float): Maximum account percentage to risk per trade
        
        Returns:
        dict: Position sizing information
        """
        if price <= 0 or atr <= 0:
            return {"error": "Price and ATR must be positive"}
            
        stop_distance = atr * risk_multiple
        risk_amount = self.account_size * max_risk_pct
        
        num_shares = int(risk_amount / stop_distance)
        total_cost = num_shares * price
        
        return {
            "price": price,
            "stop_loss": price - stop_distance,
            "stop_distance": stop_distance,
            "risk_amount": risk_amount,
            "shares": num_shares,
            "total_cost": total_cost,
            "account_percentage": total_cost / self.account_size
        }
    
    def kelly_criterion(self, win_rate, win_loss_ratio):
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Parameters:
        win_rate (float): Win rate as decimal (e.g., 0.6 for 60%)
        win_loss_ratio (float): Ratio of average win to average loss
        
        Returns:
        float: Kelly percentage
        """
        if win_loss_ratio <= 0:
            return 0
            
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Kelly can be negative if expected value is negative
        return max(0, kelly_pct)
    
    def optimal_f(self, returns):
        """
        Calculate the optimal fixed fraction (optimal f) based on historical returns
        
        Parameters:
        returns (array-like): Array of trade returns
        
        Returns:
        float: Optimal f value
        """
        if len(returns) == 0:
            return 0
            
        # Convert returns to HPR (Holding Period Return) where loss can't exceed 100%
        hpr = returns.copy()
        
        # Find the worst loss
        worst_loss = min(hpr)
        
        if worst_loss >= 0:
            return 1.0  # No losses, use full Kelly
            
        # Scale returns by the worst loss
        f_values = np.linspace(0, 2, 201)  # Test f values from 0 to 2 in steps of 0.01
        twr_values = []
        
        for f in f_values:
            # TWR = product of (1 + f * return / abs(worst_loss))
            twr = np.prod(1 + f * hpr / abs(worst_loss))
            twr_values.append(twr)
            
        optimal_f = f_values[np.argmax(twr_values)]
        
        return min(optimal_f, 1.0)  # Cap at 1.0 for safety
    
    def portfolio_heat(self, open_positions):
        """
        Calculate portfolio heat - how much of the portfolio is at risk
        
        Parameters:
        open_positions (list): List of dictionaries with 'size', 'entry', 'stop' keys
        
        Returns:
        dict: Portfolio heat information
        """
        total_risk = 0
        
        for position in open_positions:
            position_risk = position['size'] * abs(position['entry'] - position['stop'])
            total_risk += position_risk
            
        heat_percentage = total_risk / self.account_size
        
        return {
            "total_risk_amount": total_risk,
            "heat_percentage": heat_percentage
        }
    
    def correlation_based_sizing(self, returns_dict, target_position):
        """
        Adjust position size based on correlation with existing positions
        
        Parameters:
        returns_dict (dict): Dictionary mapping symbol to returns series
        target_position (str): Symbol of the position to size
        
        Returns:
        float: Adjustment factor for position size
        """
        symbols = list(returns_dict.keys())
        
        if target_position not in symbols or len(symbols) <= 1:
            return 1.0  # No adjustment needed
            
        correlations = {}
        target_returns = returns_dict[target_position]
        
        for symbol in symbols:
            if symbol != target_position:
                corr = np.corrcoef(target_returns, returns_dict[symbol])[0, 1]
                correlations[symbol] = corr
                
        # Average positive correlation reduces position size
        avg_correlation = np.mean(list(correlations.values()))
        
        # Adjustment factor: 1 for no correlation, less for positive correlation
        adjustment = 1.0 - max(0, avg_correlation)
        
        return max(0.2, adjustment)  # Don't reduce by more than 80%
    
    def risk_of_ruin(self, win_rate, risk_per_trade, total_trades=1000):
        """
        Calculate risk of ruin - probability of losing all capital
        
        Parameters:
        win_rate (float): Win rate as decimal (e.g., 0.6 for 60%)
        risk_per_trade (float): Percentage of account risked per trade
        total_trades (int): Number of trades to simulate
        
        Returns:
        float: Probability of ruin
        """
        if win_rate <= 0 or win_rate >= 1:
            return "Win rate must be between 0 and 1"
            
        if risk_per_trade <= 0 or risk_per_trade >= 1:
            return "Risk per trade must be between 0 and 1"
        
        # Simple approximation for risk of ruin
        if win_rate > 0.5:  # Positive expectancy
            ratio = (1 - win_rate) / win_rate
            return ratio ** (1 / risk_per_trade)
        else:
            return 1.0  # Guaranteed ruin with negative expectancy
    
    def monte_carlo_simulation(self, returns, initial_capital=None, num_simulations=1000, num_trades=252):
        """
        Run Monte Carlo simulation to estimate portfolio outcomes
        
        Parameters:
        returns (array-like): Array of historical trade returns
        initial_capital (float): Starting capital for simulation (defaults to self.account_size)
        num_simulations (int): Number of simulations to run
        num_trades (int): Number of trades to simulate
        
        Returns:
        dict: Simulation results
        """
        if initial_capital is None:
            initial_capital = self.account_size
            
        if len(returns) == 0:
            return {"error": "No returns data provided"}
            
        # Run simulations
        simulation_results = np.zeros((num_simulations, num_trades + 1))
        simulation_results[:, 0] = initial_capital
        
        for i in range(num_simulations):
            sampled_returns = np.random.choice(returns, size=num_trades, replace=True)
            for j in range(num_trades):
                simulation_results[i, j+1] = simulation_results[i, j] * (1 + sampled_returns[j])
                
        # Calculate statistics
        final_values = simulation_results[:, -1]
        
        percentiles = {
            "5th": np.percentile(final_values, 5),
            "25th": np.percentile(final_values, 25),
            "50th": np.percentile(final_values, 50),
            "75th": np.percentile(final_values, 75),
            "95th": np.percentile(final_values, 95)
        }
        
        max_drawdowns = np.zeros(num_simulations)
        for i in range(num_simulations):
            peaks = np.maximum.accumulate(simulation_results[i, :])
            drawdowns = (simulation_results[i, :] - peaks) / peaks
            max_drawdowns[i] = drawdowns.min()
            
        return {
            "initial_capital": initial_capital,
            "final_values": {
                "mean": final_values.mean(),
                "min": final_values.min(),
                "max": final_values.max(),
                "percentiles": percentiles
            },
            "drawdowns": {
                "mean": max_drawdowns.mean(),
                "median": np.median(max_drawdowns),
                "worst": max_drawdowns.min(),
                "5th_percentile": np.percentile(max_drawdowns, 5)
            },
            "probability_of_profit": (final_values > initial_capital).mean(),
            "simulation_data": simulation_results
        }
        
    def plot_monte_carlo(self, simulation_data, save_path=None):
        """
        Plot Monte Carlo simulation results
        
        Parameters:
        simulation_data (ndarray): Simulation data from monte_carlo_simulation
        save_path (str): Optional path to save the figure
        """
        plt.figure(figsize=(14, 7))
        
        # Plot a subset of simulations for clarity
        subset_size = min(100, simulation_data.shape[0])
        indices = np.random.choice(simulation_data.shape[0], subset_size, replace=False)
        
        for i in indices:
            plt.plot(simulation_data[i, :], alpha=0.1, color='blue')
            
        # Plot statistics
        mean_equity = simulation_data.mean(axis=0)
        median_equity = np.median(simulation_data, axis=0)
        lower_bound = np.percentile(simulation_data, 5, axis=0)
        upper_bound = np.percentile(simulation_data, 95, axis=0)
        
        plt.plot(mean_equity, color='red', linewidth=2, label='Mean')
        plt.plot(median_equity, color='black', linewidth=2, label='Median')
        plt.plot(lower_bound, color='green', linewidth=2, linestyle='--', label='5th Percentile')
        plt.plot(upper_bound, color='green', linewidth=2, linestyle='--', label='95th Percentile')
        
        plt.title('Monte Carlo Simulation of Trading Strategy')
        plt.xlabel('Number of Trades')
        plt.ylabel('Account Value')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_risk_of_ruin(self, win_rates=None, risk_per_trade=None, save_path=None):
        """
        Plot risk of ruin for different win rates and risk per trade values
        
        Parameters:
        win_rates (array-like): Array of win rates to plot
        risk_per_trade (array-like): Array of risk per trade values to plot
        save_path (str): Optional path to save the figure
        """
        if win_rates is None:
            win_rates = np.linspace(0.4, 0.7, 7)
            
        if risk_per_trade is None:
            risk_per_trade = np.array([0.01, 0.02, 0.05, 0.1])
            
        plt.figure(figsize=(12, 8))
        
        for risk in risk_per_trade:
            ror_values = []
            for wr in win_rates:
                if wr <= 0.5:
                    ror_values.append(1.0)
                else:
                    ratio = (1 - wr) / wr
                    ror_values.append(ratio ** (1 / risk))
                    
            plt.semilogy(win_rates, ror_values, marker='o', label=f'Risk {risk*100:.0f}%')
            
        plt.title('Risk of Ruin by Win Rate and Risk Per Trade')
        plt.xlabel('Win Rate')
        plt.ylabel('Risk of Ruin (log scale)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 