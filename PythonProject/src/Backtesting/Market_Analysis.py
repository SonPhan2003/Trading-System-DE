import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class MarketAnalyzer:
    def __init__(self, ohlcv_data):
        """
        Initialize with OHLCV market data
        
        Parameters:
        ohlcv_data (DataFrame): DataFrame with Open, High, Low, Close, Volume columns
        """
        self.data = ohlcv_data.copy()
        self.data.index = pd.to_datetime(self.data.index)
        
    def detect_market_regime(self, window=20, threshold=0.05):
        """
        Detect market regime (trending, ranging, volatile)
        
        Parameters:
        window (int): Rolling window for calculations
        threshold (float): Threshold for classifying regimes
        
        Returns:
        DataFrame: DataFrame with market regime classifications
        """
        # Calculate indicators
        self.data['returns'] = self.data['Close'].pct_change()
        self.data['volatility'] = self.data['returns'].rolling(window).std() * np.sqrt(252)
        self.data['adr'] = (self.data['High'] - self.data['Low']) / self.data['Close'].shift(1)
        
        # Calculate trending indicator (absolute rate of change)
        self.data['roc'] = abs(self.data['Close'].pct_change(window))
        
        # Classify regimes
        regimes = pd.DataFrame(index=self.data.index)
        regimes['volatility'] = 'normal'
        regimes['trend'] = 'ranging'
        
        # Volatility regimes
        vol_high = self.data['volatility'].rolling(window*5).mean() + threshold
        vol_low = self.data['volatility'].rolling(window*5).mean() - threshold
        
        regimes.loc[self.data['volatility'] > vol_high, 'volatility'] = 'high'
        regimes.loc[self.data['volatility'] < vol_low, 'volatility'] = 'low'
        
        # Trend regimes
        roc_high = self.data['roc'].rolling(window*5).mean() + threshold
        
        regimes.loc[self.data['roc'] > roc_high, 'trend'] = 'trending'
        
        # Combined regime
        regimes['regime'] = regimes['volatility'] + '_' + regimes['trend']
        
        return regimes
    
    def calculate_volatility_metrics(self):
        """
        Calculate various volatility metrics
        
        Returns:
        dict: Volatility metrics
        """
        # Calculate returns
        returns = self.data['Close'].pct_change().dropna()
        
        # Calculate metrics
        metrics = {
            'annualized_volatility': returns.std() * np.sqrt(252),
            'daily_avg_range': ((self.data['High'] - self.data['Low']) / self.data['Close'].shift(1)).mean(),
            'max_daily_range': ((self.data['High'] - self.data['Low']) / self.data['Close'].shift(1)).max(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'jarque_bera': stats.jarque_bera(returns.dropna().values)[0],
            'is_normal_dist': stats.jarque_bera(returns.dropna().values)[1] > 0.05
        }
        
        return metrics
    
    def volume_analysis(self, window=20):
        """
        Analyze volume patterns
        
        Parameters:
        window (int): Window for rolling calculations
        
        Returns:
        DataFrame: Volume analysis metrics
        """
        volume_data = pd.DataFrame(index=self.data.index)
        
        # Normalize volume
        volume_data['volume'] = self.data['Volume']
        volume_data['volume_sma'] = self.data['Volume'].rolling(window).mean()
        volume_data['volume_ratio'] = self.data['Volume'] / volume_data['volume_sma']
        
        # Classify volume
        volume_data['volume_class'] = 'normal'
        volume_data.loc[volume_data['volume_ratio'] > 1.5, 'volume_class'] = 'high'
        volume_data.loc[volume_data['volume_ratio'] < 0.5, 'volume_class'] = 'low'
        
        # Price-volume relationship
        self.data['price_change'] = self.data['Close'] - self.data['Open']
        volume_data['price_direction'] = np.sign(self.data['price_change'])
        
        # Rising price with rising volume is bullish
        volume_data['pv_bull'] = ((volume_data['price_direction'] > 0) & 
                                 (volume_data['volume_ratio'] > 1.2))
        
        # Falling price with rising volume is bearish
        volume_data['pv_bear'] = ((volume_data['price_direction'] < 0) & 
                                 (volume_data['volume_ratio'] > 1.2))
        
        return volume_data
    
    def support_resistance_levels(self, window=20, bins=50, tolerance=0.02):
        """
        Find potential support and resistance levels
        
        Parameters:
        window (int): Window for peak detection
        bins (int): Number of bins for histogram
        tolerance (float): Price tolerance around levels
        
        Returns:
        dict: Support and resistance levels
        """
        price_data = self.data[['High', 'Low', 'Close']].copy()
        
        # Find peaks and troughs
        price_data['high_roll_max'] = self.data['High'].rolling(window, center=True).max()
        price_data['low_roll_min'] = self.data['Low'].rolling(window, center=True).min()
        
        resistance_points = price_data[price_data['High'] >= price_data['high_roll_max']]['High']
        support_points = price_data[price_data['Low'] <= price_data['low_roll_min']]['Low']
        
        # Use histogram to find clusters
        all_prices = np.concatenate([self.data['High'].values, self.data['Low'].values, self.data['Close'].values])
        hist, bin_edges = np.histogram(all_prices, bins=bins)
        
        # Find price levels with high frequency
        threshold = np.percentile(hist, 80)
        level_indices = np.where(hist >= threshold)[0]
        
        levels = []
        for idx in level_indices:
            price_level = (bin_edges[idx] + bin_edges[idx + 1]) / 2
            
            # Check if this level was tested as support or resistance
            support_touch = any(abs(support_points - price_level) / price_level < tolerance)
            resistance_touch = any(abs(resistance_points - price_level) / price_level < tolerance)
            
            levels.append({
                'price': price_level,
                'strength': hist[idx],
                'support_touch': support_touch,
                'resistance_touch': resistance_touch
            })
        
        # Separate into support and resistance
        current_price = self.data['Close'].iloc[-1]
        
        support_levels = [level for level in levels if level['price'] < current_price]
        resistance_levels = [level for level in levels if level['price'] > current_price]
        
        # Sort by distance from current price
        support_levels.sort(key=lambda x: current_price - x['price'])
        resistance_levels.sort(key=lambda x: x['price'] - current_price)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def trend_strength(self, window=20):
        """
        Calculate trend strength indicators
        
        Parameters:
        window (int): Window for calculations
        
        Returns:
        DataFrame: Trend strength indicators
        """
        trend_data = pd.DataFrame(index=self.data.index)
        
        # ADX (Average Directional Index)
        # Simplified calculation
        high_change = self.data['High'].diff()
        low_change = -self.data['Low'].diff()
        
        # +DM and -DM
        plus_dm = (high_change > low_change) & (high_change > 0)
        minus_dm = (low_change > high_change) & (low_change > 0)
        
        # TR (True Range)
        tr = pd.DataFrame({
            'hl': self.data['High'] - self.data['Low'],
            'hc': abs(self.data['High'] - self.data['Close'].shift(1)),
            'lc': abs(self.data['Low'] - self.data['Close'].shift(1))
        }).max(axis=1)
        
        # Simplified ADX
        tr_ma = tr.rolling(window).mean()
        plus_di = (plus_dm * high_change).rolling(window).mean() / tr_ma * 100
        minus_di = (minus_dm * low_change).rolling(window).mean() / tr_ma * 100
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window).mean()
        
        trend_data['ADX'] = adx
        
        # Classify trend strength
        trend_data['trend_strength'] = 'no_trend'
        trend_data.loc[trend_data['ADX'] > 20, 'trend_strength'] = 'weak'
        trend_data.loc[trend_data['ADX'] > 25, 'trend_strength'] = 'moderate'
        trend_data.loc[trend_data['ADX'] > 30, 'trend_strength'] = 'strong'
        
        # Trend direction
        trend_data['trend_direction'] = 'sideways'
        trend_data.loc[plus_di > minus_di, 'trend_direction'] = 'up'
        trend_data.loc[minus_di > plus_di, 'trend_direction'] = 'down'
        
        return trend_data
    
    def seasonality_analysis(self, period_type='month'):
        """
        Analyze seasonality of returns
        
        Parameters:
        period_type (str): Type of period to analyze ('day', 'month', 'quarter')
        
        Returns:
        DataFrame: Seasonal analysis
        """
        # Calculate daily returns
        self.data['returns'] = self.data['Close'].pct_change()
        
        if period_type == 'day':
            # Day of week analysis
            self.data['day'] = self.data.index.day_name()
            seasonality = self.data.groupby('day')['returns'].agg(['mean', 'std'])
            
            # Calculate t-stat to check significance
            def t_stat(group):
                return stats.ttest_1samp(group, 0.0).statistic
                
            seasonality['t_stat'] = self.data.groupby('day')['returns'].apply(t_stat)
            seasonality['significant'] = abs(seasonality['t_stat']) > 1.96  # 95% confidence
            
        elif period_type == 'month':
            # Month analysis
            self.data['month'] = self.data.index.month_name()
            seasonality = self.data.groupby('month')['returns'].agg(['mean', 'std'])
            
            # Add significance test
            def month_t_stat(group):
                if len(group) < 2:
                    return 0
                return stats.ttest_1samp(group, 0.0).statistic
                
            seasonality['t_stat'] = self.data.groupby('month')['returns'].apply(month_t_stat)
            seasonality['significant'] = abs(seasonality['t_stat']) > 1.96
            
            # Reorder months
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            seasonality = seasonality.reindex(month_order)
            
        elif period_type == 'quarter':
            # Quarter analysis
            self.data['quarter'] = self.data.index.quarter
            seasonality = self.data.groupby('quarter')['returns'].agg(['mean', 'std'])
            
            # Add significance test
            def quarter_t_stat(group):
                if len(group) < 2:
                    return 0
                return stats.ttest_1samp(group, 0.0).statistic
                
            seasonality['t_stat'] = self.data.groupby('quarter')['returns'].apply(quarter_t_stat)
            seasonality['significant'] = abs(seasonality['t_stat']) > 1.96
        
        return seasonality
    
    def plot_market_regimes(self, regimes, save_path=None):
        """
        Plot price chart with market regimes
        
        Parameters:
        regimes (DataFrame): Market regime data from detect_market_regime
        save_path (str): Optional path to save the figure
        """
        plt.figure(figsize=(14, 8))
        
        # Create a colormap for regimes
        regime_colors = {
            'high_trending': 'red',
            'high_ranging': 'orange',
            'normal_trending': 'blue',
            'normal_ranging': 'gray',
            'low_trending': 'green',
            'low_ranging': 'lightgreen'
        }
        
        # Plot price
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(self.data.index, self.data['Close'], 'k-')
        ax1.set_title('Price and Market Regimes')
        ax1.set_ylabel('Price')
        
        # Highlight regimes
        for regime in regime_colors:
            regime_periods = regimes[regimes['regime'] == regime]
            if not regime_periods.empty:
                for start_idx in range(len(regime_periods)):
                    if start_idx == len(regime_periods) - 1:
                        continue
                        
                    start_date = regime_periods.index[start_idx]
                    if start_idx + 1 < len(regime_periods.index):
                        end_date = regime_periods.index[start_idx + 1]
                    else:
                        end_date = regime_periods.index[-1]
                        
                    ax1.axvspan(start_date, end_date, color=regime_colors[regime], alpha=0.3)
        
        # Plot volatility
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(self.data.index, self.data['volatility'], 'b-')
        ax2.set_title('Volatility')
        ax2.set_ylabel('Annualized Volatility')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_support_resistance(self, levels, window=50, save_path=None):
        """
        Plot price chart with support and resistance levels
        
        Parameters:
        levels (dict): Support and resistance levels from support_resistance_levels
        window (int): Number of periods to display
        save_path (str): Optional path to save the figure
        """
        plt.figure(figsize=(14, 7))
        
        # Get recent data
        recent_data = self.data.iloc[-window:].copy()
        
        # Plot candlestick-like chart
        plt.plot(recent_data.index, recent_data['Close'], 'k-')
        
        # Plot all support levels
        for level in levels['support']:
            plt.axhline(y=level['price'], color='g', linestyle='--', alpha=0.5 + 0.5 * level['strength'] / max([s['strength'] for s in levels['support']]))
            plt.text(recent_data.index[-1], level['price'], f"S: {level['price']:.2f}", va='center', ha='left')
            
        # Plot all resistance levels
        for level in levels['resistance']:
            plt.axhline(y=level['price'], color='r', linestyle='--', alpha=0.5 + 0.5 * level['strength'] / max([r['strength'] for r in levels['resistance']]))
            plt.text(recent_data.index[-1], level['price'], f"R: {level['price']:.2f}", va='center', ha='left')
            
        plt.title('Price with Support and Resistance Levels')
        plt.ylabel('Price')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_seasonality(self, seasonality, save_path=None):
        """
        Plot seasonality analysis
        
        Parameters:
        seasonality (DataFrame): Seasonality data from seasonality_analysis
        save_path (str): Optional path to save the figure
        """
        plt.figure(figsize=(14, 7))
        
        # Bar colors based on positive/negative returns
        colors = ['green' if x > 0 else 'red' for x in seasonality['mean']]
        
        # Bold bars for statistically significant results
        alphas = [1.0 if x else 0.5 for x in seasonality['significant']]
        
        # Create bars with error bars
        bars = plt.bar(seasonality.index, seasonality['mean'], yerr=seasonality['std'], 
                     alpha=alphas, color=colors, capsize=5)
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.title('Seasonality Analysis')
        plt.ylabel('Average Return')
        plt.xticks(rotation=45)
        
        # Add annotations for significant results
        for i, significant in enumerate(seasonality['significant']):
            if significant:
                plt.annotate('*', 
                           (i, seasonality['mean'].iloc[i]),
                           xytext=(0, 10 if seasonality['mean'].iloc[i] > 0 else -15),
                           textcoords='offset points',
                           ha='center', va='center',
                           fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 