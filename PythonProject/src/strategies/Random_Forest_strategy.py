import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators for feature engineering"""
    
    # Price-based features
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    df['EMA_12'] = df['close'].ewm(span=12).mean()
    df['EMA_26'] = df['close'].ewm(span=26).mean()
    
    # Price position features
    df['price_vs_sma20'] = df['close'] / df['SMA_20']
    df['price_vs_sma50'] = df['close'] / df['SMA_50']
    df['price_vs_high_20'] = df['close'] / df['high'].rolling(window=20).max()
    df['price_vs_low_20'] = df['close'] / df['low'].rolling(window=20).min()
    
    # Momentum features
    df['RSI'] = calculate_rsi(df['close'], 14)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    df['ROC_5'] = df['close'].pct_change(5)
    df['ROC_10'] = df['close'].pct_change(10)
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    
    # Volatility features
    df['ATR'] = calculate_atr(df, 14)
    df['volatility_10'] = df['close'].rolling(window=10).std()
    df['volatility_20'] = df['close'].rolling(window=20).std()
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['price_volume_trend'] = df['close'].pct_change() * df['volume']
    
    # Trend features
    df['trend_5'] = df['close'].rolling(window=5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    df['trend_10'] = df['close'].rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    df['trend_20'] = df['close'].rolling(window=20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    
    # Gap features
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_pct'] = df['gap'] / df['close'].shift(1)
    
    return df

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, window=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def create_target_variable(df, lookforward_days=1, threshold=0.005):
    """Create target variable for classification (1 = up, 0 = down)"""
    future_return = df['close'].shift(-lookforward_days) / df['close'] - 1
    target = (future_return > threshold).astype(int)
    return target

def prepare_features(df):
    """Prepare feature matrix for machine learning"""
    
    # Select feature columns
    feature_columns = [
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26',
        'price_vs_sma20', 'price_vs_sma50', 'price_vs_high_20', 'price_vs_low_20',
        'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
        'ROC_5', 'ROC_10', 'momentum_5', 'momentum_10',
        'ATR', 'volatility_10', 'volatility_20', 'BB_position',
        'volume_ratio', 'price_volume_trend',
        'trend_5', 'trend_10', 'trend_20',
        'gap_pct'
    ]
    
    # Create feature matrix
    features = df[feature_columns].copy()
    
    # Handle missing values
    features = features.fillna(method='ffill').fillna(0)
    
    return features, feature_columns

def random_forest_strategy(df, 
                         lookback_period=50,
                         confidence_threshold=0.5,
                         stop_loss_pct=0.025,
                         max_holding_days=10,
                         n_estimators=50,
                         max_depth=6,
                         min_samples_split=5,
                         min_samples_leaf=2,
                         retrain_frequency=50):
    """
    Random Forest Trading Strategy
    
    Parameters:
    - lookback_period: Days of history for training (default: 100)
    - confidence_threshold: Minimum confidence to trade (default: 0.6)
    - stop_loss_pct: Stop loss percentage (default: 0.03)
    - max_holding_days: Maximum position duration (default: 10)
    - n_estimators: Number of trees (default: 100)
    - max_depth: Maximum tree depth (default: 10)
    - min_samples_split: Minimum samples to split (default: 5)
    - min_samples_leaf: Minimum samples per leaf (default: 2)
    - retrain_frequency: How often to retrain model (default: 20)
    """
    
    print("Starting Random Forest Strategy...")
    print(f"Parameters: lookback={lookback_period}, confidence={confidence_threshold}, stop_loss={stop_loss_pct}")
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Create target variable
    df['target'] = create_target_variable(df)
    
    # Debug: Check target distribution (commented out for production)
    # target_dist = df['target'].value_counts()
    # print(f"Target distribution: {target_dist}")
    # print(f"Target percentage: {df['target'].mean():.3f}")
    
    # Prepare features
    features, feature_columns = prepare_features(df)
    
    # Initialize results
    trades = []
    positions = []
    current_position = None
    model = None
    scaler = StandardScaler()
    
    # Training data
    train_data = []
    train_targets = []
    
    # Pre-populate training data with initial lookback period
    print(f"Pre-populating training data with {lookback_period} samples...")
    for i in range(lookback_period):
        train_data.append(features.iloc[i].values)
        train_targets.append(df['target'].iloc[i])
    
    print(f"Processing {len(df)} days of data...")
    
    for i in range(lookback_period, len(df)):
        current_date = df.index[i]
        current_price = df['close'].iloc[i]
        
        # Retrain model periodically
        if i % retrain_frequency == 0 or model is None:
            if len(train_data) > 20:  # Lower threshold for initial training
                # print(f"Retraining model at day {i} with {len(train_data)} samples...")
                
                # Prepare training data
                X_train = np.array(train_data)
                y_train = np.array(train_targets)
                
                # Scale features
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Train Random Forest
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train_scaled, y_train)
                
                # print(f"Model trained with {len(train_data)} samples")
                # print(f"Training accuracy: {model.score(X_train_scaled, y_train):.3f}")
        
        # Make prediction if model is available
        if model is not None:
            # Get current features
            current_features = features.iloc[i].values.reshape(1, -1)
            current_features_scaled = scaler.transform(current_features)
            
            # Make prediction
            prediction = model.predict(current_features_scaled)[0]
            confidence = model.predict_proba(current_features_scaled)[0].max()
            
            # Debug: Print some predictions (commented out for production)
            # if i % 50 == 0:  # Print every 50th prediction
            #     print(f"Day {i}: Prediction={prediction}, Confidence={confidence:.3f}, Target={df['target'].iloc[i]}")
            
            # Debug: Print when we get high confidence predictions (commented out for production)
            # if confidence > confidence_threshold:
            #     print(f"High confidence at day {i}: Prediction={prediction}, Confidence={confidence:.3f}")
            
            # Add to training data for next retraining (only if we have space)
            if len(train_data) < lookback_period:
                train_data.append(features.iloc[i].values)
                train_targets.append(df['target'].iloc[i])
            else:
                # Replace oldest data with new data
                train_data = train_data[1:] + [features.iloc[i].values]
                train_targets = train_targets[1:] + [df['target'].iloc[i]]
        else:
            prediction = 0
            confidence = 0.5
        
        # Handle existing position
        if current_position is not None:
            current_position['holding_days'] += 1
            
            # Check exit conditions
            exit_reason = None
            exit_price = current_price
            
            # Stop loss
            if current_position['type'] == 'LONG':
                if current_price <= current_position['entry_price'] * (1 - stop_loss_pct):
                    exit_reason = 'Stop Loss'
                elif current_position['holding_days'] >= max_holding_days:
                    exit_reason = 'Max Holding Days'
                elif prediction == 0 and confidence > confidence_threshold:
                    exit_reason = 'Signal Change'
            else:  # SHORT
                if current_price >= current_position['entry_price'] * (1 + stop_loss_pct):
                    exit_reason = 'Stop Loss'
                elif current_position['holding_days'] >= max_holding_days:
                    exit_reason = 'Max Holding Days'
                elif prediction == 1 and confidence > confidence_threshold:
                    exit_reason = 'Signal Change'
            
            # Exit position if conditions met
            if exit_reason:
                profit_loss = 0
                if current_position['type'] == 'LONG':
                    profit_loss = exit_price - current_position['entry_price']
                else:
                    profit_loss = current_position['entry_price'] - exit_price
                
                trades.append({
                    'entry_date': current_position['entry_date'],
                    'exit_date': current_date,
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'trade_type': current_position['type'],
                    'holding_days': current_position['holding_days'],
                    'exit_reason': exit_reason,
                    'confidence': current_position['confidence']
                })
                
                current_position = None
        
        # Enter new position if conditions met
        if current_position is None and confidence > confidence_threshold:
            # Relaxed momentum filter - only filter extreme cases
            recent_trend = df['close'].iloc[max(0, i-5):i].mean()
            price_momentum = current_price / recent_trend
            
            # More lenient conditions - only filter if in very strong opposite trend
            if prediction == 1 and price_momentum > 0.98:  # Predict up, allow unless in strong downtrend
                current_position = {
                    'type': 'LONG',
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'holding_days': 0,
                    'confidence': confidence
                }
            elif prediction == 0 and price_momentum < 1.02:  # Predict down, allow unless in strong uptrend
                current_position = {
                    'type': 'SHORT',
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'holding_days': 0,
                    'confidence': confidence
                }
        
        # Store position for tracking
        positions.append({
            'date': current_date,
            'position': current_position['type'] if current_position else 'NONE',
            'confidence': confidence,
            'prediction': prediction
        })
    
    # Close any remaining position
    if current_position is not None:
        final_price = df['close'].iloc[-1]
        profit_loss = 0
        if current_position['type'] == 'LONG':
            profit_loss = final_price - current_position['entry_price']
        else:
            profit_loss = current_position['entry_price'] - final_price
        
        trades.append({
            'entry_date': current_position['entry_date'],
            'exit_date': df.index[-1],
            'entry_price': current_position['entry_price'],
            'exit_price': final_price,
            'profit_loss': profit_loss,
            'trade_type': current_position['type'],
            'holding_days': current_position['holding_days'],
            'exit_reason': 'End of Data',
            'confidence': current_position['confidence']
        })
    
    # Create results DataFrame
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df['entry_date'] = pd.to_datetime(trade_df['entry_date'])
        trade_df['exit_date'] = pd.to_datetime(trade_df['exit_date'])
    else:
        trade_df = pd.DataFrame(columns=[
            'entry_date', 'exit_date', 'entry_price', 'exit_price', 
            'profit_loss', 'trade_type', 'holding_days', 'exit_reason', 'confidence'
        ])
    
    # Feature importance analysis
    feature_importance = None
    if model is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    print(f"Random Forest Strategy completed!")
    print(f"Total trades: {len(trades)}")
    print(f"Long trades: {len([t for t in trades if t['trade_type'] == 'LONG'])}")
    print(f"Short trades: {len([t for t in trades if t['trade_type'] == 'SHORT'])}")
    
    if feature_importance is not None:
        print("\nTop 5 Most Important Features:")
        print(feature_importance.head())
    
    return trade_df, feature_importance
