import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from Data.Loader import load_stock_data

def simulate_lstm_trading(predictions, dates, close_prices, initial_capital=10000, 
                         stop_loss_pct=0.03, max_holding_days=15, take_profit_pct=0.05):
    """
    Simulate trading based on LSTM predictions with risk management.
    
    Parameters:
    - predictions: List of predictions (0=Sell, 1=Buy)
    - dates: List of dates corresponding to predictions
    - close_prices: List of close prices corresponding to predictions
    - initial_capital: Starting capital for simulation
    - stop_loss_pct: Stop loss percentage (default: 0.03 = 3%)
    - max_holding_days: Maximum holding period (default: 15 days)
    - take_profit_pct: Take profit percentage (default: 0.05 = 5%)
    
    Returns:
    - DataFrame with trade information matching other strategies
    """
    trades = []
    position = None
    entry_price = None
    entry_date = None
    capital = initial_capital
    
    for i in range(len(predictions)):
        current_date = dates[i]
        current_price = close_prices[i]
        current_prediction = predictions[i]
        
        # Trading logic
        if position is None:  # No position
            if current_prediction == 1:  # Buy signal
                position = "long"
                entry_price = current_price
                entry_date = current_date
        else:  # Have position
            holding_days = (current_date - entry_date).days
            exit_reason = None
            exit_price = current_price
            
            # Check exit conditions
            if current_prediction == 0:  # Sell signal
                exit_reason = "signal"
            elif holding_days >= max_holding_days:
                exit_reason = "max_holding"
            elif current_price <= entry_price * (1 - stop_loss_pct):  # Stop loss
                exit_reason = "stop_loss"
            elif current_price >= entry_price * (1 + take_profit_pct):  # Take profit
                exit_reason = "take_profit"
            
            if exit_reason:
                profit_loss = exit_price - entry_price
                
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit_loss": profit_loss,
                    "trade_type": "long",
                    "holding_days": holding_days,
                    "exit_reason": exit_reason
                })
                
                # Reset position
                position = None
                entry_price = None
                entry_date = None
    
    # Handle case where we still have a position at the end
    if position is not None:
        exit_price = close_prices[-1]
        profit_loss = exit_price - entry_price
        
        trades.append({
            "entry_date": entry_date,
            "exit_date": dates[-1],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_loss": profit_loss,
            "trade_type": "long",
            "holding_days": (dates[-1] - entry_date).days,
            "exit_reason": "end_of_data"
        })
    
    return pd.DataFrame(trades)

def lstm_strategy(symbol="AMZN", lookback=30, batch_size=64, epochs=15, lr=1e-3, threshold=0.001):
    """
    LSTM strategy for stock price prediction and trading.
    
    Parameters:
    - symbol: Stock symbol
    - lookback: Number of days to look back for prediction
    - batch_size: Batch size for training
    - epochs: Number of training epochs
    - lr: Learning rate
    - threshold: Threshold for up/down classification
    
    Returns:
    - DataFrame with trade information matching other strategies
    """
    # 1. Load data
    df = load_stock_data(symbol)
    df = df.sort_index()

    # 1b. Add technical indicators (using only pandas/numpy)
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    # Exponential Moving Average
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    # Bollinger Bands (20)
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    # 1c. Drop rows with NaN values from indicator calculation
    df = df.dropna()
    # Remove MACD outliers (extreme values)
    df = df[(df['macd'].abs() < 100) & (df['macd_hist'].abs() < 100)]

    # Save original close price before normalization
    df["close_orig"] = df["close"]

    # 2. Create 2-class label: 1 (up > threshold), 0 (down < -threshold)
    returns = (df["close"].shift(-1) - df["close"]) / df["close"]
    labels = np.where(returns > threshold, 1, 0)
    df["target"] = labels
    df = df.iloc[:-1]  # Remove last row (no next day)

    # 3. Normalize OHLCV features
    scaler = StandardScaler()
    features = ["open", "high", "low", "close", "volume", "sma_5", "sma_10", "ema_10", "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14", "bb_upper", "bb_middle", "bb_lower"]
    df[features] = scaler.fit_transform(df[features])

    # 4. Create sliding windows
    def create_windows(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[features].iloc[i:i+lookback].values)
            y.append(data["target"].iloc[i+lookback])
        return np.array(X), np.array(y)

    X, y = create_windows(df, lookback)

    # 5. Split data (70% train, 30% test)
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 6. Build the LSTM 2-class classifier
    class LSTMClassifier2(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_classes=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out  # logits

    model = LSTMClassifier2(input_size=len(features))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 7. Train the model
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += len(yb)
        train_acc = train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {np.mean(train_losses):.4f} - Train Acc: {train_acc:.4f}")

    # 8. Evaluate on test set
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(yb.numpy())
            y_pred.extend(preds.numpy())
            y_prob.extend(probs.numpy())

    # 8b. Simulate trading and return trade DataFrame
    # Get the test indices (dates) from the original df
    test_indices = df.index[-len(y_true):]
    test_close = df.loc[test_indices, "close_orig"].values  # Use original close price

    # Simulate trading based on predictions
    lstm_trades = simulate_lstm_trading(y_pred, test_indices, test_close)

    # 9. Save the model
    model_save_path = os.path.join(os.path.dirname(__file__), f"lstm_{symbol}_2class.pt")
    torch.save(model.state_dict(), model_save_path)

    # Return the trade DataFrame (like other strategies)
    return lstm_trades

def lstm_predict_on_dates(symbol, start_date, end_date, lookback=30, batch_size=64, epochs=15, lr=1e-3, threshold=0.001, 
                          stop_loss_pct=0.03, max_holding_days=15, take_profit_pct=0.05):
    """
    LSTM strategy for user-specified date range.
    
    Parameters:
    - symbol: Stock symbol
    - start_date: Start date for prediction (datetime or string)
    - end_date: End date for prediction (datetime or string)
    - lookback: Number of days to look back for prediction
    - batch_size: Batch size for training
    - epochs: Number of training epochs
    - lr: Learning rate
    - threshold: Threshold for up/down classification
    
    Returns:
    - DataFrame with trade information matching other strategies
    """
    # Convert dates to datetime if needed
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Load data with extra lookback period for training
    lookback_start = start_date - pd.Timedelta(days=lookback * 2)
    df = load_stock_data(symbol)
    df = df.sort_index()
    df = df[(df.index >= lookback_start) & (df.index <= end_date)]
    
    if len(df) < lookback * 3:
        print(f"Not enough data for LSTM training. Need at least {lookback * 3} days, got {len(df)}")
        return pd.DataFrame()

    # 1b. Add technical indicators (using only pandas/numpy)
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    # Exponential Moving Average
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    # Bollinger Bands (20)
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    # 1c. Drop rows with NaN values from indicator calculation
    df = df.dropna()
    # Remove MACD outliers (extreme values)
    df = df[(df['macd'].abs() < 100) & (df['macd_hist'].abs() < 100)]

    # Save original close price before normalization
    df["close_orig"] = df["close"]

    # 2. Create 2-class label: 1 (up > threshold), 0 (down < -threshold)
    returns = (df["close"].shift(-1) - df["close"]) / df["close"]
    labels = np.where(returns > threshold, 1, 0)
    df["target"] = labels
    df = df.iloc[:-1]  # Remove last row (no next day)

    # 3. Normalize OHLCV features
    scaler = StandardScaler()
    features = ["open", "high", "low", "close", "volume", "sma_5", "sma_10", "ema_10", "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14", "bb_upper", "bb_middle", "bb_lower"]
    df[features] = scaler.fit_transform(df[features])

    # 4. Create sliding windows
    def create_windows(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[features].iloc[i:i+lookback].values)
            y.append(data["target"].iloc[i+lookback])
        return np.array(X), np.array(y)

    X, y = create_windows(df, lookback)

    # 5. Split data (70% train, 30% test) - but only for training
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 6. Build the LSTM 2-class classifier
    class LSTMClassifier2(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_classes=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out  # logits

    model = LSTMClassifier2(input_size=len(features))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 7. Train the model
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += len(yb)
        train_acc = train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {np.mean(train_losses):.4f} - Train Acc: {train_acc:.4f}")

    # 8. Save the model
    model_save_path = os.path.join(os.path.dirname(__file__), f"lstm_{symbol}_2class.pt")
    torch.save(model.state_dict(), model_save_path)

    # 9. Make predictions on user's exact date range
    # Get data for user's date range only
    user_data = df[(df.index >= start_date) & (df.index <= end_date)]
    
    if len(user_data) < lookback:
        print(f"Not enough data in user date range. Need at least {lookback} days, got {len(user_data)}")
        return pd.DataFrame()

    # Create windows for user's data
    X_user, y_user = create_windows(user_data, lookback)
    X_user = torch.tensor(X_user, dtype=torch.float32)
    
    # Make predictions
    model.eval()
    y_pred = []
    with torch.no_grad():
        for i in range(len(X_user)):
            x = X_user[i:i+1]  # Single sample
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            y_pred.extend(preds.numpy())

    # 10. Simulate trading on user's date range
    user_indices = user_data.index[lookback:lookback+len(y_pred)]
    user_close = user_data.loc[user_indices, "close_orig"].values

    # Simulate trading based on predictions
    lstm_trades = simulate_lstm_trading(y_pred, user_indices, user_close,
                                       stop_loss_pct=stop_loss_pct,
                                       max_holding_days=max_holding_days,
                                       take_profit_pct=take_profit_pct)

    return lstm_trades

def lstm_predict_from_saved_model(symbol, start_date, end_date, lookback=30):
    """
    Load saved LSTM model and make predictions on user-specified dates.
    
    Parameters:
    - symbol: Stock symbol
    - start_date: Start date for prediction (datetime or string)
    - end_date: End date for prediction (datetime or string)
    - lookback: Number of days to look back for prediction (must match training)
    
    Returns:
    - DataFrame with trade information matching other strategies
    """
    # Convert dates to datetime if needed
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Check if saved model exists
    model_save_path = os.path.join(os.path.dirname(__file__), f"lstm_{symbol}_2class.pt")
    if not os.path.exists(model_save_path):
        print(f"No saved model found for {symbol}. Please train the model first using lstm_strategy().")
        return pd.DataFrame()
    
    # Load data with extra lookback period for feature calculation
    lookback_start = start_date - pd.Timedelta(days=lookback * 2)
    df = load_stock_data(symbol)
    df = df.sort_index()
    df = df[(df.index >= lookback_start) & (df.index <= end_date)]
    
    if len(df) < lookback * 2:
        print(f"Not enough data for prediction. Need at least {lookback * 2} days, got {len(df)}")
        return pd.DataFrame()

    # Add technical indicators (same as training)
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    # Exponential Moving Average
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    # Bollinger Bands (20)
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    # Drop rows with NaN values from indicator calculation
    df = df.dropna()
    # Remove MACD outliers (extreme values)
    df = df[(df['macd'].abs() < 100) & (df['macd_hist'].abs() < 100)]

    # Save original close price before normalization
    df["close_orig"] = df["close"]

    # Normalize features (same as training)
    scaler = StandardScaler()
    features = ["open", "high", "low", "close", "volume", "sma_5", "sma_10", "ema_10", "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14", "bb_upper", "bb_middle", "bb_lower"]
    df[features] = scaler.fit_transform(df[features])

    # Create sliding windows function
    def create_windows(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[features].iloc[i:i+lookback].values)
            y.append(0)  # Dummy target for prediction
        return np.array(X), np.array(y)

    # Get data for user's date range only
    user_data = df[(df.index >= start_date) & (df.index <= end_date)]
    
    if len(user_data) < lookback:
        print(f"Not enough data in user date range. Need at least {lookback} days, got {len(user_data)}")
        return pd.DataFrame()

    # Create windows for user's data
    X_user, y_user = create_windows(user_data, lookback)
    X_user = torch.tensor(X_user, dtype=torch.float32)
    
    # Load saved model
    class LSTMClassifier2(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_classes=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out  # logits

    model = LSTMClassifier2(input_size=len(features))
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    # Make predictions
    y_pred = []
    with torch.no_grad():
        for i in range(len(X_user)):
            x = X_user[i:i+1]  # Single sample
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            y_pred.extend(preds.numpy())

    # Simulate trading on user's date range
    user_indices = user_data.index[lookback:lookback+len(y_pred)]
    user_close = user_data.loc[user_indices, "close_orig"].values

    # Simulate trading based on predictions
    lstm_trades = simulate_lstm_trading(y_pred, user_indices, user_close)

    return lstm_trades



