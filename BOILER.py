import pandas as pd
from binance.client import Client
from binance.enums import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scikeras.wrappers import KerasClassifier
#from keras.wrappers.scikit_learn import KerasClassifier

# Initialize Binance API client
client = Client(api_key='api_key', api_secret='api_secret')

# Define parameters
symbol = 'BTCBUSD'
interval = Client.KLINE_INTERVAL_1HOUR
lookback = 72  # Increase lookback period to capture more price movements
threshold_range = [0.6, 0.7, 0.8]  # Range of threshold values to test
stop_loss = 0.01
take_profit = 0.02
ORDER_TYPE_STOP_LOSS = "STOP_LOSS_LIMIT"
ORDER_TYPE_TAKE_PROFIT = "TAKE_PROFIT_LIMIT"
quantity = 0.001

# Retrieve historical data from Binance
klines = client.get_historical_klines(symbol, interval, "100 days ago UTC")
# Convert klines to a pandas dataframe
df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignored'])

# Clean up the dataframe
df = df.drop(['timestamp', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignored'], axis=1)
df['open'] = df['open'].astype(float)
df['high'] = df['high'].astype(float)
df['low'] = df['low'].astype(float)
df['close'] = df['close'].astype(float)
df = df.tail(lookback)

# Calculate RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100.0 - (100.0 / (1.0 + rs))

# Add machine learning
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df[['open', 'high', 'low', 'close']].values)
data = tf.constant(data, dtype=tf.float32)

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu'),  # Use a convolutional neural network for improved accuracy
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define hyperparameters for GridSearchCV
param_grid = {
    #'learning_rate': [0.01, 0.1],
    'epochs': [50, 100],
    'batch_size': [32, 64]
}

#model_cv = GridSearchCV(estimator=KerasClassifier(build_fn=create_model), param_grid=param_grid, cv=3, scoring=accuracy_score)
model_cv = GridSearchCV(
    estimator=KerasClassifier(model=create_model), 
    param_grid=param_grid, 
    cv=3, 
    scoring='accuracy',
    n_jobs=-1
)

# Train the model using GridSearchCV
x_train = data[:-1].numpy() # convert X_train tensor to numpy array
y_train = tf.where(df['close'].diff().shift(-1) > 0, 1.0, 0.0)[:-1]
model_cv.fit(x_train, y_train.numpy().astype(int))  # convert y_train to numpy array and then to integers

# Print best hyperparameters and score
print("Best hyperparameters: ", model_cv.best_params_)
print("Best score: ", model_cv.best_score_)

# Use the model to make predictions
best_threshold = None
best_accuracy = 0
for threshold in threshold_range:
    prediction = model_cv.predict(data[-1:])
    if prediction > threshold:
        # Buy BTC
        order = client.create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_STOP_LOSS,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=df['close'].iloc[-1] * (1 + stop_loss),
            stopPrice=df['close'].iloc[-1] * (1 + stop_loss),
            newOrderRespType='FULL'
        )
        print("Buy order placed")
    elif prediction < (1 - threshold):
        # Sell BTC
        order = client.create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_STOP_LOSS,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=df['close'].iloc[-1] * (1 - stop_loss),
            stopPrice=df['close'].iloc[-1] * (1 - stop_loss),
            newOrderRespType='FULL'
        )
        print("Sell order placed")
    else:("No action taken")
    # Calculate accuracy of model for current threshold value
    accuracy = sum(y_train.numpy() == (model_cv.predict(x_train) > 0.5).astype(int)) / len(y_train)
print("Threshold: ", threshold)
print("Accuracy: ", accuracy)
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_threshold = threshold

# Implement risk management strategy
# Limit number of trades per day
max_trades_per_day = 20
trades_today = 0
for order in client.get_all_orders(symbol=symbol):
    if order['status'] == 'FILLED' and pd.Timestamp.now().date() == pd.Timestamp(order['updateTime'], unit='ms').date():
        trades_today += 1
if trades_today >= max_trades_per_day:
    print("Max trades per day reached, no action taken")
else:
    # Set stop loss and take profit based on a percentage of your account balance
    account_info = client.get_account()
    balance = float(account_info['balances'][0]['free'])
    stop_loss_price = df['close'].iloc[-1] * (1 - stop_loss)
    take_profit_price = df['close'].iloc[-1] * (1 + take_profit)
    stop_loss_quantity = balance * stop_loss / stop_loss_price
    take_profit_quantity = balance * take_profit / take_profit_price

    # Place stop loss and take profit orders
    stop_loss_order = client.create_order(
        symbol=symbol,
        side=SIDE_SELL,
        type=ORDER_TYPE_STOP_LOSS,
        timeInForce=TIME_IN_FORCE_GTC,
        quantity=stop_loss_quantity,
        price=stop_loss_price,
        stopPrice=stop_loss_price,
        newOrderRespType='FULL'
    )
    take_profit_order = client.create_order(
        symbol=symbol,
        side=SIDE_SELL,
        type=ORDER_TYPE_TAKE_PROFIT,
        timeInForce=TIME_IN_FORCE_GTC,
        quantity=take_profit_quantity,
        price=take_profit_price,
        stopPrice=take_profit_price,
        newOrderRespType='FULL'
    )
    print("Stop loss and take profit orders placed")
