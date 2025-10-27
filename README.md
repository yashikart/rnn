# rnn

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 1. Download stock data (example: RELIANCE.NS)
data = yf.download("RELIANCE.NS", start="2018-01-01", end="2024-12-31")
close_prices = data['Close'].values.reshape(-1, 1)

# 2. Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

# 3. Create sequences for RNN
def create_sequences(dataset, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(dataset)):
        X.append(dataset[i-seq_len:i])
        y.append(dataset[i])
    return np.array(X), np.array(y)

SEQ_LEN = 60
X, y = create_sequences(scaled_data, SEQ_LEN)

# 4. Train/Test split 80/20
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build a SIMPLE RNN model
model = Sequential([
    SimpleRNN(50, return_sequences=False, input_shape=(SEQ_LEN, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 6. Train
model.fit(X_train, y_train, epochs=15, batch_size=32)

# 7. Predict
preds = model.predict(X_test)
preds = scaler.inverse_transform(preds)

print(preds[:10])




Key Points about RNN (Recurrent Neural Network)

RNN = Recurrent Neural Network — It processes sequential data (data with time or order, like stock prices, text, or sensor readings).

Memory — It remembers previous inputs using a hidden state, which helps it understand time patterns.

Use case — Time-series prediction, speech, or natural language tasks.

Main Layers

SimpleRNN: learns from sequences step by step.

Dense: gives the final output (e.g., the next predicted number).

Activation Function (tanh) — Keeps values between -1 and 1 for stability.

Loss Function (MSE) — Measures how close predictions are to the actual data.

Optimizer (Adam) — Adjusts weights to minimize error automatically.

💻 Code Explained Simply 1️⃣ Generate Sequential Data x = np.linspace(0, 50, 500) y = np.sin(x)

Makes a sine wave (a repeating curve).

The goal is to predict the next point in the wave.

2️⃣ Prepare Data X, Y = [], [] for i in range(len(y) - seq_length): X.append(y[i:i + seq_length]) Y.append(y[i + seq_length])

Each input (X) is a sequence of 20 past values.

Each output (Y) is the next value after those 20.

3️⃣ Reshape Data X = X.reshape((X.shape[0], X.shape[1], 1))

RNNs need 3D input: (samples, timesteps, features) → here: (480, 20, 1)

4️⃣ Build Model model = Sequential([ SimpleRNN(50, activation='tanh', input_shape=(seq_length, 1)), Dense(1) ])

One RNN layer with 50 memory units.

One output neuron (predicts next number).

5️⃣ Compile Model model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

Uses Adam optimizer.

MSE = Mean Squared Error (good for numeric prediction).

6️⃣ Train Model model.fit(X, y, epochs=50, batch_size=16, verbose=0)

Teaches the RNN for 50 rounds (epochs).

7️⃣ Predict and Plot preds = model.predict(X) plt.plot(y, label='True Sequence') plt.plot(preds, label='Predicted Sequence')

Predicts next points.

Plots true vs predicted sine wave.

📈 Output

You’ll see:

The blue line → real sine wave

The red dashed line → RNN’s predicted wave (closely follows the true one)

✅ Summary in one line: This RNN learns the pattern of a sine wave and predicts the next values based on previous steps.


