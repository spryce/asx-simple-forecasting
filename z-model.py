"""Python script updated with z-score normalization () implemented to preprocess stock
market data for model training and forecasting.
This preserves the scale and dynamics of the financial data while standardizing it."""

import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import json

with open("filtered_data.json", "r") as f:
    data = json.load(f)

# Extract date and stock price data
time = np.array(data["date"])
series = np.array(data["data"])

# Define constants
SPLIT_TIME = int(len(series) * 0.8)  # 80% training, 20% validation
WINDOW_SIZE = 30
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

# Split the data into training and validation sets
series_train = series[:SPLIT_TIME]
series_valid = series[SPLIT_TIME:]

# Z-score normalization
mean = np.mean(series_train)
std = np.std(series_train)
series_train_normalized = (series_train - mean) / std
series_valid_normalized = (series_valid - mean) / std


# Define a function to create windowed datasets
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)  # Add a feature dimension
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# Prepare the training dataset
train_dataset = windowed_dataset(
    series_train_normalized, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE
)

# Build the model
model = keras.models.Sequential(
    [
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ]
)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.Huber(),
    metrics=["mae"],
)

# Train the model
history = model.fit(train_dataset, epochs=2)


# Define forecasting function
def model_forecast(model, series, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=False)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(1).prefetch(tf.data.AUTOTUNE)
    return model.predict(dataset)


# Generate forecast for validation set
forecast_normalized = model_forecast(
    model, series_valid_normalized, WINDOW_SIZE
).squeeze()
forecast = forecast_normalized * std + mean  # Rescale back to original scale

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time[SPLIT_TIME:], series_valid, label="Validation Data")
plt.plot(time[SPLIT_TIME:], forecast, label="Forecast", color="orange")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.grid()
plt.show()
