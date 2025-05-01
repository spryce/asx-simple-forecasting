import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


# Load your data from the file
import json

with open("filtered_data.json", "r") as f:
    data = json.load(f)

# Extract the time and series data
time = np.array(data["date"])
series = np.array(data["data"])

# Set parameters
SPLIT_TIME = int(len(series) * 0.8)  # 80% training, 20% validation
WINDOW_SIZE = 30
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

# Split the data into training and validation
time_train = time[:SPLIT_TIME]
series_train = series[:SPLIT_TIME]
time_valid = time[SPLIT_TIME:]
series_valid = series[SPLIT_TIME:]


""" # Define a function to create windowed datasets
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):


    series = tf.expand_dims(series, axis=-1)  # Add an extra dimension
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset """


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # Normalize the series to the range [0, 1]
    scaler = MinMaxScaler()
    series = scaler.fit_transform(series.reshape(-1, 1))  # Normalizes the series

    # Add the extra dimension required for TensorFlow models
    series = tf.expand_dims(series.squeeze(), axis=-1)

    # Convert series into a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)

    # Split each window into inputs (window[:-1]) and labels (window[-1])
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    # dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset, scaler


# Prepare the training dataset
train_dataset, scaler = windowed_dataset(
    series_train, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE
)

for X, y in train_dataset.take(3):  # Inspect the first 3 batches
    print(
        f"X shape: {X.shape}, should be ({BATCH_SIZE, WINDOW_SIZE, 1})"
    )  # Should be (batch_size, window_size, 1)
    print(
        f"y shape: {y.shape}, should be ({BATCH_SIZE}, 1)"
    )  # Should be (batch_size, 1)

# Build the model
""" model = keras.models.Sequential(
    [
        keras.Input((WINDOW_SIZE, 1)),
        keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
        # keras.layers.Lambda(lambda x: x * 100.0),
    ]
) """

model = keras.models.Sequential(
    [
        keras.layers.Dense(10, activation="relu", input_shape=[WINDOW_SIZE]),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1),
    ]
)

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=1e-4)

loss = keras.losses.mean_squared_error
# loss=keras.losses.Huber()

model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])

# Train the model
history = model.fit(train_dataset, epochs=5)

""" plt.plot(history.history["loss"], label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show() """


# Define a forecasting function
def model_forecast(model, series, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    forecast = model.predict(dataset)
    return forecast


""" # Generate forecast for validation set
forecast = model_forecast(
    model, series[SPLIT_TIME - WINDOW_SIZE : -1], WINDOW_SIZE
).squeeze() """

""" forecast_series = series[SPLIT_TIME - WINDOW_SIZE : -1].reshape(-1, WINDOW_SIZE, 1)
forecast = model.predict(forecast_series).squeeze()
# Generate forecast for validation set
forecast = model_forecast(model, forecast_series, WINDOW_SIZE).squeeze()
forecast_rescaled = scaler.inverse_transform(forecast.reshape(-1, 1)).squeeze() """

print(
    f"Training data range: {series[:SPLIT_TIME].min()} to {series[:SPLIT_TIME].max()}"
)
print(
    f"Validation data range: {series[SPLIT_TIME:].min()} to {series[SPLIT_TIME:].max()}"
)

print(f"series_train.shape: {series_train.shape}")
print(f"series_valid.shape: {series_valid.shape}")

forecast_series = series[SPLIT_TIME - WINDOW_SIZE : -1]  # Correct slicing

# Use sliding windows to create inputs for forecasting
ds = tf.data.Dataset.from_tensor_slices(forecast_series)
ds = ds.window(WINDOW_SIZE, shift=1, drop_remainder=True)
ds = ds.flat_map(lambda w: w.batch(WINDOW_SIZE))
ds = ds.batch(BATCH_SIZE).prefetch(1)  # Batch size of 1 for prediction ??

# Predict using the model
forecast = model.predict(ds).squeeze()

forecast_rescaled = scaler.inverse_transform(forecast.reshape(-1, 1)).squeeze()

print(f"Forecast shape: {forecast_rescaled.shape}")
print(f"Validation series shape: {series[SPLIT_TIME:].shape}")


print("First few predictions:", forecast_rescaled[:10])
print("Max prediction:", np.max(forecast_rescaled))
print("Min prediction:", np.min(forecast_rescaled))


print(f"Length of time_valid: {len(time_valid)}")
print(f"Length of series_valid: {len(series_valid)}")
print(f"Length of forecast: {len(forecast_rescaled)}")

""" print(f"forecast shape: {forecast.shape}")
print(f"series_valid shape: {series_valid.shape}")
print(f"First few forecast values: {forecast[:10]}")
print(f"First few series_valid values: {series_valid[:10]}") """


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_valid, series_valid, label="Validation Data")
plt.plot(time_valid, forecast_rescaled, label="Forecast", color="orange")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
