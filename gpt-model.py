import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

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


# Define a function to create windowed datasets
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)  # Add an extra dimension
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# Prepare the training dataset
train_dataset = windowed_dataset(
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
model = keras.models.Sequential(
    [
        keras.layers.Dense(10, activation="relu", input_shape=[WINDOW_SIZE]),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1),
    ]
)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(), loss=keras.losses.Huber(), metrics=["mae"]
)

# Train the model
history = model.fit(train_dataset, epochs=2)


# Define a forecasting function
def model_forecast(model, series, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    forecast = model.predict(dataset)
    return forecast


# Generate forecast for validation set
forecast = model_forecast(
    model, series[SPLIT_TIME - WINDOW_SIZE : -1], WINDOW_SIZE
).squeeze()


print(f"Length of time_valid: {len(time_valid)}")
print(f"Length of series_valid: {len(series_valid)}")
print(f"Length of forecast: {len(forecast)}")

""" print(f"forecast shape: {forecast.shape}")
print(f"series_valid shape: {series_valid.shape}")
print(f"First few forecast values: {forecast[:10]}")
print(f"First few series_valid values: {series_valid[:10]}") """


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_valid, series_valid, label="Validation Data")
plt.plot(time_valid, forecast, label="Forecast", color="orange")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
