# This is a simple time series forecasting model using TensorFlow and Keras.

import json
import numpy as np

import tensorflow as tf
import keras

import matplotlib.pyplot as plt

DATA_PATH = "./filtered_data.json"  # Path to the JSON file containing the data
SPLIT_TIME = 4500  # Time to split the data into training and validation sets (20 years = approx 5000 days)
WINDOW_SIZE = 64
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 1000


def plot_series(time, series, format="-", title="", label=None, start=0, end=None):
    """Plot the series"""
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    if label:
        plt.legend()
    plt.grid(True)
    # plt.show()


# generate a function to parse data from the json file
def parse_data(file_path):
    """Parse the data from the JSON file"""

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    dates = np.array(data["date"])
    closing_prices = np.array(data["data"]).astype(float)

    # print the length of the data
    print(f"Length of dates: {len(dates)}")
    print(f"Length of closing prices: {len(closing_prices)}")

    return dates, closing_prices


TIME, SERIES = parse_data(DATA_PATH)

# Plot the series!
""" plt.figure(figsize=(10, 6))
plot_series(TIME, SERIES)
plt.show()
quit() """
# ==================+++++========================
# Data Preprocessing
# ==================+++++========================


def train_val_split(time, series):
    """Splits time series into train and validations sets"""
    time_train = time[:SPLIT_TIME]
    series_train = series[:SPLIT_TIME]
    time_valid = time[SPLIT_TIME:]
    series_valid = series[SPLIT_TIME:]

    return time_train, series_train, time_valid, series_valid


# Split the dataset
time_train_set, series_train_set, time_valid_set, series_valid_set = train_val_split(
    TIME, SERIES
)


""" plt.figure(figsize=(10, 8))
plot_series(time_train_set, series_train_set, title="Training")

plt.figure(figsize=(10, 8))
plot_series(time_valid_set, series_valid_set, title="Validation") """


def windowed_dataset(series, window_size):
    """Creates windowed dataset"""
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    window_count = sum(1 for _ in dataset)
    print(f"Total windows after windowing: {window_count}")  # Should be 4437

    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    flat_window_count = sum(1 for _ in dataset)
    print(f"Total windows after flat_map: {flat_window_count}")  # Should still be 4437

    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    # dataset = dataset.batch(BATCH_SIZE).prefetch(1) # terrible. Why did we do this in the course?
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    # dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    batch_count = sum(1 for _ in dataset)
    print(f"Total batches after batching: {batch_count}")  # Should be 17

    return dataset


# Normalize the data
""" series_normalized = (series_train_set - series_train_set.min()) / (
    series_train_set.max() - series_train_set.min()
) """

# Apply the transformation to the training set
train_dataset = windowed_dataset(series_train_set, window_size=WINDOW_SIZE)


count = len(list(train_dataset.as_numpy_iterator()))
print(f"Number of windows: {count}")

# Formula: Total Windows = Dataset Length - Window Size + 1
print(
    f"Total Windows = {len(series_train_set)} - {WINDOW_SIZE} + 1 = {len(series_train_set) - WINDOW_SIZE + 1}"
)

# Total Batches = Total Windows / Batch Size
print(
    f"Total Batches = {len(series_train_set) - WINDOW_SIZE + 1} // {BATCH_SIZE} = {(len(series_train_set) - WINDOW_SIZE + 1) // BATCH_SIZE}"
)


for X, y in train_dataset.take(3):  # Inspect the first 3 batches
    print(
        f"X shape: {X.shape}, should be ({BATCH_SIZE, WINDOW_SIZE, 1})"
    )  # Should be (batch_size, window_size, 1)
    print(
        f"y shape: {y.shape}, should be ({BATCH_SIZE}, 1)"
    )  # Should be (batch_size, 1)


# quit()


def create_uncompiled_model():
    """Define uncompiled model

    Returns:
        tf.keras.Model: uncompiled model
    """

    model = keras.models.Sequential(
        [
            keras.Input(shape=(WINDOW_SIZE, 1)),
            keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                strides=1,
                activation="relu",
                padding="causal",
            ),
            # keras.layers.LSTM(64, return_sequences=True),
            keras.layers.LSTM(64),
            keras.layers.Dense(30, activation="relu"),
            # keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(1),
            keras.layers.Lambda(lambda x: x * 100.0),
        ]
    )

    return model


uncompiled_model = create_uncompiled_model()
test_batch = train_dataset.take(1)

try:
    """for X_batch, y_batch in test_batch:  # Extract features (X) and labels (y)
    print(f"X_batch shape: {X_batch.shape}")  # Check shape of inputs
    print(f"y_batch shape: {y_batch.shape}")  # Check shape of labels
    print(f"Batch size: {len(X_batch)}")  # Check the batch size"""

    predictions = uncompiled_model.predict(test_batch, verbose=False)
    # print(f"test_batch predictions: {predictions}")
except Exception as e:
    print(f"Error with test batch: {e}")
    print(
        "Your model is not compatible with the dataset you defined earlier. Check that the loss function and last layer are compatible with one another."
    )
else:
    print("Your current architecture is compatible with the windowed dataset! :)")
    print(f"predictions have shape: {predictions.shape}")

uncompiled_model.summary()


def create_model():
    """Creates and compiles the model

    Returns:
        tf.keras.Model: compiled model
    """

    model = create_uncompiled_model()

    # Set the learning rate
    learning_rate = 1e-3  # ?????????????

    # Set the optimizer
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(loss=keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])

    return model


compiled_model = create_model()

# Check ++++++++++++++++++++++++++++++++++++++++++++
predictions = compiled_model.predict(train_dataset.take(1))
print(predictions)


history = compiled_model.fit(train_dataset, epochs=10)

# Plot the training loss for each epoch

loss = history.history["loss"]

print(f"Loss: {loss}")
print(f"Length of loss: {len(loss)}")

epochs = range(20, len(loss))
loss = loss[20:]

""" plt.plot(epochs, loss, "r", label="Training loss")
plt.title("Training loss")
plt.legend(loc=0)
plt.show() """


def compute_metrics(true_series, forecast):
    """Computes MSE and MAE metrics for the forecast"""
    mse = keras.losses.mean_squared_error(true_series, forecast)
    mae = keras.losses.mean_absolute_error(true_series, forecast)
    return mse, mae


def model_forecast(model, series, window_size):
    """Generates a forecast using your trained model"""
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


# Compute the forecast for the validation dataset. Remember you need the last WINDOW SIZE values to make the first prediction
rnn_forecast = model_forecast(
    compiled_model, SERIES[SPLIT_TIME - WINDOW_SIZE : -1], WINDOW_SIZE
).squeeze()


print(f"rnn_forecast shape: {rnn_forecast.shape}")
print(f"series_valid_set shape: {series_valid_set.shape}")
print(f"First few rnn_forecast values: {rnn_forecast[:10]}")
print(f"First few series_valid_set values: {series_valid_set[:10]}")

# Plot the forecast
plt.figure(figsize=(10, 6))
plot_series(time_valid_set, series_valid_set)
plot_series(time_valid_set, rnn_forecast)

mse, mae = compute_metrics(series_valid_set, rnn_forecast)

print(f"mse: {mse:.2f}, mae: {mae:.2f} for forecast")

plt.show()
