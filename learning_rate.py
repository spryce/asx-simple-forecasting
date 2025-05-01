### ==================+++++========================
# Adjust learning rate
# ==================+++++========================


def adjust_learning_rate(dataset):
    """Fit model using different learning rates

    Args:
        dataset (tf.data.Dataset): train dataset

    Returns:
        tf.keras.callbacks.History: callback history
    """

    model = create_uncompiled_model()

    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-5 * 10 ** (epoch / 20)
    )
    optimizer = keras.optimizers.SGD(momentum=0.9)

    # Compile the model passing in the appropriate loss
    model.compile(loss=keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])

    history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])

    return history


# Run the training with dynamic LR
lr_history = adjust_learning_rate(train_dataset)

plt.semilogx(lr_history.history["learning_rate"], lr_history.history["loss"])
plt.show()
