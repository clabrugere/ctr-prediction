import tensorflow as tf

from helpers.callbacks import EpochTimer


def train(
    model,
    ds_train,
    ds_val,
    epochs=20,
    lr=0.1,
    min_lr=1e-6,
    verbose=1,
    return_train_time=False,
):

    tf.keras.backend.clear_session()
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["mse"]
    )
    timer = EpochTimer()
    callbacks = [
        timer,
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, mode="min"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    ]
    
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    if return_train_time:
        return history, timer.durations
    else:
        return history
