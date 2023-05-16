from timeit import default_timer as timer

import tensorflow as tf


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        
        self._reset()

    def _reset(self):
        self.durations = []

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_begin(self, epoch, logs=None):
        self._start_timestamp = timer()

    def on_epoch_end(self, epoch, logs=None):
        self.durations.append(timer() - self._start_timestamp)
