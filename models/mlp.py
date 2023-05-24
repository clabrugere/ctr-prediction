import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def __init__(self, num_hidden, dim_hidden, dim_out=None, dropout=0.0):
        super().__init__()

        self.dim_out = dim_out
        self.layers = []
        for _ in range(num_hidden):
            self.layers.append(tf.keras.layers.Dense(dim_hidden))
            self.layers.append(tf.keras.layers.BatchNormalization())
            self.layers.append(tf.keras.layers.ReLU())
            self.layers.append(tf.keras.layers.Dropout(dropout))

        if dim_out:
            self.projection_head = tf.keras.layers.Dense(dim_out)

    def call(self, inputs, training=False):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)

        if self.dim_out:
            outputs = self.projection_head(outputs, training=training)

        return outputs
