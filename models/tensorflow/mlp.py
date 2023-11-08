import tensorflow as tf


class MLP(tf.keras.Sequential):
    def __init__(self, num_hidden, dim_hidden, dim_out=None, batch_norm=True, dropout=0.0, name="MLP"):
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(tf.keras.layers.Dense(dim_hidden))

            if batch_norm:
                layers.append(tf.keras.layers.BatchNormalization())

            layers.append(tf.keras.layers.ReLU())

            if dropout > 0.0:
                layers.append(tf.keras.layers.Dropout(dropout))

        if dim_out:
            layers.append(tf.keras.layers.Dense(dim_out))
        else:
            layers.append(tf.keras.layers.Dense(dim_hidden))

        super().__init__(layers, name=name)
