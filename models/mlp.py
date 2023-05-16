import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(
        self, dim_input, n_hidden, dim_hidden, dim_out, dropout=0.0, name="MLP"
    ):
        super().__init__(name=name)

        self.blocks = tf.keras.Sequential(name="MLP")
        for _ in range(n_hidden):
            self.blocks.add(tf.keras.layers.Dense(dim_hidden))
            self.blocks.add(tf.keras.layers.BatchNormalization())
            self.blocks.add(tf.keras.layers.ReLU())
            self.blocks.add(tf.keras.layers.Dropout(dropout))

        self.projection_head = tf.keras.layers.Dense(dim_out)

        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=False):
        latent = self.blocks(inputs, training=training)
        out = self.projection_head(latent)

        return out
