import tensorflow as tf
from keras import Model, activations
from keras.layers import Dense, Embedding, Layer

from models.tensorflow.mlp import MLP


class GDCNS(Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding,
        num_cross,
        num_hidden,
        dim_hidden,
        dropout=0.0,
        name="GDCN",
    ):
        super().__init__(name=name)
        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        self.embedding = Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            name="embedding",
        )

        self.cross = GatedCrossNetwork(num_cross)
        self.projector = MLP(num_hidden, dim_hidden, dim_out=1, dropout=dropout)
        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=None):
        out = self.embedding(inputs, training=training)
        out = tf.reshape(out, (-1, self.dim_input * self.dim_embedding))
        out = self.cross(out, training=training)
        out = self.projector(out, training=training)
        out = activations.sigmoid(out)

        return out


class GDCNP(Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding,
        num_cross,
        num_hidden,
        dim_hidden,
        dropout=0.0,
        name="GDCN",
    ):
        super().__init__(name=name)
        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        self.embedding = Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            name="embedding",
        )

        self.cross = GatedCrossNetwork(num_cross)
        self.mlp = MLP(num_hidden, dim_hidden, dropout=dropout)
        self.projector = Dense(1)
        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=None):
        out = self.embedding(inputs, training=training)
        out = tf.reshape(out, (-1, self.dim_input * self.dim_embedding))

        out_1 = self.cross(out, training=training)
        out_2 = self.mlp(out, training=training)

        out = tf.concat((out_1, out_2), axis=-1)
        out = self.projector(out, training=training)
        out = tf.sigmoid(out)

        return out


class GatedCrossNetwork(Layer):
    def __init__(
        self, num_layers, weights_initializer="glorot_uniform", bias_initializer="zeros", name="GatedCrossNetwork"
    ):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        dim_input = input_shape[-1]

        self._layers = []
        for i in range(self.num_layers):
            W_c = self.add_weight(
                name=f"W_c{i}",
                shape=(dim_input, dim_input),
                initializer=self.weights_initializer,
                dtype=self.dtype,
                trainable=True,
            )
            b_c = self.add_weight(
                name=f"b_c{i}",
                shape=(dim_input,),
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True,
            )
            W_g = self.add_weight(
                name=f"W_g{i}",
                shape=(dim_input, dim_input),
                initializer=self.weights_initializer,
                dtype=self.dtype,
                trainable=True,
            )
            self._layers.append((W_c, b_c, W_g))

    def call(self, inputs, training=None):
        out = inputs  # (bs, dim_input)
        for W_c, b_c, W_g in self._layers:
            out = inputs * (tf.matmul(out, W_c) + b_c) * tf.sigmoid(tf.matmul(out, W_g)) + out  # (bs, dim_input)

        return out
