import tensorflow as tf
from keras import Model, activations
from keras.layers import Embedding, Layer

from models.tensorflow.mlp import MLP


class FeatureSelection(Layer):
    def __init__(self, num_hidden, dim_hidden, dropout):
        super().__init__()

        self.num_hidden = num_hidden
        self.dim_hidden = dim_hidden
        self.dropout = dropout

    def build(self, input_shape):
        dim_inputs = input_shape[-1]

        self.gate_1 = MLP(
            num_hidden=self.num_hidden,
            dim_hidden=self.dim_hidden,
            dim_out=dim_inputs,
            dropout=self.dropout,
            batch_norm=False,
            name="feature_selection_gate_1",
        )

        self.gate_2 = MLP(
            num_hidden=self.num_hidden,
            dim_hidden=self.dim_hidden,
            dim_out=dim_inputs,
            dropout=self.dropout,
            batch_norm=False,
            name="feature_selection_gate_2",
        )

        self.built = True

    def call(self, inputs, training=None):
        gate_score_1 = self.gate_1(inputs, training=training)  # (1, dim_input)
        out_1 = 2.0 * activations.sigmoid(gate_score_1) * inputs  # (bs, dim_input)

        gate_score_2 = self.gate_2(inputs, training=training)  # (1, dim_input)
        out_2 = 2.0 * activations.sigmoid(gate_score_2) * inputs  # (bs, dim_input)

        return out_1, out_2  # (bs, dim_feature), (bs, dim_feature)


class Aggregation(Layer):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def build(self, input_shapes):
        input_shape_1, input_shape_2 = input_shapes
        dim_inputs_1 = input_shape_1[-1]
        dim_inputs_2 = input_shape_2[-1]

        self.dim_head_1 = dim_inputs_1 // self.num_heads
        self.dim_head_2 = dim_inputs_2 // self.num_heads

        self.w_1 = self.add_weight(
            name="w_1",
            shape=(self.dim_head_1, self.num_heads, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.w_2 = self.add_weight(
            name="w_2",
            shape=(self.dim_head_2, self.num_heads, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.w_12 = self.add_weight(
            name="w_12",
            shape=(self.num_heads, self.dim_head_1, self.dim_head_2, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(shape=(1, self.num_heads, 1), initializer="zeros", trainable=True)
        self.built = True

    def call(self, inputs):
        # bilinear aggregation of the two latent representations
        # y = b + w_1.T o_1 + w_2.T o_2 + o_1.T W_3 o_2
        inputs_1, inputs_2 = inputs
        inputs_1 = tf.reshape(inputs_1, (-1, self.num_heads, self.dim_head_1))  # (bs, num_heads, dim_head_1)
        inputs_2 = tf.reshape(inputs_2, (-1, self.num_heads, self.dim_head_2))  # (bs, num_heads, dim_head_2)

        first_order = tf.einsum("bhi,iho->bho", inputs_1, self.w_1)  # (bs, num_heads, 1)
        first_order += tf.einsum("bhi,iho->bho", inputs_2, self.w_2)  # (bs, num_heads, 1)
        second_order = tf.einsum("bhi,hijo,bhj->bho", inputs_1, self.w_12, inputs_2)  # (bs, num_heads, 1)

        out = tf.reduce_sum(first_order + second_order + self.bias, axis=1)  # (bs, 1)

        return out


class FinalMLP(Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding,
        num_hidden_fs,
        dim_hidden_fs,
        num_hidden_1,
        dim_hidden_1,
        num_hidden_2,
        dim_hidden_2,
        num_heads,
        dropout=0.0,
        name="FinalMLP",
    ):
        super().__init__(name=name)

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        # embedding layer
        self.embedding = Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            name="embedding",
        )

        # feature selection layer that projects a learnable vector to the flatened embedded feature space
        self.feature_selection = FeatureSelection(
            num_hidden=num_hidden_fs,
            dim_hidden=dim_hidden_fs,
            dropout=dropout,
        )

        # branch 1
        self.interaction_1 = MLP(
            num_hidden=num_hidden_1,
            dim_hidden=dim_hidden_1,
            dropout=dropout,
            name="MLP_1",
        )
        # branch 2
        self.interaction_2 = MLP(
            num_hidden=num_hidden_2,
            dim_hidden=dim_hidden_2,
            dropout=dropout,
            name="MLP_2",
        )

        # final aggregation layer
        self.aggregation = Aggregation(num_heads=num_heads)

    def call(self, inputs, training=None):
        embeddings = self.embedding(inputs, training=training)  # (bs, num_emb, dim_emb)
        embeddings = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))  # (bs, num_emb * dim_emb)

        # weight features of the two streams using a gating mechanism
        # (bs, num_emb * dim_emb), (bs, num_emb * dim_emb)
        emb_1, emb_2 = self.feature_selection(embeddings, training=training)

        # get interactions from the two branches
        # (bs, dim_hidden_1), (bs, dim_hidden_1)
        latent_1 = self.interaction_1(emb_1, training=training)
        latent_2 = self.interaction_2(emb_2, training=training)

        # merge the representations using an aggregation scheme
        logits = self.aggregation([latent_1, latent_2])  # (bs, 1)
        outputs = activations.sigmoid(logits)  # (bs, 1)

        return outputs
