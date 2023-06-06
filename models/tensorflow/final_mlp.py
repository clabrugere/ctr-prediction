import tensorflow as tf
from models.tensorflow.mlp import MLP


class FeatureSelection(tf.keras.Model):
    def __init__(self, dim_out, dim_gate, num_hidden=1, dim_hidden=64, dropout=0.0):
        super().__init__()

        self.gate_1 = MLP(
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            dropout=dropout,
            name="fs_gate_1",
        )
        self.gate_1_bias = self.add_weight(shape=(1, dim_gate), initializer="ones", trainable=True)

        self.gate_2 = MLP(
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            dropout=dropout,
            name="fs_gate_2",
        )
        self.gate_2_bias = self.add_weight(shape=(1, dim_gate), initializer="ones", trainable=True)

    def call(self, embeddings):
        gate_score_1 = self.gate_1(self.gate_1_bias)  # (bs, num_emb, dim_emb)
        out_1 = 2.0 * tf.nn.sigmoid(gate_score_1) * embeddings  # (bs, num_emb, dim_emb)

        gate_score_2 = self.gate_2(self.gate_2_bias)  # (bs, num_emb, dim_emb)
        out_2 = 2.0 * tf.nn.sigmoid(gate_score_2) * embeddings  # (bs, num_emb, dim_emb)

        return out_1, out_2


class Aggregation(tf.keras.Model):
    def __init__(self, dim_latent_1, dim_latent_2):
        super().__init__()
        self.out_1 = tf.keras.layers.Dense(1)
        self.out_2 = tf.keras.layers.Dense(1)
        self.out_12 = self.add_weight(shape=(dim_latent_1, dim_latent_2), initializer="glorot_uniform", trainable=True)

    def call(self, latent_1, latent_2):
        first_order = self.out_1(latent_1) + self.out_2(latent_2)  # (bs, 1)
        second_order = tf.einsum("bi,ik,bk->b", latent_1, self.out_12, latent_2)  # (bs, 1)
        out = first_order + tf.expand_dims(second_order, axis=-1)  # (bs, 1)

        return out


class FinalMLP(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        regularization=0.00001,
        dim_hidden_fs=64,
        num_hidden_1=2,
        dim_hidden_1=64,
        num_hidden_2=2,
        dim_hidden_2=64,
        dropout=0.0,
        name="FinalMLP",
    ):
        super().__init__(name=name)

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        # embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            input_length=dim_input,
            embeddings_regularizer=tf.keras.regularizers.l2(regularization),
            name="embedding",
        )

        # feature selection layer that projects a learnable vector to the flatened embedded feature space
        self.feature_selection = FeatureSelection(
            dim_out=dim_input * dim_embedding,
            dim_gate=dim_input,
            dim_hidden=dim_hidden_fs,
            dropout=dropout,
        )

        # branch 1
        self.interaction_1 = MLP(
            num_hidden=num_hidden_1,
            dim_hidden=dim_hidden_1,
            dim_out=None,
            dropout=dropout,
            name="MLP_1",
        )
        # branch 2
        self.interaction_2 = MLP(
            num_hidden=num_hidden_2,
            dim_hidden=dim_hidden_2,
            dim_out=None,
            dropout=dropout,
            name="MLP_2",
        )

        # final aggregation layer
        self.aggregation = Aggregation(dim_latent_1=dim_hidden_1, dim_latent_2=dim_hidden_2)

        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=False):
        embeddings = self.embedding(inputs, training=training)  # (bs, num_emb, dim_emb)
        embeddings = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))  # (bs, num_emb * dim_emb)

        # weight features of the two streams using a gating mechanism
        emb_1, emb_2 = self.feature_selection(embeddings)  # (bs, num_emb * dim_emb), (bs, num_emb * dim_emb)

        # get interactions from the two branches
        latent_1, latent_2 = self.interaction_1(emb_1), self.interaction_2(emb_2)  # (bs, dim_hidden_1), (bs, dim_hidden_2)

        # merge the representations using an aggregation scheme
        logits = self.aggregation(latent_1, latent_2)  # (bs, 1)
        outputs = tf.nn.sigmoid(logits)  # (bs, 1)

        return outputs
