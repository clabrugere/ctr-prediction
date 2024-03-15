import tensorflow as tf
from keras import Model, activations
from keras.layers import Dense, Embedding

from models.tensorflow.mlp import MLP


class DeepWide(Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding,
        num_hidden,
        dim_hidden,
        dropout=0.0,
        name="DeepWide",
    ):
        super().__init__(name=name)

        self.dim_input = dim_input
        self.dim_emb = dim_embedding

        # embedding layer
        self.embedding = Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            name="embedding",
        )

        # wide part
        self.wide = Embedding(
            input_dim=num_embedding,
            output_dim=1,
            name="wide_emb",
        )

        # deep part
        self.deep = MLP(num_hidden=num_hidden, dim_hidden=dim_hidden, dim_out=1, dropout=dropout)

        # final layer
        self.projection_head = Dense(1, name="projection_head")

    def call(self, inputs, training=None):
        embeddings = self.embedding(inputs, training=training)  # (bs, dim_input, dim_emb)
        embeddings = tf.reshape(embeddings, (-1, self.dim_input * self.dim_emb))  # (bs, dim_input * dim_emb)

        latent_wide = tf.reduce_sum(self.wide(inputs, training=training), axis=1)  # (bs, 1)
        latent_deep = self.deep(embeddings, training=training)  # (bs, 1)

        logits = latent_deep + latent_wide  # (bs, 1)
        output = activations.sigmoid(logits)  # (bs, 1)

        return output
