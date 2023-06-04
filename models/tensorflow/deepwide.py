import tensorflow as tf
from models.tensorflow.mlp import MLP


class DeepWide(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        num_hidden=3,
        dim_hidden=100,
        dropout=0.0,
        name="DeepWide",
    ):
        super().__init__(name=name)

        self.dim_input = dim_input
        self.dim_emb = dim_embedding

        # embedding layers
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            input_length=dim_input,
            name="embedding",
        )

        # wide part which is a linear model without bias
        self.wide = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=1,
            input_length=dim_input,
            name="wide_emb",
        )

        # interaction layer using MLP
        self.interaction_mlp = MLP(num_hidden=num_hidden, dim_hidden=dim_hidden, dim_out=1, dropout=dropout)

        # final layer
        self.projection_head = tf.keras.layers.Dense(1, name="projection_head")

        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=False):
        embeddings = self.embedding(inputs, training=training)  # (bs, dim_input, dim_emb)
        embeddings = tf.reshape(embeddings, (-1, self.dim_input * self.dim_emb))  # (bs, dim_input * dim_emb)

        latent_wide = tf.reduce_sum(self.wide(inputs, training=training), axis=1)  # (bs, 1)
        latent_deep = self.interaction_mlp(embeddings, training=training)  # (bs, 1)

        logits = latent_deep + latent_wide  # (bs, 1)
        output = tf.nn.sigmoid(logits)  # (bs, 1)

        return output
