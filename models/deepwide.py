import tensorflow as tf
from models.mlp import MLP


class DeepWide(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        num_hidden=3,
        dim_hidden=100,
        regularization=1e-5,
        dropout=0.0,
        name="DeepWide",
    ):
        super().__init__(name=name)

        self.dim_input = dim_input
        self.dim_emb = dim_embedding

        # embedding layers
        self.dense_emb = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            input_length=dim_input,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=regularization),
            name="dense_emb",
        )
        
        # wide part which is a linear model without bias
        self.wide_emb = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=1,
            input_length=dim_input,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=regularization),
            name="wide_emb",
        )

        # interaction layer using MLP
        self.interaction_mlp = MLP(num_hidden=num_hidden, dim_hidden=dim_hidden, dropout=dropout)

        # final layer
        self.projection_head = tf.keras.layers.Dense(1, name="projection_head")

        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=False):
        embedding = self.dense_emb(inputs, training=training) # (batch size, num_embedding, dim_embedding)
        embedding = tf.reshape(embedding, (-1, self.dim_input * self.dim_emb)) # (batch size, num_embedding * dim_embedding)
        
        wide = self.wide_emb(inputs, training=training) # (batch size, num_embedding, 1)
        wide = tf.reduce_sum(wide, axis=1) # (batch size, 1)
        
        latent = self.interaction_mlp(embedding, training=training) # (batch size, dim_hidden)

        logits = self.projection_head(latent, training=training) + wide # (batch size, 1)
        output = tf.nn.sigmoid(logits) # (batch size, 1)

        return output
