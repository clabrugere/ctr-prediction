import tensorflow as tf


class DeepWide(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        n_hidden=3,
        dim_hidden=100,
        regularization=0.00001,
        name="DeepWide",
    ):
        super().__init__(name=name)

        self.dim_input = dim_input
        self.dim_emb = dim_embedding

        # embedding layers
        self.sparse_emb = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=1,
            input_length=dim_input,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=regularization),
            name="sparse_emb",
        )
        self.dense_emb = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            input_length=dim_input,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=regularization),
            name="dense_emb",
        )

        # MLP
        self.mlp = tf.keras.Sequential(name="MLP")
        for i in range(n_hidden):
            self.mlp.add(
                tf.keras.layers.Dense(
                    dim_hidden,
                    activation="swish",
                    name=f"dense_{i+1}",
                )
            )

        # final layer
        self.projection_head = tf.keras.layers.Dense(1, name="projection_head")

        # not really necessary but its annoying to print summary and plot model without it
        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=False):
        emb_sparse = self.sparse_emb(inputs, training=training) # (batch size, num_embedding, dim_embedding)
        emb_sparse = tf.reduce_sum(emb_sparse, axis=1) # (batch size, 1)

        emb_dense = self.dense_emb(inputs, training=training) # (batch size, num_embedding, dim_embedding)
        emb_dense = tf.reshape(emb_dense, (-1, self.dim_input * self.dim_emb)) # (batch size, num_embedding * dim_embedding)
        
        latent = self.mlp(emb_dense, training=training) # (batch size, dim_hidden)

        logits = self.projection_head(latent, training=training) # (batch size, 1)
        logits = logits + emb_sparse # (batch size, 1)

        output = tf.nn.sigmoid(logits) # (batch size, 1)

        return output
