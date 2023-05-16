import tensorflow as tf


class AutoInt(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        regularization=0.00001,
        num_heads=1,
        dim_key=16,
        num_hidden=2,
        dim_hidden=16,
        num_attention=1,
        dropout=0.0,
        name="AutoInt",
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

        # interaction layer using MLP
        self.interaction_mlp = tf.keras.Sequential(name="MLP")
        for _ in range(num_hidden):
            self.interaction_mlp.add(tf.keras.layers.Dense(dim_hidden))
            self.interaction_mlp.add(tf.keras.layers.BatchNormalization())
            self.interaction_mlp.add(tf.keras.layers.ReLU())
            self.interaction_mlp.add(tf.keras.layers.Dropout(dropout))

        # projects mlp output on R
        self.interaction_mlp.add(tf.keras.layers.Dense(1, name="mlp_projection_head"))

        # interaction layer using stacked self-attention
        self.interaction_attention = []
        for i in range(num_attention):
            self.interaction_attention.append((
                tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=dim_key,
                    dropout=dropout,
                    name=f"interaction_layer_{i+1}",
                ),
                tf.keras.layers.LayerNormalization()
                # tf.keras.layers.Attention(
                #     use_scale=True,
                #     score_mode="dot",
                #     dropout=dropout,
                #     name=f"interaction_layer_{i+1}",
                # )
            ))
        
        # projects attention layers on R
        self.attn_projection_head = tf.keras.layers.Dense(1, name="attn_projection_head")

        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=False):
        # extract embeddings
        embeddings = self.embedding(inputs, training=training) # (batch_size, nb_features, embedding_dim)

        # interaction layer using stack of self attention layers like Transformer's encoder
        attn_in = embeddings
        for (interaction_layer, layer_norm) in self.interaction_attention:
            # (batch_size, nb_features, embedding_dim)
            attn_out = interaction_layer(attn_in, attn_in, training=training) + attn_in
            attn_out = layer_norm(attn_out)
            attn_in = attn_out

        # project concatenated attention outputs
        # (batch_size, nb_features * embedding_dim)
        attn_out = tf.reshape(attn_out, (-1, self.dim_input * self.dim_embedding))
        attn_out = self.attn_projection_head(attn_out, training=training) # (batch_size, 1)

        # interaction layer using MLP
        mlp_out = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding)) # (batch_size, nb_features * embedding_dim)
        mlp_out = self.interaction_mlp(mlp_out, training=training) # (batch_size, 1)

        # add the two representations
        logits = tf.add(attn_out, mlp_out, name="sum_representations") # (batch_size, 1)
        outputs = tf.nn.sigmoid(logits) # (batch_size, 1)

        return outputs
