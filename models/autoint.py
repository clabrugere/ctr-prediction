import tensorflow as tf
from models.mlp import MLP


__AGGREGATION_MODES = ["add", "concat"]


class AutoInt(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        regularization=1e-5,
        num_heads=1,
        dim_key=16,
        num_hidden=2,
        dim_hidden=16,
        num_attention=1,
        dropout=0.0,
        aggregation_mode="add",
        name="AutoInt",
    ):
        super().__init__(name=name)
        
        if aggregation_mode not in __AGGREGATION_MODES:
            raise ValueError(f"'aggregation_mode' must be one of {__AGGREGATION_MODES}")

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding
        self.aggregation_mode = aggregation_mode

        # embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            input_length=dim_input,
            embeddings_regularizer=tf.keras.regularizers.l2(regularization),
            name="embedding",
        )
        
        # interaction layer using MLP
        self.interaction_mlp = MLP(
            num_hidden=num_hidden, 
            dim_hidden=dim_hidden, 
            dim_out=1 if aggregation_mode == "add" else None, 
            dropout=dropout
        )

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
            ))
        
        if aggregation_mode == "add":
            self.attn_projection_head = tf.keras.layers.Dense(1, name="attn_projection_head")
        elif aggregation_mode == "concat":
            self.projection_head = tf.keras.layers.Dense(1, name="projection_head")

        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=False):
        # extract embeddings
        embeddings = self.embedding(inputs, training=training) # (batch_size, nb_features, embedding_dim)

        # interaction layer using stack of self attention layers like Transformer's encoder
        # (batch_size, nb_features, embedding_dim)
        attn_out = embeddings
        for (interaction_layer, layer_norm) in self.interaction_attention:
            attn_out = interaction_layer(attn_out, attn_out, training=training) + attn_out
            attn_out = layer_norm(attn_out)

        attn_out = tf.reshape(attn_out, (-1, self.dim_input * self.dim_embedding))

        # interaction layer using MLP
        mlp_out = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))
        mlp_out = self.interaction_mlp(mlp_out, training=training)
        
        # combine the two representations
        if self.aggregation_mode == "add":
            attn_out = self.attn_projection_head(attn_out, training=training) # (batch_size, 1)
            logits = tf.add(attn_out, mlp_out) # (batch_size, 1)
        elif self.aggregation_mode == "concat":
            latent = tf.concat((attn_out, mlp_out), axis=-1)
            logits = self.projection_head(latent, training=training)
            
        outputs = tf.nn.sigmoid(logits) # (batch_size, 1)

        return outputs
