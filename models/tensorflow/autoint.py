import tensorflow as tf
from models.tensorflow.mlp import MLP


__AGGREGATION_MODES = ["add", "concat"]


class AttentionInteraction(tf.keras.Model):
    def __init__(self, num_layer, num_heads, dim_key, dropout=0.0):
        self.layers = []
        for i in range(num_layer):
            self.layers.append(
                (
                    tf.keras.layers.MultiHeadAttention(
                        num_heads=num_heads,
                        key_dim=dim_key,
                        dropout=dropout,
                        name=f"interaction_layer_{i+1}",
                    ),
                    tf.keras.layers.LayerNormalization(),
                )
            )

    def call(self, inputs, training=False):
        attn_out = inputs  # (bs, dim_input, dim_emb)
        for mha_layer, layer_norm in self.layers:
            attn_out = mha_layer(attn_out, attn_out, training=training) + attn_out
            attn_out = layer_norm(attn_out)  # (bs, dim_input, dim_emb)

        return attn_out


class AutoInt(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
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
            name="embedding",
        )

        # interaction layer using MLP
        self.interaction_mlp = MLP(
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=1 if aggregation_mode == "add" else None,
            dropout=dropout,
        )

        # interaction layer using stacked self-attention
        self.interaction_attention = AttentionInteraction(
            num_attention, num_heads=num_heads, dim_key=dim_key, dropout=dropout
        )

        if aggregation_mode == "add":
            self.attn_projection_head = tf.keras.layers.Dense(1, name="attn_projection_head")
        elif aggregation_mode == "concat":
            self.projection_head = tf.keras.layers.Dense(1, name="projection_head")

    def call(self, inputs, training=False):
        embeddings = self.embedding(inputs, training=training)  # (bs, dim_input, dim_emb)

        attn_out = self.interaction_attention(embeddings, training=training)
        attn_out = tf.reshape(attn_out, (-1, self.dim_input * self.dim_embedding))  # (bs, dim_input * dim_emb)

        mlp_out = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))  # (bs, dim_input, dim_emb)
        mlp_out = self.interaction_mlp(mlp_out, training=training)

        # combine the two representations
        if self.aggregation_mode == "add":
            attn_out = self.attn_projection_head(attn_out, training=training)  # (bs, 1)
            logits = tf.add(attn_out, mlp_out)  # (bs, 1)
        elif self.aggregation_mode == "concat":
            latent = tf.concat((attn_out, mlp_out), axis=-1)  # (bs, dim_input * dim_emb + dim_hidden)
            logits = self.projection_head(latent, training=training)  # (bs, 1)

        outputs = tf.nn.sigmoid(logits)  # (bs, 1)

        return outputs
