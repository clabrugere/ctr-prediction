import tensorflow as tf
from keras import Model, Sequential, activations
from keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention

from models.tensorflow.mlp import MLP


class AttentionInteraction(Model):
    def __init__(self, num_heads, dim_key, dropout, name="AttentionInteraction"):
        super().__init__(name=name)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=dim_key, dropout=dropout)
        self.norm = LayerNormalization()

    def call(self, inputs, training=None):
        return self.norm(self.mha(inputs, inputs, training=training) + inputs)


class AttentionInteractionBlock(Sequential):
    def __init__(self, num_layer, num_heads, dim_key, dropout):
        super().__init__(
            [
                AttentionInteraction(num_heads, dim_key, dropout=dropout, name=f"attn_interaction_{i}")
                for i in range(num_layer)
            ]
        )


class AutoInt(Model):
    __AGGREGATION_MODES = ["add", "concat"]

    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding,
        num_heads,
        dim_key,
        num_hidden,
        dim_hidden,
        num_attention,
        dropout=0.0,
        aggregation_mode="add",
        name="AutoInt",
    ):
        super().__init__(name=name)

        if aggregation_mode not in self.__AGGREGATION_MODES:
            raise ValueError(f"'aggregation_mode' must be one of {self.__AGGREGATION_MODES}")

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding
        self.aggregation_mode = aggregation_mode

        # embedding layer
        self.embedding = Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
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
        self.interaction_attention = AttentionInteractionBlock(
            num_layer=num_attention, num_heads=num_heads, dim_key=dim_key, dropout=dropout
        )

        if aggregation_mode == "add":
            self.attn_projection_head = Dense(1, name="attn_projection_head")
        elif aggregation_mode == "concat":
            self.projection_head = Dense(1, name="projection_head")

    def call(self, inputs, training=None):
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

        outputs = activations.sigmoid(logits)  # (bs, 1)

        return outputs
