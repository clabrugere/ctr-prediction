import tensorflow as tf

from models.tensorflow.mlp import MLP


class DCN(tf.keras.Model):
    __CROSS_VARIANTS = ["cross", "cross_mix"]

    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        num_interaction=2,
        num_expert=1,
        dim_low=32,
        num_hidden=2,
        dim_hidden=16,
        dropout=0.0,
        parallel_mlp=True,
        cross_type="cross_mix",
        name="DCN",
    ):
        super().__init__(name=name)

        if cross_type not in self.__CROSS_VARIANTS:
            raise ValueError(f"'cross_layer' argument must be one of {self.__CROSS_VARIANTS}")

        self.parallel_mlp = parallel_mlp
        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        # embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            input_length=dim_input,
            name="embedding",
        )

        # cross layer
        self.interaction_cross = []
        if cross_type == "cross_mix":
            for _ in range(num_interaction):
                self.interaction_cross.append(
                    (CrossMixLayer(dim_low=dim_low, num_expert=num_expert), tf.keras.layers.Dropout(dropout))
                )
        else:
            for _ in range(num_interaction):
                self.interaction_cross.append((CrossLayer(), tf.keras.layers.Dropout(dropout)))

        # mlp
        self.interaction_mlp = MLP(num_hidden=num_hidden, dim_hidden=dim_hidden, dropout=dropout)

        # final projection head
        self.projection_head = tf.keras.layers.Dense(1, name="projection_head")

    def call(self, inputs, training=False):
        embeddings = self.embedding(inputs, training=training)  # (bs, dim_input, dim_emb)
        embeddings = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))  # (bs, dim_input * dim_emb)

        latent_cross = embeddings
        for interaction, dropout in self.interaction_cross:
            latent_cross = interaction(embeddings, latent_cross)  # (bs, dim_input * dim_emb)
            latent_cross = dropout(latent_cross, training=training)  # (bs, dim_input * dim_emb)

        if self.parallel_mlp:
            latent_mlp = self.interaction_mlp(embeddings, training=training)  # (bs, dim_hidden)
            latent = tf.concat((latent_cross, latent_mlp), axis=-1)  # (bs, dim_input * dim_emb + dim_hidden)
        else:
            latent_cross = tf.nn.relu(latent_cross)  # (bs, dim_input * dim_emb)
            latent = self.interaction_mlp(latent_cross, training=training)  # (bs, dim_hidden)

        logits = self.projection_head(latent, training=training)  # (bs, 1)
        outputs = tf.nn.sigmoid(logits)  # (bs, 1)

        return outputs


class CrossLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        weights_initializer="glorot_uniform",
        bias_initializer="zeros",
    ):
        super().__init__()

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        dim_input = tf.compat.dimension_value(input_shape[-1])

        self.W = self.add_weight(
            name="weights",
            shape=(dim_input, dim_input),
            initializer=self.weights_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(dim_input,),
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, x_0, x_l):
        return x_0 * (tf.matmul(x_l, self.W) + self.b) + x_l


class CrossMixLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_low,
        num_expert,
        activation="relu",
        weights_initializer="he_uniform",
        bias_initializer="zeros",
        gate_function="softmax",
    ):
        super().__init__()

        self.dim_low = dim_low
        self.num_experts = num_expert
        self.gate_function = gate_function
        self.activation = tf.keras.activations.get(activation)
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.dim_input = tf.compat.dimension_value(input_shape[-1])

        self.U = self.add_weight(
            name="U",
            shape=(self.dim_low * self.num_experts, self.dim_input * self.num_experts),
            initializer=self.weights_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.V = self.add_weight(
            name="V",
            shape=(self.dim_input, self.dim_low * self.num_experts),
            initializer=self.weights_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.C = self.add_weight(
            name="C",
            shape=(self.dim_low * self.num_experts, self.dim_low * self.num_experts),
            initializer=self.weights_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.dim_input * self.num_experts,),
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True,
        )

        self.gate = tf.keras.layers.Dense(self.num_experts, activation=self.gate_function, use_bias=False)
        self.built = True

    def call(self, x_0, x_l, training=None):
        out = self.activation(tf.matmul(x_l, self.V))  # (bs, dim_low * num_experts)
        out = self.activation(tf.matmul(out, self.C))  # (bs, dim_low * num_experts)
        out = tf.matmul(out, self.U) + self.b  # (bs, dim * num_experts)
        out = tf.reshape(out, (-1, self.dim_input, self.num_experts))  # (bs, dim, num_experts)
        out = tf.expand_dims(x_0, -1) * out  # (bs, dim, num_experts)

        gate_score = self.gate(x_0, training=training)  # (bs, num_experts)

        return tf.einsum("bde,be->bd", out, gate_score) + x_l  # (bs, dim)
