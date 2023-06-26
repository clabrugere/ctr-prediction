import tensorflow as tf
from models.tensorflow.mlp import MLP


__CROSS_VARIANTS = ["cross", "cross_mix"]


class DCN(tf.keras.Model):
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

        if cross_type not in __CROSS_VARIANTS:
            raise ValueError(f"'cross_layer' argument must be one of {__CROSS_VARIANTS}")

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
                    (CrossLayerV2(dim_low=dim_low, num_expert=num_expert), tf.keras.layers.Dropout(dropout))
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

        super().build(input_shape)

    def call(self, x_0, x_l):
        return x_0 * (tf.matmul(x_l, self.W) + self.b) + x_l


class CrossLayerV2(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_low,
        num_expert=1,
        activation="relu",
        weights_initializer="glorot_uniform",
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
        dim_input = tf.compat.dimension_value(input_shape[-1])
        self.experts = []

        for i in range(self.num_experts):
            U = self.add_weight(
                name=f"U_{i}",
                shape=(dim_input, self.dim_low),
                initializer=self.weights_initializer,
                dtype=self.dtype,
                trainable=True,
            )
            V = self.add_weight(
                name=f"V_{i}",
                shape=(dim_input, self.dim_low),
                initializer=self.weights_initializer,
                dtype=self.dtype,
                trainable=True,
            )
            C = self.add_weight(
                name=f"C_{i}",
                shape=(self.dim_low, self.dim_low),
                initializer=self.weights_initializer,
                dtype=self.dtype,
                trainable=True,
            )
            bias = self.add_weight(
                name=f"bias_{i}",
                shape=(dim_input,),
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True,
            )
            self.experts.append((U, V, C, bias))

        self.gate = tf.keras.layers.Dense(
            self.num_experts,
            activation=self.gate_function,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Ones(),
        )

        self.built = True

    def call(self, x_0, x_l, training=None):
        # x_0 and x_l are of shape (batch_size, dim_input * dim_embedding = dim_last)
        expert_outputs = []
        for U, V, C, bias in self.experts:
            # project input in a low dimensional space and pass through non linearity
            low_rank_proj = self.activation(tf.matmul(x_l, V))  # (bs, dim_low)

            # project into an intermediate space with same dimension
            low_rank_inter = self.activation(tf.matmul(low_rank_proj, C))  # (bs, dim_low)

            # project back to initial space
            expert_output = x_0 * tf.matmul(low_rank_inter, U, transpose_b=True) + bias  # (bs, dim_last)

            expert_outputs.append(expert_output)

        # aggregate expert representations using gate score and add residual connection
        gate_score = self.gate(x_0, training=training)  # (bs, num_experts)
        expert_outputs = tf.stack(expert_outputs, axis=-1)
        outputs = tf.einsum("bie,be->bi", expert_outputs, gate_score) + x_l

        return outputs
