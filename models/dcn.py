import tensorflow as tf
from models.mlp import MLP


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
        regularization=1e-5,
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
            embeddings_regularizer=tf.keras.regularizers.l2(regularization),
            name="embedding",
        )

        # interaction layer
        self.interaction_cross = []
        if cross_type == "cross_mix":
            for _ in range(num_interaction):
                self.interaction_cross.append(CrossLayerV2(dim_low=dim_low, num_expert=num_expert))
        else:
            for _ in range(num_interaction):
                self.interaction_cross.append(CrossLayer())

        # interaction layer using MLP
        self.interaction_mlp = MLP(num_hidden=num_hidden, dim_hidden=dim_hidden, dropout=dropout)

        # final projection head
        self.projection_head = tf.keras.layers.Dense(1, name="projection_head")

        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=False):
        # (batch_size, dim_input, embedding_dim)
        embeddings = self.embedding(inputs, training=training)

        # (batch_size, dim_input * dim_embedding)
        embeddings = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))

        interaction_out = embeddings
        for interaction in self.interaction_cross:
            interaction_out = interaction(embeddings, interaction_out)  # (batch_size, dim_input * dim_embedding)

        if self.parallel_mlp:
            latent_mlp = self.interaction_mlp(embeddings, training=training)  # (batch_size, dim_hidden)
            # (batch_size, dim_input * dim_embedding + dim_hidden)
            latent = tf.concat((interaction_out, latent_mlp), axis=-1)
        else:
            interaction_out = tf.nn.relu(interaction_out)  # (batch_size, dim_input * dim_embedding)
            latent = self.interaction_mlp(interaction_out, training=training)  # (batch_size, dim_hidden)

        logits = self.projection_head(latent, training=training)  # (batch_size, 1)
        outputs = tf.nn.sigmoid(logits)  # (batch_size, 1)

        return outputs


class CrossLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        weights_initializer="glorot_uniform",
        weights_regularizer=None,
        weights_contraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
    ):
        super().__init__()

        self.weights_initializer = weights_initializer
        self.weights_regularizer = weights_regularizer
        self.weights_contraint = weights_contraint

        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        dim_last = tf.compat.dimension_value(input_shape[-1])

        self.W = self.add_weight(
            name="weights",
            shape=(dim_last, dim_last),
            initializer=self.weights_initializer,
            regularizer=self.weights_regularizer,
            constraint=self.weights_contraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(dim_last,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        self.built = True

    def call(self, x_0, x_l):
        return x_0 * (tf.matmul(x_l, self.W) + self.b) + x_l


class CrossLayerV2(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_low,
        num_expert=1,
        activation="relu",
        weights_initializer="glorot_uniform",
        weights_regularizer=None,
        weights_contraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
    ):
        super().__init__()

        self.dim_low = dim_low
        self.num_experts = num_expert
        self.activation = tf.keras.activations.get(activation)

        self.weights_initializer = weights_initializer
        self.weights_regularizer = weights_regularizer
        self.weights_contraint = weights_contraint

        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        dim_last = tf.compat.dimension_value(input_shape[-1])
        self.experts = []

        for i in range(self.num_experts):
            U = self.add_weight(
                name=f"U_{i}",
                shape=(dim_last, self.dim_low),
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                constraint=self.weights_contraint,
                dtype=self.dtype,
                trainable=True,
            )
            V = self.add_weight(
                name=f"V_{i}",
                shape=(dim_last, self.dim_low),
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                constraint=self.weights_contraint,
                dtype=self.dtype,
                trainable=True,
            )
            C = self.add_weight(
                name=f"C_{i}",
                shape=(self.dim_low, self.dim_low),
                initializer=self.weights_initializer,
                regularizer=self.weights_regularizer,
                constraint=self.weights_contraint,
                dtype=self.dtype,
                trainable=True,
            )
            bias = self.add_weight(
                name=f"bias_{i}",
                shape=(dim_last,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
            gate_projection = tf.keras.layers.Dense(1, use_bias=False)

            self.experts.append((U, V, C, bias, gate_projection))

        self.built = True

    def call(self, x_0, x_l):
        # x_0 and x_l are of shape (batch_size, dim_input * dim_embedding = dim_last)
        expert_outputs = []
        for U, V, C, bias, gate_projection in self.experts:
            # project input in a low dimensional space and pass through non linearity
            # a(x_l @ V)) -> (batch_size, dim_low)
            low_rank_proj = self.activation(tf.matmul(x_l, V))

            # project into an intermediate space with same dimension
            # a( a(x_l @ V) @ C ) -> (batch_size, dim_low)
            low_rank_inter = self.activation(tf.matmul(low_rank_proj, C))

            # project back to initial space
            # E(x_0, x_l) = x_0 * ( a( a(x_l @ V) @ C ) @ U + b )  -> (batch_size, dim_last)
            expert_output = x_0 * tf.matmul(low_rank_inter, U, transpose_b=True) + bias

            # weight the expert representation
            # G(x_l) * E(x_0, x_l) -> (batch_size, dim_last)
            gate_score = tf.nn.sigmoid(gate_projection(x_l))
            expert_output = gate_score * expert_output

            expert_outputs.append(expert_output)

        # sum representations and add residual connection
        outputs = tf.add_n(expert_outputs) + x_l

        return outputs
