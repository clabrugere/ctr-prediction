import tensorflow as tf


class DCN(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        num_interaction=2,
        num_hidden=2,
        dim_hidden=16,
        regularization=0.00001,
        dropout=0.0,
        parallel_mlp=True,
        name="DCN",
    ):
        super().__init__(name=f"{name}_parallel" if parallel_mlp else name)

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

        # MLP
        self.interaction_mlp = tf.keras.Sequential(name="MLP")
        for _ in range(num_hidden):
            self.interaction_mlp.add(tf.keras.layers.Dense(dim_hidden))
            self.interaction_mlp.add(tf.keras.layers.BatchNormalization())
            self.interaction_mlp.add(tf.keras.layers.ReLU())
            self.interaction_mlp.add(tf.keras.layers.Dropout(dropout))

        # interaction layer
        self.interaction_cross = []
        for _ in range(num_interaction):
            self.interaction_cross.append(
                #tf.keras.layers.Dense(dim_input * dim_embedding, name=f"interaction_layer_{i+1}")
                CrossLayer()
            )

        # final projection head
        self.projection_head = tf.keras.layers.Dense(1, name="projection_head")

        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=False):
        # (batch_size, dim_input, embedding_dim)
        embeddings = self.embedding(inputs, training=training)

        # (batch_size, dim_input * dim_embedding)
        embeddings = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))

        # interaction is defined as x_0 * Dense(x) + x
        interaction_in = embeddings
        for interaction in self.interaction_cross:
            # (batch_size, dim_input * dim_embedding)
            interaction_out = interaction(embeddings, interaction_in)
            interaction_in = interaction_out

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
        outputs = x_0 * (tf.matmul(x_l, self.W) + self.b) + x_l

        return outputs
