import pytest
import tensorflow as tf

from models.tensorflow import DCN, GDCNP, GDCNS, MLP, AutoInt, DeepWide, FinalMLP

SIZE = 320
NUM_FEATURES = 10
NUM_EMBEDDING = 100


@pytest.fixture(scope="module")
def binary_classification_data() -> tf.data.Dataset:
    n, m = 320, 10
    ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.random.categorical(tf.math.log([[1 / 10 for _ in range(m)]] * n), m),
                tf.cast(tf.random.uniform(shape=(n, 1)) > 0.5, tf.int32),
            )
        )
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


@pytest.fixture(scope="module")
def sample_categorical_input():
    return tf.random.categorical(
        tf.random.uniform((SIZE, NUM_EMBEDDING), minval=0, maxval=1), NUM_FEATURES, dtype=tf.int32
    )


@pytest.fixture(scope="module")
def sample_continuous_input():
    return tf.random.uniform(shape=(SIZE, NUM_FEATURES), dtype=tf.float32)


@pytest.fixture
def mlp_model() -> MLP:
    return MLP(num_hidden=3, dim_hidden=16, dim_out=1)


@pytest.fixture
def deepwide_model() -> DeepWide:
    return DeepWide(dim_input=NUM_FEATURES, num_embedding=NUM_EMBEDDING, dim_embedding=8, num_hidden=3, dim_hidden=16)


@pytest.fixture
def dcn_model() -> DCN:
    return DCN(
        dim_input=NUM_FEATURES,
        num_embedding=NUM_EMBEDDING,
        dim_embedding=8,
        num_interaction=2,
        num_expert=2,
        dim_low=8,
        num_hidden=3,
        dim_hidden=16,
    )


@pytest.fixture
def autoint_model() -> AutoInt:
    return AutoInt(
        dim_input=NUM_FEATURES,
        num_embedding=NUM_EMBEDDING,
        dim_embedding=8,
        num_attention=2,
        dim_key=8,
        num_heads=2,
        num_hidden=3,
        dim_hidden=16,
    )


@pytest.fixture
def final_mlp_model() -> FinalMLP:
    return FinalMLP(
        dim_input=NUM_FEATURES,
        num_embedding=NUM_EMBEDDING,
        dim_embedding=8,
        dim_hidden_fs=8,
        num_hidden_1=3,
        dim_hidden_1=16,
        num_hidden_2=3,
        dim_hidden_2=16,
        num_heads=2,
    )


@pytest.fixture
def gdcnp_model() -> GDCNP:
    return GDCNP(
        dim_input=NUM_FEATURES, num_embedding=NUM_EMBEDDING, dim_embedding=8, num_cross=2, num_hidden=3, dim_hidden=16
    )


@pytest.fixture
def gdcns_model() -> GDCNS:
    return GDCNS(
        dim_input=NUM_FEATURES, num_embedding=NUM_EMBEDDING, dim_embedding=8, num_cross=2, num_hidden=3, dim_hidden=16
    )
