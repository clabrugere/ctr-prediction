import pytest
import torch

from models.pytorch import DCN, MLP, AutoInt, DeepWide, FinalMLP

SIZE = 320
NUM_FEATURES = 10
NUM_EMBEDDING = 100


@pytest.fixture(scope="module")
def sample_categorical_input():
    return torch.multinomial(
        torch.rand((SIZE, NUM_EMBEDDING)),
        NUM_FEATURES,
        replacement=True,
    )


@pytest.fixture(scope="module")
def sample_continuous_input():
    return torch.rand(SIZE, NUM_FEATURES, dtype=torch.float32)


@pytest.fixture
def mlp_model() -> MLP:
    return MLP(dim_in=NUM_FEATURES, num_hidden=3, dim_hidden=16, dim_out=1)


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
