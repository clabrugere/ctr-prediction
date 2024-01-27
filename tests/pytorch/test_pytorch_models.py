import pytest
from torch import Tensor
from torch.nn import Module


@pytest.mark.parametrize(
    "model, input",
    [
        ("mlp_model", "sample_continuous_input"),
        ("deepwide_model", "sample_categorical_input"),
        ("dcn_model", "sample_categorical_input"),
        ("autoint_model", "sample_categorical_input"),
        ("final_mlp_model", "sample_categorical_input"),
    ],
    indirect=["model", "input"],
)
def test_model(model: Module, input: Tensor) -> None:
    output = model(input)

    assert output.shape == (input.shape[0], 1)
