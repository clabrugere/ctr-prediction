import pytest
from tensorflow import Tensor
from tensorflow.python.keras import Model


@pytest.mark.parametrize(
    "model, input",
    [
        ("mlp_model", "sample_continuous_input"),
        ("deepwide_model", "sample_categorical_input"),
        ("dcn_model", "sample_categorical_input"),
        ("autoint_model", "sample_categorical_input"),
        ("final_mlp_model", "sample_categorical_input"),
        ("gdcnp_model", "sample_categorical_input"),
        ("gdcns_model", "sample_categorical_input"),
    ],
    indirect=["model", "input"],
)
def test_model(model: Model, input: Tensor) -> None:
    output = model(input)

    assert output.shape == (input.shape[0], 1)
