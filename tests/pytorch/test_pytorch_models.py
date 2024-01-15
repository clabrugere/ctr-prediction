from torch import Tensor

from models.pytorch import DCN, MLP, AutoInt, DeepWide, FinalMLP


def test_mlp(mlp_model: MLP, sample_continuous_input: Tensor) -> None:
    n, _ = sample_continuous_input.shape
    output = mlp_model(sample_continuous_input)

    assert output.shape == (n, 1)


def test_deepwide(deepwide_model: DeepWide, sample_categorical_input: Tensor) -> None:
    n, _ = sample_categorical_input.shape
    output = deepwide_model(sample_categorical_input)

    assert output.shape == (n, 1)


def test_dcn(dcn_model: DCN, sample_categorical_input: Tensor) -> None:
    n, _ = sample_categorical_input.shape
    output = dcn_model(sample_categorical_input)

    assert output.shape == (n, 1)


def test_autoint(autoint_model: AutoInt, sample_categorical_input: Tensor) -> None:
    n, _ = sample_categorical_input.shape
    output = autoint_model(sample_categorical_input)

    assert output.shape == (n, 1)


def test_final_mlp(final_mlp_model: FinalMLP, sample_categorical_input: Tensor) -> None:
    n, _ = sample_categorical_input.shape
    output = final_mlp_model(sample_categorical_input)

    assert output.shape == (n, 1)
