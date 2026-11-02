
from src.model import build_model


def test_model_output_has_three_classes():
    model = build_model()

    assert model.output_shape == (None, 3)


def test_model_has_trainable_parameters():
    model = build_model()

    assert model.trainable_params if hasattr(model, "trainable_params") else model.trainable_weights
