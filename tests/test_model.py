import torch
from torch import nn

from model import QuantumCNN


def test_model_forward_shape_and_range():
    torch.manual_seed(42)
    model = QuantumCNN()
    features = torch.rand(3, 4)

    logits = model(features)

    assert logits.shape == (3, 1)
    assert torch.isfinite(logits).all()


def test_gradient_reaches_quantum_parameters():
    torch.manual_seed(42)
    model = QuantumCNN()
    features = torch.rand(2, 4)
    labels = torch.tensor([[0.0], [1.0]])

    loss = nn.BCEWithLogitsLoss()(model(features), labels)
    loss.backward()

    quantum_parameters = list(model.quantum_layer.parameters())
    assert quantum_parameters
    assert all(parameter.grad is not None for parameter in quantum_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in quantum_parameters)
