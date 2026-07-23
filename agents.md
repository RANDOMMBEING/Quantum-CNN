# Suggested AI/LLM Agent Roles

This small project does not require multiple agents to run. These roles are a
documentation aid for future collaborative or hackathon work.

## Data agent

Owns MNIST filtering, feature extraction, deterministic sampling, and tests in
`data.py` and `tests/test_data.py`. It should preserve the `[batch, 4]` feature
contract expected by the quantum circuit.

## Quantum-circuit agent

Owns `quantum_layer.py`. It may experiment with encodings, convolution gates,
pooling, or more qubits while preserving a normal `nn.Module` interface and
verifying that PyTorch gradients reach quantum parameters.

## Training agent

Owns `model.py`, `train.py`, and `train_qcnn.py`. It monitors loss, accuracy,
runtime, reproducibility, CSV logging, and saved model compatibility.

## Testing and documentation agent

Runs the complete test suite and a small end-to-end MNIST experiment. It keeps
`README.md` and `PRD.md` aligned with behavior actually verified in the code.

## Collaboration rule

Before changing a module boundary, record its input and output shapes, update
the related tests, and run `python -m pytest -q`. Experimental accuracy should
always include the seed, training size, test size, epoch count, and learning
rate so another developer can reproduce it.
