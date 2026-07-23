# Product Requirements: QCNN for Binary MNIST

## Product overview

**Product:** Quantum Convolutional Neural Network for Binary MNIST
Classification.

The product is a hybrid PennyLane/PyTorch learning example that classifies
MNIST digits 0 and 1. It runs a small four-qubit QCNN on PennyLane's local CPU
simulator and exposes it as a standard PyTorch module.

## Objectives and success measures

- Provide an end-to-end data, training, evaluation, and logging pipeline.
- Demonstrate a PennyLane QNode using `interface="torch"` and `TorchLayer`.
- Run with one command: `python train_qcnn.py`.
- Target at least 85-90% held-out binary MNIST accuracy.
- Cover feature preprocessing, model forward behavior, and quantum gradient
  flow with automated tests.

## Scope

Included: MNIST digits 0 and 1, four quadrant features, four simulated qubits,
PyTorch optimization, full binary test evaluation, stdout logging, CSV logs,
saved weights, and beginner documentation.

Excluded from this version: ten-class MNIST, real quantum hardware, large
hyperparameter searches, production deployment, and claims of quantum
advantage.

## Functional requirements

1. Download MNIST with `torchvision.datasets.MNIST` and `ToTensor`.
2. Keep only labels 0 and 1 and convert every image into four quadrant means.
3. Encode features with `RY(pi * feature)` on four wires.
4. Apply trainable `RY`/`RZ` gates and nearest-neighbour CNOT convolution.
5. Pool with CNOTs from wires 3, 2, and 1 to wire 0.
6. Return `PauliZ(0)` and wrap the QNode in `qml.qnn.TorchLayer`.
7. Map the quantum scalar to one logit with `nn.Linear(1, 1)`.
8. Train with Adam and `BCEWithLogitsLoss`; report loss and accuracy.
9. Configure batch size, epochs, learning rate, seed, and subset sizes by CLI.
10. Save epoch metrics as CSV and final model parameters as a PyTorch file.

## Non-functional requirements

- Use a small number of clearly named Python modules and straightforward code.
- Run on a CPU laptop with four simulated qubits and bounded training subsets.
- Keep the quantum circuit isolated so its encoding, gates, or wire count can
  be changed later.
- Seed Python, NumPy, PyTorch, and DataLoader randomness for reproducibility.

## Risks

Quantum simulation is slower than an ordinary neural layer and accuracy can
depend on circuit initialization and subset size. The product is a learning
and experimentation artifact; comparable or better classical performance is
expected and does not invalidate the integration demonstration.
