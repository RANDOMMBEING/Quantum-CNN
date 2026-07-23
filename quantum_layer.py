import pennylane as qml
import torch
from torch import nn


# demo basic rakhne ke liye sirf 4 qubits use ho rahe hain
NUMBER_OF_QUBITS = 4

# default.qubit laptop ke CPU par quantum circuit simulate karta hai
device = qml.device("default.qubit", wires=NUMBER_OF_QUBITS)


def encode_features(features):
    # har image feature ko uske apne qubit par bhej rahe hain
    for wire in range(NUMBER_OF_QUBITS):
        # pi se multiply karke pixel value ko rotation angle banaya
        qml.RY(torch.pi * features[..., wire], wires=wire)


def quantum_convolution(weights):
    # ye angles normal neural network weights ki tarah learn honge
    for wire in range(NUMBER_OF_QUBITS):
        # RY aur RZ qubit ko do different directions mein rotate karte hain
        qml.RY(weights[wire, 0], wires=wire)
        qml.RZ(weights[wire, 1], wires=wire)

    # CNOT se paas wale qubits aapas mein information share karte hain
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])


def quantum_pooling():
    # sab qubits ki information ko qubit 0 par collect kar rahe hain
    qml.CNOT(wires=[3, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[1, 0])


# torch interface ki wajah se gradients PyTorch tak wapas ja sakte hain
@qml.qnode(device, interface="torch", diff_method="backprop")
def qcnn_qnode(inputs, weights):
    # circuit ka order simple hai: encode, convolution, phir pooling
    encode_features(inputs)
    quantum_convolution(weights)
    quantum_pooling()
    # end mein qubit 0 ko measure karke ek number milta hai
    return qml.expval(qml.PauliZ(0))


class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # TorchLayer circuit ko normal nn.Module jaisa use karne deta hai
        weight_shapes = {"weights": (NUMBER_OF_QUBITS, 2)}
        self.circuit = qml.qnn.TorchLayer(qcnn_qnode, weight_shapes)

    def forward(self, features):
        # poora batch ek saath quantum circuit ke andar jayega
        expectation_values = self.circuit(features)
        return expectation_values.reshape(-1, 1).to(dtype=features.dtype)
