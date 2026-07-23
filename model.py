from torch import nn

from quantum_layer import QuantumLayer


class QuantumCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # pehle 4 features quantum layer mein jayenge
        self.quantum_layer = QuantumLayer()
        # phir ek normal linear layer final answer nikalegi
        self.classifier = nn.Linear(in_features=1, out_features=1)

    def forward(self, features):
        # bas quantum output ko classical layer se pass karna hai
        quantum_output = self.quantum_layer(features)
        return self.classifier(quantum_output)
