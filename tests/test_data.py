import torch

from data import extract_quadrant_features, make_binary_dataset


# fake data se tests fast chalenge aur MNIST download bhi nahi karna padega
class TinyMNIST:
    def __init__(self):
        self.targets = torch.tensor([0, 2, 1])
        self.images = [
            torch.zeros(1, 4, 4),
            torch.full((1, 4, 4), 0.5),
            torch.ones(1, 4, 4),
        ]

    def __getitem__(self, index):
        return self.images[index], int(self.targets[index])

    def __len__(self):
        return len(self.targets)


def test_extract_quadrant_features():
    image = torch.zeros(1, 1, 4, 4)
    image[:, :, :2, :2] = 0.1
    image[:, :, :2, 2:] = 0.2
    image[:, :, 2:, :2] = 0.3
    image[:, :, 2:, 2:] = 0.4

    features = extract_quadrant_features(image)

    assert features.shape == (1, 4)
    assert torch.allclose(features, torch.tensor([[0.1, 0.2, 0.3, 0.4]]))


def test_make_binary_dataset_filters_other_digits():
    dataset = make_binary_dataset(TinyMNIST())
    features, labels = dataset.tensors

    assert features.shape == (2, 4)
    assert labels.shape == (2, 1)
    assert labels.flatten().tolist() == [0.0, 1.0]
