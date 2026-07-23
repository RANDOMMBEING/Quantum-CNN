import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def extract_quadrant_features(images):
    # image ko seedha 4 equal parts mein divide kar rahe hain
    middle_row = images.shape[2] // 2
    middle_column = images.shape[3] // 2

    # har quadrant ki average brightness ek feature banegi
    top_left = images[:, :, :middle_row, :middle_column].mean(dim=(1, 2, 3))
    top_right = images[:, :, :middle_row, middle_column:].mean(dim=(1, 2, 3))
    bottom_left = images[:, :, middle_row:, :middle_column].mean(dim=(1, 2, 3))
    bottom_right = images[:, :, middle_row:, middle_column:].mean(dim=(1, 2, 3))

    # 4 values ko side by side rakh kar final input bana diya
    four_features = [top_left, top_right, bottom_left, bottom_right]
    return torch.stack(four_features, dim=1)


def make_binary_dataset(source_dataset, limit=None, seed=42):
    # labels mein se bas digit 0 aur 1 ko select karna hai
    targets = torch.as_tensor(source_dataset.targets)
    binary_indices = torch.where((targets == 0) | (targets == 1))[0]

    # chhota subset lene se quantum training ka demo jaldi chalega
    if limit:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(binary_indices), generator=generator)
        binary_indices = binary_indices[order[:limit]]

    # selected images aur labels ko pehle simple lists mein collect kar rahe hain
    images = []
    labels = []
    for index in binary_indices.tolist():
        image, label = source_dataset[index]
        images.append(image)
        labels.append(float(label))

    # lists ko tensors mein badal kar PyTorch dataset ready ho jayega
    image_tensor = torch.stack(images)
    feature_tensor = extract_quadrant_features(image_tensor)
    label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return TensorDataset(feature_tensor, label_tensor)


def create_data_loaders(
    data_dir="data",
    batch_size=128,
    train_size=2000,
    test_size=None,
    seed=42,
    download=True,
):
    # download=True se MNIST first run par automatically download ho jayega
    to_tensor = transforms.ToTensor()
    train_mnist = datasets.MNIST(
        root=data_dir, train=True, transform=to_tensor, download=download
    )
    test_mnist = datasets.MNIST(
        root=data_dir, train=False, transform=to_tensor, download=download
    )

    train_dataset = make_binary_dataset(train_mnist, train_size, seed)
    test_dataset = make_binary_dataset(test_mnist, test_size, seed)

    # same seed rakhenge to shuffled batches bhi same order mein milenge
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader
