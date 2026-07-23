import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import create_data_loaders
from model import QuantumCNN


def set_random_seed(seed):
    # same seed se har run ka result almost repeatable rahega
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_correct_predictions(logits, labels):
    # sigmoid ke baad 0.5 se bada matlab digit 1
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    return int((predictions == labels).sum().item())


def train_one_epoch(model, data_loader, optimizer, loss_function):
    # model ko bataya ki ab training start ho rahi hai
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in data_loader:
        # har batch se pehle purane gradients zero karne padte hain
        optimizer.zero_grad()
        # forward pass ke baad prediction aur actual label ka loss niklega
        logits = model(features)
        loss = loss_function(logits, labels)
        # backward aur step milkar model ke weights update karte hain
        loss.backward()
        optimizer.step()

        # poore epoch ka average nikalne ke liye totals add kar rahe hain
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += count_correct_predictions(logits, labels)
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


# test ke time gradients aur weight updates ki zarurat nahi hoti
@torch.no_grad()
def evaluate(model, data_loader, loss_function):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in data_loader:
        logits = model(features)
        loss = loss_function(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += count_correct_predictions(logits, labels)
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def write_log(log_path, rows):
    # results ko simple CSV mein rakhna baad mein compare karne mein useful hai
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser():
    # common settings command line se change ho sakti hain
    parser = argparse.ArgumentParser(
        description="Train a four-qubit QCNN on MNIST digits 0 and 1."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-size",
        type=int,
        default=2000,
        help="Number of binary training images. Use 0 for all images.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=0,
        help="Number of binary test images. Use 0 for the full test split.",
    )
    parser.add_argument("--log-csv", default="results/training_log.csv")
    parser.add_argument("--model-path", default="results/qcnn_model.pt")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download MNIST; only use files already in --data-dir.",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()

    print("\nBeginner QCNN: MNIST digit 0 versus digit 1")
    print("This uses four simulated qubits on the CPU.")

    print("\nStep 1/4: Preparing the MNIST data...")
    set_random_seed(args.seed)

    # size 0 ka simple meaning hai ki poora dataset use karo
    train_size = args.train_size if args.train_size > 0 else None
    test_size = args.test_size if args.test_size > 0 else None
    train_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_size=train_size,
        test_size=test_size,
        seed=args.seed,
        download=not args.no_download,
    )
    print("Data preparation is complete.")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Test images:     {len(test_loader.dataset)}")

    print("\nStep 2/4: Creating the quantum-classical model...")
    # loss aur optimizer bilkul normal PyTorch model wale hi hain
    model = QuantumCNN()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"Trainable values: {sum(p.numel() for p in model.parameters())}")

    print("\nStep 3/4: Starting training...")
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}: training the model...")
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, loss_function
        )
        print("Training pass complete. Checking the test images...")
        test_loss, test_accuracy = evaluate(model, test_loader, loss_function)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        history.append(row)
        print(
            f"Epoch {epoch} result -> loss: {train_loss:.4f}, "
            f"train accuracy: {train_accuracy:.1%}, "
            f"test accuracy: {test_accuracy:.1%}\n"
        )

    print("Step 4/4: Training is complete. Saving the results...")
    # last mein log aur trained weights dono save kar dete hain
    write_log(Path(args.log_csv), history)
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    print(f"Training log saved to: {args.log_csv}")
    print(f"Model weights saved to: {args.model_path}")
    print(f"Done. Final test accuracy: {history[-1]['test_accuracy']:.1%}\n")


if __name__ == "__main__":
    main()
