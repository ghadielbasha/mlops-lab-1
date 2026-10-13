import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "food11"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Food-11 ResNet18 training with MLflow"
    )

    parser.add_argument(
        "--dataset",
        choices=["mini", "processed"],
        default="mini",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)

    return parser.parse_args()


def load_data(dataset_name, batch_size):
    folder_name = (
        "food11_processed_mini"
        if dataset_name == "mini"
        else "food11_processed"
    )

    dataset_path = DATA_ROOT / folder_name

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(
        dataset_path / "training",
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        dataset_path / "validation",
        transform=transform,
    )

    test_dataset = datasets.ImageFolder(
        dataset_path / "evaluation",
        transform=transform,
    )

    if not (
        train_dataset.classes
        == val_dataset.classes
        == test_dataset.classes
    ):
        raise ValueError("Dataset classes do not match across splits.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Dataset:", dataset_name)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))
    print("Classes:", train_dataset.classes)

    return train_loader, val_loader, test_loader, len(train_dataset.classes)


def create_model(num_classes, device):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze pretrained layers for faster CPU training.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the original 1000-class classifier.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    args = parse_args()

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_loader, val_loader, test_loader, num_classes = load_data(
        args.dataset,
        args.batch_size,
    )

    model = create_model(num_classes, device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=args.lr,
    )

    with mlflow.start_run() as run:
        mlflow.log_params({
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "architecture": "resnet18",
            "pretrained": True,
            "trainable_layers": "fc",
        })

        print("MLflow run ID:", run.info.run_id)

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            mlflow.log_metric(
                "train_loss", train_loss, step=epoch
            )
            mlflow.log_metric(
                "val_loss", val_loss, step=epoch
            )
            mlflow.log_metric(
                "val_accuracy", val_accuracy, step=epoch
            )

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.4f}"
            )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )

        print("\nTraining completed!")
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print("MLflow Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()

