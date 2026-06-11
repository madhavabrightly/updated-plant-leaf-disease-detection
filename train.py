#!/usr/bin/env python3
"""Standalone PlantVillage trainer for AMD ROCm, NVIDIA CUDA, or CPU."""

import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def ensure_dependency(import_name, package_name):
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing missing dependency: {package_name}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", package_name]
        )


# Keep the cloud platform's existing torch build: it contains ROCm support.
ensure_dependency("kagglehub", "kagglehub")
ensure_dependency("sklearn", "scikit-learn")

import kagglehub
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
try:
    from torchvision import datasets, models, transforms
except (ImportError, RuntimeError) as exc:
    raise RuntimeError(
        "A torchvision build compatible with the platform's existing ROCm "
        "PyTorch is required. Do not install a generic CPU torchvision wheel."
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
DEFAULT_DATASET = "abdallahalidev/plantvillage-dataset"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 on PlantVillage using PyTorch/ROCm"
    )
    parser.add_argument("--data", default=None, help="Optional local dataset path")
    parser.add_argument("--kaggle-dataset", default=DEFAULT_DATASET)
    parser.add_argument("--epochs1", type=int, default=10)
    parser.add_argument("--epochs2", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr1", type=float, default=1e-3)
    parser.add_argument("--lr2", type=float, default=1e-4)
    parser.add_argument("--fine-tune-at", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output", default="model.pt")

    # Jupyter injects "-f <kernel.json>"; remove only that pair.
    cli_args = sys.argv[1:]
    filtered = []
    index = 0
    while index < len(cli_args):
        if cli_args[index] == "-f" and index + 1 < len(cli_args):
            index += 2
        else:
            filtered.append(cli_args[index])
            index += 1
    args = parser.parse_args(filtered)
    if args.epochs1 < 1 or args.epochs2 < 1:
        parser.error("--epochs1 and --epochs2 must both be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be between 0 and 1")
    if args.validation_size <= 0 or args.test_size <= 0:
        parser.error("--validation-size and --test-size must be greater than 0")
    if args.validation_size + args.test_size >= 0.5:
        parser.error("--validation-size + --test-size must be less than 0.5")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(require_gpu):
    if torch.cuda.is_available():
        # PyTorch intentionally uses the "cuda" API name for both ROCm and CUDA.
        device = torch.device("cuda")
        backend = "AMD ROCm" if torch.version.hip else "NVIDIA CUDA"
        print(f"GPU backend: {backend}")
        print(f"GPU device:  {torch.cuda.get_device_name(0)}")
        print(f"PyTorch:     {torch.__version__}")
        return device

    if require_gpu:
        raise RuntimeError("No PyTorch GPU is visible, and --require-gpu was requested.")

    print("WARNING: No GPU detected; training will run on CPU.")
    return torch.device("cpu")


def find_class_directory(dataset_root):
    root = Path(dataset_root).expanduser().resolve()
    candidates = []

    for current, dirs, _ in os.walk(root):
        current_path = Path(current)
        class_count = 0
        for dirname in dirs:
            class_path = current_path / dirname
            try:
                if any(
                    item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
                    for item in class_path.iterdir()
                ):
                    class_count += 1
            except OSError:
                pass

        if class_count >= 2:
            color_priority = 0 if current_path.name.lower() == "color" else 1
            candidates.append(
                (color_priority, -class_count, len(current_path.parts), current_path)
            )

    if not candidates:
        raise FileNotFoundError(f"No class-folder dataset found under: {root}")

    candidates.sort(key=lambda item: item[:3])
    return str(candidates[0][3])


def resolve_data_directory(data, kaggle_dataset):
    if data:
        root = data
        print(f"Using local dataset: {root}")
    else:
        print(f"Downloading/caching Kaggle dataset: {kaggle_dataset}")
        root = kagglehub.dataset_download(kaggle_dataset)
        print(f"Kaggle cache: {root}")

    class_directory = find_class_directory(root)
    print(f"Class directory: {class_directory}")
    return class_directory


def create_loaders(
    data_dir, batch_size, workers, seed, device, validation_size, test_size
):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            # RandomPerspective calls torch.linalg.lstsq and fails on some
            # ROCm cloud images whose CPU PyTorch build omits LAPACK.
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                shear=(-8, 8),
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), value='random'),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    base_dataset = datasets.ImageFolder(data_dir)
    indices = np.arange(len(base_dataset))
    train_indices, held_out_indices = train_test_split(
        indices,
        test_size=validation_size + test_size,
        random_state=seed,
        stratify=base_dataset.targets,
    )
    validation_fraction = validation_size / (validation_size + test_size)
    validation_indices, test_indices = train_test_split(
        held_out_indices,
        test_size=1 - validation_fraction,
        random_state=seed,
        stratify=np.asarray(base_dataset.targets)[held_out_indices],
    )

    train_dataset = Subset(datasets.ImageFolder(data_dir, train_transform), train_indices)
    validation_dataset = Subset(
        datasets.ImageFolder(data_dir, validation_transform), validation_indices
    )
    test_dataset = Subset(
        datasets.ImageFolder(data_dir, validation_transform), test_indices
    )
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    return (
        train_loader,
        validation_loader,
        test_loader,
        base_dataset.classes,
        base_dataset.targets,
    )


def build_model(num_classes, pretrained):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(input_features, num_classes),
    )
    return model


def set_feature_extraction(model):
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def set_fine_tuning(model, fine_tune_at):
    blocks = list(model.features.children())
    fine_tune_at = max(0, min(fine_tune_at, len(blocks) - 1))
    for block_index, block in enumerate(blocks):
        requires_grad = block_index >= fine_tune_at
        for parameter in block.parameters():
            parameter.requires_grad = requires_grad
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    print(f"Fine-tuning feature blocks {fine_tune_at} through {len(blocks) - 1}")


def make_class_weights(targets, train_indices, num_classes, device):
    counts = np.bincount(np.asarray(targets)[train_indices], minlength=num_classes)
    weights = len(train_indices) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    if training:
        # Frozen blocks must stay in evaluation mode so BatchNorm running
        # statistics are not modified during feature extraction/fine-tuning.
        for block in model.features.children():
            if not any(parameter.requires_grad for parameter in block.parameters()):
                block.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predictions = []
    targets = []
    logits_list = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_predictions = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (batch_predictions == labels).sum().item()
        total_samples += labels.size(0)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
        logits_list.append(outputs.detach().float().cpu())

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "predictions": predictions,
        "targets": targets,
        "logits": torch.cat(logits_list),
    }


def calibrate_temperature(logits, targets, device):
    """Fit one temperature value so softmax confidence better matches accuracy."""
    logits = logits.to(device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    log_temperature = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = criterion(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 10.0).item())


def save_checkpoint(
    path, model, classes, args, phase, epoch, validation_accuracy, temperature=1.0
):
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "architecture": "efficientnet_b0",
            "phase": phase,
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
            "temperature": temperature,
            "image_size": 224,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "arguments": vars(args),
        },
        path,
    )


def train_phase(
    phase,
    model,
    train_loader,
    validation_loader,
    criterion,
    device,
    classes,
    args,
    epochs,
    learning_rate,
):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    best_accuracy = -math.inf
    stale_epochs = 0
    checkpoint_path = f"best_{phase}.pt"

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device
        )
        validation_f1 = f1_score(
            validation_metrics["targets"],
            validation_metrics["predictions"],
            average="macro",
            zero_division=0,
        )
        scheduler.step()

        print(
            f"{phase} {epoch:03d}/{epochs:03d} | "
            f"train loss {train_metrics['loss']:.4f}, "
            f"acc {train_metrics['accuracy']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f}, "
            f"acc {validation_metrics['accuracy']:.4f}, "
            f"macro-F1 {validation_f1:.4f} | "
            f"{time.time() - started:.1f}s"
        )

        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path, model, classes, args, phase, epoch, best_accuracy
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping {phase}; best validation accuracy: {best_accuracy:.4f}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return best_accuracy


def main():
    args = parse_arguments()
    seed_everything(args.seed)
    device = select_device(args.require_gpu)
    data_dir = resolve_data_directory(args.data, args.kaggle_dataset)

    train_loader, validation_loader, test_loader, classes, targets = create_loaders(
        data_dir,
        args.batch_size,
        args.workers,
        args.seed,
        device,
        args.validation_size,
        args.test_size,
    )
    print(f"Classes: {len(classes)}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(validation_loader.dataset)}")
    print(f"Untouched test images: {len(test_loader.dataset)}")

    with open("classes.json", "w", encoding="utf-8") as class_file:
        json.dump(classes, class_file, indent=2)

    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    train_indices = train_loader.dataset.indices
    class_weights = make_class_weights(
        targets, train_indices, len(classes), device
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )

    print("\nPHASE 1: training classification head")
    set_feature_extraction(model)
    phase1_accuracy = train_phase(
        "phase1",
        model,
        train_loader,
        validation_loader,
        criterion,
        device,
        classes,
        args,
        args.epochs1,
        args.lr1,
    )

    print("\nPHASE 2: fine-tuning EfficientNet")
    set_fine_tuning(model, args.fine_tune_at)
    phase2_accuracy = train_phase(
        "phase2",
        model,
        train_loader,
        validation_loader,
        criterion,
        device,
        classes,
        args,
        args.epochs2,
        args.lr2,
    )

    validation_metrics = run_epoch(model, validation_loader, criterion, device)
    temperature = calibrate_temperature(
        validation_metrics["logits"], validation_metrics["targets"], device
    )
    final_metrics = run_epoch(model, test_loader, criterion, device)
    final_macro_f1 = f1_score(
        final_metrics["targets"],
        final_metrics["predictions"],
        average="macro",
        zero_division=0,
    )
    save_checkpoint(
        args.output,
        model,
        classes,
        args,
        "complete",
        args.epochs1 + args.epochs2,
        final_metrics["accuracy"],
        temperature,
    )
    with open("classification_report.txt", "w", encoding="utf-8") as report_file:
        report_file.write(
            classification_report(
                final_metrics["targets"],
                final_metrics["predictions"],
                labels=list(range(len(classes))),
                target_names=classes,
                zero_division=0,
            )
        )

    print("\nTRAINING COMPLETE")
    print(f"Phase 1 best validation accuracy: {phase1_accuracy:.4f}")
    print(f"Phase 2 best validation accuracy: {phase2_accuracy:.4f}")
    print(f"Untouched test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Untouched test macro-F1: {final_macro_f1:.4f}")
    print(f"Confidence temperature: {temperature:.4f}")
    print(f"Final model: {args.output}")
    print("Class labels: classes.json")
    print("Evaluation: classification_report.txt")


if __name__ == "__main__":
    main()
