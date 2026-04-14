import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

import CNN


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Prefer the user's provided folder name; fallback to original 'Dataset'
PREFERRED_DATASET_NAME = 'Plant_leave_diseases_dataset_with_augmentation'
PREFERRED_DATASET_DIR = os.path.join(PROJECT_ROOT, PREFERRED_DATASET_NAME)
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, 'Dataset')
DATASET_DIR = PREFERRED_DATASET_DIR if os.path.isdir(PREFERRED_DATASET_DIR) else DEFAULT_DATASET_DIR  # expects class subfolders
MODEL_PATH = os.path.join(APP_DIR, 'plant_disease_model_1_latest.pt')
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model() -> torch.nn.Module:
    model = CNN.CNN(39)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and any(k.startswith(('conv_layers', 'dense_layers')) for k in state.keys()):
        model.load_state_dict(state)
    else:
        model = state
    model.to(DEVICE)
    model.eval()
    return model


def get_dataset_and_samplers() -> Tuple[datasets.ImageFolder, SubsetRandomSampler, SubsetRandomSampler, SubsetRandomSampler]:
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            (
                "Dataset folder not found. Checked:\n"
                f"- {PREFERRED_DATASET_DIR}\n"
                f"- {DEFAULT_DATASET_DIR}\n"
                "Please place your data as class subfolders inside one of these."
            )
        )

    tfms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    ds = datasets.ImageFolder(DATASET_DIR, transform=tfms)

    indices = list(range(len(ds)))
    # Reproduce notebook split: 85% first part, inside that first part: 70% train, 30% validation; remain 15% test
    split = int((0.85 * len(ds)) // 1)
    validation_cut = int((0.70 * split) // 1)

    train_indices = indices[:validation_cut]
    validation_indices = indices[validation_cut:split]
    test_indices = indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return ds, train_sampler, validation_sampler, test_sampler


def accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            n_correct += (preds == targets).sum().item()
            n_total += targets.size(0)
    return (n_correct / n_total) if n_total else 0.0


def main():
    try:
        ds, train_sampler, validation_sampler, test_sampler = get_dataset_and_samplers()
    except FileNotFoundError as e:
        print(str(e))
        return

    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    validation_loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=validation_sampler, num_workers=0)
    test_loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=0)

    model = load_model()

    train_acc = accuracy(model, train_loader)
    val_acc = accuracy(model, validation_loader)
    test_acc = accuracy(model, test_loader)

    print(f"Samples -> train: {len(train_sampler.indices)}, val: {len(validation_sampler.indices)}, test: {len(test_sampler.indices)}")
    print(f"Train accuracy: {train_acc*100:.2f}%")
    print(f"Validation accuracy: {val_acc*100:.2f}%")
    print(f"Test accuracy: {test_acc*100:.2f}%")


if __name__ == '__main__':
    main()
