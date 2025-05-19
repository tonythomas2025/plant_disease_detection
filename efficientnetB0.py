import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

#  CONFIG
SOURCE_DIR = "C:/Users/thoms/OneDrive/Desktop/Capstone/PlantVillage"
DEST_DIR = "C:/Users/thoms/PycharmProjects/efficientnet/lastsplit"
SPLIT_RATIOS = (0.7, 0.15, 0.15)
IMAGE_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 5e-5
PATIENCE = 3
MODEL_SAVE_PATH = "efficientnet_b0_best1.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  AUTO-SPLIT FUNCTION
def auto_split_dataset(source_dir, dest_dir, ratios):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        if class_name.lower() in ['train', 'val', 'test']:
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])
        n_test = n_total - n_train - n_val

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, files in splits.items():
            split_dir = os.path.join(dest_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for file in tqdm(files, desc=f"{split}/{class_name}"):
                shutil.copy2(os.path.join(class_path, file), os.path.join(split_dir, file))

#  MAIN EXECUTION 
if __name__ == "__main__":
    if not os.path.exists(os.path.join(DEST_DIR, 'train')):
        print("Splitting dataset...")
        auto_split_dataset(SOURCE_DIR, DEST_DIR, SPLIT_RATIOS)
    else:
        print("Split folders already exist. Skipping auto-split.")

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DEST_DIR, 'train'), transform=train_transforms)

    class_counts = {}
    for _, label in train_dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1

    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]

    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(
        datasets.ImageFolder(os.path.join(DEST_DIR, 'val'), transform=val_test_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        datasets.ImageFolder(os.path.join(DEST_DIR, 'test'), transform=val_test_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    NUM_CLASSES = len(train_loader.dataset.classes)
    print(f"Classes Detected: {NUM_CLASSES} – {train_loader.dataset.classes}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Best model saved!")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break

    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100 * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
