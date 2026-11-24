import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import OilSpillDataset
from models.resnet import build_resnet18

# 1. Settings
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# 2. Load our dataset
root = Path(__file__).resolve().parent.parent
manifest = root / "dataset_manifest.csv"

train_data = OilSpillDataset(manifest, split="train")
val_data = OilSpillDataset(manifest, split="validation")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# 3. Create ResNet18
device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_resnet18().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Train and validate
for epoch in range(EPOCHS):

    # Training
    model.train()
    for images, labels in train_loader:
        images = images.float().to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.float().to(device)
            labels = labels.long().to(device)

            predictions = model(images).argmax(1)
            correct += (predictions == labels).sum().item()
            total += len(labels)

    print(f"Epoch {epoch + 1}: Accuracy = {correct / total:.2%}")

# 5. Save trained model
output = root / "results" / "resnet18_baseline.pth"
output.parent.mkdir(exist_ok=True)

torch.save(model.state_dict(), output)
print("Model saved!")