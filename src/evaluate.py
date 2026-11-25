from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, log_loss, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset import OilSpillDataset
from models.resnet import build_resnet18

# Find our project files
root = Path(__file__).resolve().parent.parent

# Load validation images
data = OilSpillDataset(root / "dataset_manifest.csv", split="validation")
loader = DataLoader(data, batch_size=8)

# Load our trained ResNet18
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_resnet18().to(device)
model.load_state_dict(
    torch.load(root / "results/resnet18_baseline.pth",
               map_location=device, weights_only=True)
)
model.eval()

# Collect predictions
actual = []
predicted = []
probabilities = []
total_loss = 0

with torch.no_grad():
    for images, labels in loader:
        images = images.float().to(device)
        labels = labels.long().to(device)

        outputs = model(images)
        probs = torch.softmax(outputs.double(), dim=1)

        actual.extend(labels.cpu().tolist())
        predicted.extend(probs.argmax(1).cpu().tolist())
        probabilities.extend(probs.double().cpu().tolist())

# Calculate metrics
f1 = f1_score(actual, predicted, average="macro")
loss = log_loss(actual, probabilities, labels=[0, 1, 2])

print(f"Macro F1: {f1:.4f}")
print(f"Log Loss: {loss:.4f}")

# Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    actual,
    predicted,
    labels=[0, 1, 2],
    display_labels=["Oil", "No_oil", "Lookalike"],
    cmap="Blues"
)

plt.title("ResNet18 Confusion Matrix")
plt.savefig(root / "results/resnet18_confusion_matrix.png")
print("Confusion matrix saved!")