
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss
)

from dataset import OilSpillDataset
from models.random_forest import (
    create_random_forest,
    extract_features
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "dataset_manifest.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Load the dataset
train_data = OilSpillDataset(MANIFEST, split="train")
val_data = OilSpillDataset(MANIFEST, split="validation")

print("Training images:", len(train_data))
print("Validation images:", len(val_data))


def prepare_data(dataset):
    features = []
    labels = []

    for i in range(len(dataset)):
        image, label = dataset[i]

        features.append(extract_features(image))
        labels.append(label)

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images")

    return np.array(features), np.array(labels)


print("\nPreparing training data...")
X_train, y_train = prepare_data(train_data)

print("\nPreparing validation data...")
X_val, y_val = prepare_data(val_data)

print("\nTraining Random Forest...")
model = create_random_forest()
model.fit(X_train, y_train)

print("Training complete!")

predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)

print("\nAccuracy:", accuracy_score(y_val, predictions))
print("Macro F1:", f1_score(y_val, predictions, average="macro"))

print("\nClassification report:")
print(classification_report(
    y_val, predictions,
    labels=[0, 1, 2],
    target_names=["Oil", "No oil", "Look-alike"],
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_val, predictions, labels=[0, 1, 2]))

print("\nLog loss:", log_loss(
    y_val, probabilities,
    labels=model.classes_
))

joblib.dump(model, RESULTS / "random_forest.joblib")
print("\nModel saved!")
