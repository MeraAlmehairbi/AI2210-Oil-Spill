
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def extract_features(image):
    """Convert a 2-channel SAR image into numerical features."""
    image = np.asarray(image)

    features = []

    for channel in image:
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.min(channel),
            np.max(channel),
            np.median(channel)
        ])

    return np.array(features)


def create_random_forest():
    """Create a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    return model
