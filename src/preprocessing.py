import numpy as np
import tifffile


def preprocess_image(path, size=256):
    # Read the two radar channels (VV and VH)
    image = tifffile.imread(path).astype(np.float32)

    # Replace invalid numbers, just in case
    image = np.nan_to_num(image, nan=-60.0, posinf=10.0, neginf=-60.0)

    # Normalize each channel separately
    vv = np.clip((image[:, :, 0] + 60) / 60, 0, 1)
    vh = np.clip((image[:, :, 1] + 55) / 65, 0, 1)

    # Resize using evenly spaced pixels
    rows = np.linspace(0, image.shape[0] - 1, size).astype(int)
    cols = np.linspace(0, image.shape[1] - 1, size).astype(int)

    result = np.stack([
        vv[np.ix_(rows, cols)],
        vh[np.ix_(rows, cols)]
    ])

    return result.astype(np.float32)
