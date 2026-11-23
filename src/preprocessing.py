
import numpy as np
import tifffile
from scipy.ndimage import median_filter


def preprocess_image(path, size=256, channels="both",
                     noise_filter="none", normalization="fixed"):

    # Open the satellite image
    image = tifffile.imread(path)
    image = image.astype(np.float32)

    # Replace invalid numbers
    image = np.nan_to_num(image, nan=-60, posinf=10, neginf=-60)

    # Get the two radar channels
    vv = image[:, :, 0]
    vh = image[:, :, 1]

    # P4: Try different normalization methods
    if normalization == "fixed":

        # Our original method
        vv = (vv + 60) / 60
        vh = (vh + 55) / 65

        vv = np.clip(vv, 0, 1)
        vh = np.clip(vh, 0, 1)

    elif normalization == "percentile":

        # Find low and high values for each channel
        vv_low = np.percentile(vv, 1)
        vv_high = np.percentile(vv, 99)

        vh_low = np.percentile(vh, 1)
        vh_high = np.percentile(vh, 99)

        # Normalize the channels separately
        vv = (vv - vv_low) / max(vv_high - vv_low, 0.000001)
        vh = (vh - vh_low) / max(vh_high - vh_low, 0.000001)

        vv = np.clip(vv, 0, 1)
        vh = np.clip(vh, 0, 1)

    else:
        raise ValueError("Unknown normalization method")

    # P3: Change the image resolution
    # Default is 256, but we can also try 512
    rows = np.linspace(0, image.shape[0] - 1, size).astype(int)
    cols = np.linspace(0, image.shape[1] - 1, size).astype(int)

    vv = vv[np.ix_(rows, cols)]
    vh = vh[np.ix_(rows, cols)]

    # P1: Try noise reduction
    if noise_filter == "median":

        # A median filter makes the image less noisy
        vv = median_filter(vv, size=3)
        vh = median_filter(vh, size=3)

    elif noise_filter != "none":
        raise ValueError("Unknown noise filter")

    # P2: Choose which radar channels to use
    if channels == "vv":
        result = np.stack([vv])

    elif channels == "vh":
        result = np.stack([vh])

    elif channels == "both":
        result = np.stack([vv, vh])

    else:
        raise ValueError("Unknown channel option")

    # Return the prepared image
    return result.astype(np.float32)
