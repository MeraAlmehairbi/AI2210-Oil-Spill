
import sys
from pathlib import Path

# Find our project files
project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src"))

from preprocessing import preprocess_image

# Choose one oil spill image
path = project / "extracted" / "Oil" / "01162.tif"

# P0: Original method
p0 = preprocess_image(path)

# P1: Noise reduction
p1 = preprocess_image(path, noise_filter="median")

# P2: Different radar channels
p2_vv = preprocess_image(path, channels="vv")
p2_vh = preprocess_image(path, channels="vh")

# P3: Different image resolution
p3 = preprocess_image(path, size=512)

# P4: Different normalization
p4 = preprocess_image(path, normalization="percentile")

# Check the results
results = {
    "P0 Baseline": p0,
    "P1 Noise reduction": p1,
    "P2 VV only": p2_vv,
    "P2 VH only": p2_vh,
    "P3 Resolution": p3,
    "P4 Normalization": p4
}

for name, image in results.items():
    print(name)
    print("Shape:", image.shape)
    print("Minimum:", image.min())
    print("Maximum:", image.max())
    print()
