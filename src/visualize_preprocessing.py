
from pathlib import Path
import matplotlib.pyplot as plt
from preprocessing import preprocess_image

# Find our sample satellite image
project = Path(__file__).resolve().parents[1]
path = project / "extracted" / "Oil" / "01162.tif"

# Process the same image in different ways
images = [
    ("P0 Baseline", preprocess_image(path)),
    ("P1 Noise reduction", preprocess_image(path, noise_filter="median")),
    ("P2 VV only", preprocess_image(path, channels="vv")),
    ("P2 VH only", preprocess_image(path, channels="vh")),
    ("P3 Resolution 512", preprocess_image(path, size=512)),
    ("P4 Percentile", preprocess_image(path, normalization="percentile")),
]

# Create a figure with 6 images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, (name, image) in zip(axes.flat, images):
    # Display the first available radar channel
    ax.imshow(image[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()

# Save the figure for our project report
output = project / "results" / "preprocessing_comparison.png"
output.parent.mkdir(exist_ok=True)

plt.savefig(output, dpi=150)
plt.show()

print("Saved:", output)
