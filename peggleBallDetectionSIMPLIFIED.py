# Author: Squirrel Modeller

from PIL import Image
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#NOTICE! stage1.jpg should be the image that you want to process. Due to copyright no test images can be uploaded to this repo.

debugging = False

image_path = './mnt/data/stage4.jpg'  
mask_path = './mnt/data/Mask.png'

# Load the image
image = Image.open(image_path)
image_array = np.array(image)

mask = Image.open(mask_path).convert('L')
mask_array = np.array(mask)

# Apply the mask to the image before processing. Here, we'll set the ignored areas (black in mask) to a color (black)
masked_image_array = image_array.copy()
non_processing_color = np.array([0, 0, 0], dtype=np.uint8)  # Black color
mask_inv = mask_array == 0  # Inverted mask to select black areas

# Set the non-processing areas to black
for c in range(3):  # Apply the inverse mask to each of the three color channels
    masked_image_array[..., c][mask_inv] = non_processing_color[c]

# Define the orange color range
orange_min = np.array([120, 20, 0], dtype=np.uint8)
orange_max = np.array([255, 60, 30], dtype=np.uint8)

# Create mask for the orange pegs
mask_orange_pegs = np.all(np.logical_and(orange_min <= masked_image_array, masked_image_array <= orange_max), axis=-1)

# Get the indices of the orange pixels
orange_pixels_y, orange_pixels_x = np.where(mask_orange_pegs)
orange_pixels = np.column_stack((orange_pixels_x, orange_pixels_y))

# Apply DBSCAN clustering algorithm on the individual pixels
dbscan = DBSCAN(eps=5, min_samples=10).fit(orange_pixels)
labels = dbscan.labels_

# Number of clusters (ignoring noise)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Create a dictionary for clusters
clusters = {i: orange_pixels[labels == i] for i in range(n_clusters_)}

# Filter clusters based on pixel density
peg_positions = [cluster.mean(axis=0).round().astype(int).tolist() for cluster in clusters.values() if len(cluster) > 10]

print(f"Identified pegs: {peg_positions}")

# Visualization
fig, ax = plt.subplots(frameon=False, dpi=162.5)
ax.axis('off')
ax.imshow(image, interpolation='nearest')

# Plot pegs as red circles
for peg in peg_positions:
    ax.plot(peg[0], peg[1], 'ro', markersize=10)

# Debugging, bad code!
if (debugging == True):
    ax.imshow(mask_orange_pegs, interpolation='nearest')
    for peg in clusters:
        for pixel in clusters.get(peg):
            ax.plot(pixel[0], pixel[1], 'ro', markersize=0.2, color=('blue', 1))


# Save the figure with the peg positions marked
output_path = './mnt/data/peg_positions_rounded.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches = 0)

# Close the plot figure
plt.close(fig)