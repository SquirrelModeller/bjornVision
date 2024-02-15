# Author: Squirrel Modeller

from PIL import Image
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#NOTICE! stage1.jpg should be the image that you want to process. Due to copyright no test images can be uploaded to this repo.

image_path = './mnt/data/stage1.jpg'  
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
orange_min = np.array([130, 20, 20], dtype=np.uint8)
orange_max = np.array([255, 110, 30], dtype=np.uint8)

# Create mask for the orange pegs
mask_orange_pegs = np.all(np.logical_and(orange_min <= masked_image_array, masked_image_array <= orange_max), axis=-1)

# Label the connected regions of the mask
labeled_array, num_features = ndimage.label(mask_orange_pegs)

# Find the centers of the pegs
centers = ndimage.center_of_mass(mask_orange_pegs, labeled_array, range(1, num_features + 1))

# Convert centers to (x, y) format and round off the coordinates
centers = [(round(center[1]), round(center[0])) for center in centers]

# Convert centers to a numpy array for clustering
centers_np = np.array(centers)

# Apply DBSCAN clustering algorithm
dbscan = DBSCAN(eps=10, min_samples=4).fit(centers_np)
labels = dbscan.labels_

# Number of clusters (ignoring noise)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Create a dictionary for clusters
clusters = {i: centers_np[labels == i] for i in range(n_clusters_)}

# Filter clusters based on size to identify pegs
peg_positions = [cluster.mean(axis=0).round().astype(int).tolist() for cluster in clusters.values() if len(cluster) > 3]

# Print coordinates of the identified pegs
print(f"Identified pegs: {peg_positions}")

# Visualization
fig, ax = plt.subplots(frameon=False)
ax.axis('off')
ax.imshow(image_array, interpolation='nearest')

# Plot pegs as red circles
for peg in peg_positions:
    ax.plot(peg[0], peg[1], 'ro', markersize=10)

# Save the figure with the peg positions marked
output_path = './mnt/data/peg_positions_rounded.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches = 0)

# Close the plot figure
plt.close(fig)