# Author: Squirrel Modeller

from PIL import Image
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

_debuggingBallDetection = False

def imageProcessing(image_array):
    mask_path = './mnt/data/Mask.png'
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask)

    # Apply the mask to the image before processing. Here, we'll set the ignored areas (black in mask) to a color (black)
    masked_image_array = image_array.copy()

    # Expand the mask to cover all three channels
    mask_bool_3d = np.stack((mask_array,)*3, axis=-1)

    # Apply the mask: keep the pixel from the original image where the mask is True, else set to black
    masked_image_array = np.where(mask_bool_3d, image_array, [0, 0, 0])

    # Define the orange color range
    orange_min = np.array([120, 20, 0], dtype=np.uint8)
    orange_max = np.array([255, 60, 30], dtype=np.uint8)

    # Create mask for the orange pegs
    mask_orange_pegs = np.all(np.logical_and(orange_min <= masked_image_array, masked_image_array <= orange_max), axis=-1)

    # Get the indices of the orange pixels
    orange_pixels_y, orange_pixels_x = np.where(mask_orange_pegs)
    orange_pixels = np.column_stack((orange_pixels_x, orange_pixels_y))

    if _debuggingBallDetection:
        global _debuggingImage
        _debuggingImage = mask_orange_pegs

    return orange_pixels

def getPositions(dataImage):
    # Apply DBSCAN clustering algorithm on the individual pixels
    dbscan = DBSCAN(eps=5, min_samples=10).fit(dataImage)
    labels = dbscan.labels_

    # Number of clusters (ignoring noise)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Create a dictionary for clusters
    clusters = {i: dataImage[labels == i] for i in range(n_clusters_)}

    # Filter clusters based on pixel density
    peg_positions = [cluster.mean(axis=0).round().astype(int).tolist() for cluster in clusters.values() if len(cluster) > 10]

    if _debuggingBallDetection:
        global _debuggingClusters
        _debuggingClusters = clusters

    return peg_positions

def visualization(dataImage, peg_positions):
    fig, ax = plt.subplots(frameon=False, dpi=162.5)
    ax.imshow(dataImage, interpolation='nearest')
    ax.axis('off')

    # Draw peg_positions on image
    for peg in peg_positions:
        ax.plot(peg[0], peg[1], 'ro', markersize=10)

    if _debuggingBallDetection:
        ax.clear()
        ax.axis('off')
        ax.imshow(_debuggingImage, interpolation='nearest')
        for cluster in _debuggingClusters:
            for sample in _debuggingClusters.get(cluster):
                ax.plot(sample[0], sample[1], 'ro', color=('blue', 0.5), markersize=0.2)

    # Save the figure with the peg positions marked
    output_path = './mnt/data/peg_positions_rounded.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches = 0)

    # Close the plot figure
    plt.close(fig)

def analyzeImage(image_path):
    """Return array of found pegs in provided image"""
    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Isolate targeted pegs
    dataImage = imageProcessing(image_array)

    # Send isolated pegs to cluster analysis
    peg_positions = getPositions(dataImage)

    # Visualize data
    visualization(image, peg_positions)

    # Debug warning
    if _debuggingBallDetection:
        print(f"WARNING! Debugging is {_debuggingBallDetection}") 

    return peg_positions

def main():
    # #NOTICE! stage*.jpg should be the image that you want to process. Due to copyright no test images can be uploaded to this repo.
    print(f"Identified pegs: {analyzeImage('./mnt/data/stage4.jpg')}")

if __name__ == "__main__":
    main()