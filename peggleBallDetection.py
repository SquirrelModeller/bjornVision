# Author: Squirrel Modeller

from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

_debuggingBallDetection = False

def imageProcessing(image_array, colortoAnalyze):
    # Creating masking image for excluding UI
    im = Image.new('RGB', (800, 600), (255, 255, 255))
    # im = Image.open('./mnt/data/stage2.jpg')
    draw = ImageDraw.Draw(im, 'RGBA')
    draw.rectangle((0,560,800,600), fill=(0,0,0))
    draw.rectangle((0,0,800,60), fill=(0,0,0))
    draw.rectangle((0,0,80,600), fill=(0,0,0))
    draw.rectangle((720,0,800,600), fill=(0,0,0))
    draw.rectangle((0,0,90,100), fill=(0,0,0))
    draw.rectangle((710,0,800,100), fill=(0,0,0))
    draw.ellipse((300, -20, 500, 170), fill=(0,0,0))
    draw.ellipse((-10, 75, 85, 170), fill=(0,0,0))
    draw.ellipse((710, 75, 805, 170), fill=(0,0,0))
    mask_array = np.array(im.convert('L'))

    # Define the orange color range
    color_min = None
    color_max = None
    if colortoAnalyze == 0:
        color_min = np.array([0, 20, 120], dtype=np.uint8)
        color_max = np.array([30, 60, 255], dtype=np.uint8)
    if colortoAnalyze == 1:
        color_min = np.array([120, 20, 0], dtype=np.uint8)
        color_max = np.array([255, 60, 30], dtype=np.uint8)
    if colortoAnalyze == 2:
        color_min = np.array([0, 120, 0], dtype=np.uint8)
        color_max = np.array([30, 255, 30], dtype=np.uint8)

    # Apply the mask to the image before processing. Here, we'll set the ignored areas (black in mask) to a color (black)
    masked_image_array = image_array.copy()

    # Expand the mask to cover all three channels
    mask_bool_3d = np.stack((mask_array,)*3, axis=-1)

    # Apply the mask: keep the pixel from the original image where the mask is True, else set to black
    masked_image_array = np.where(mask_bool_3d, image_array, [0, 0, 0])

    # Create mask for the pegs
    mask_pegs = np.all(np.logical_and(color_min <= masked_image_array, masked_image_array <= color_max), axis=-1)

    # Get the indices of the pixels
    color_pixels_y, color_pixels_x = np.where(mask_pegs)
    indentified_pixels = np.column_stack((color_pixels_x, color_pixels_y))

    if _debuggingBallDetection:
        global _debuggingImage
        _debuggingImage = mask_pegs

    return indentified_pixels

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
        ax.plot(peg[0], peg[1], 'ro', markersize=4)

    if _debuggingBallDetection:
        ax.clear()
        ax.axis('off')
        ax.imshow(_debuggingImage, interpolation='nearest')
        for cluster in _debuggingClusters:
            for sample in _debuggingClusters.get(cluster):
                ax.plot(sample[0], sample[1], 'ro', color=('blue'), markersize=0.2)

    # Save the figure with the peg positions marked
    output_path = './mnt/data/peg_positions_rounded.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches = 0)

    # Close the plot figure
    plt.close(fig)

def analyzeImage(image_path, colortoAnalyze = 0):
    """Return array of found pegs in provided image"""

    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Isolate targeted pegs
    dataImage = imageProcessing(image_array, colortoAnalyze)

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
    print(f"Identified pegs: {analyzeImage('./mnt/data/stage5.jpg', 1)}")

if __name__ == "__main__":
    main()