# Bjorn Vision

This image recognition tool identifies the X and Y coordinates of orange balls within an image. It filters out UI elements using a mask and processes the image to detect the specified objects.

## Functionality
The tool utilizes a mask to exclude UI elements from the processing area. It then converts the image into an array, examining each pixel against a pre-defined color threshold. Pixels within the threshold are considered part of the object and are included in a new mask. This mask undergoes clustering analysis via the DBSCAN algorithm to locate areas with high pixel concentration, indicating the presence of an object.

## Features
- Prints the locations of orange balls.
- Generates an image highlighting the identified targets.

## Planned Enhancements
- Inclusion of blue ball detection.
- Option to switch coordinate systems.
- Improved color filtering for more precise detection.
- Refined calculations for cluster analysis.
