import os
import cv2
import numpy as np
import tifffile
import imageio
from scipy.ndimage import label, binary_closing, binary_dilation, binary_erosion, generate_binary_structure
import matplotlib.pyplot as plt
from natsort import natsorted  # Natural sorting


for stackName in [str(e) for e in range(18,38)]:
    predPath = os.path.join("pred","3D","stacks","liver3D",stackName)
    tiffList = natsorted([f for f in os.listdir(predPath) if f.endswith(".tif")])
    print(tiffList)

    volume = np.stack([imageio.v2.imread(os.path.join(predPath, f)) for f in tiffList], axis=0)

    # Compute volume size
    depth, height, width = volume.shape
    total_pixels = depth * height * width
    print(f"Full volume size: {depth} x {height} x {width} = {total_pixels} pixels")

    print("--- PRED ---")
    # Label connected components
    labelCC, numCC = label(volume)
    print(f"Number of connected components: {numCC}")

    # Compute size of each connected component
    sizeCC = np.bincount(labelCC.ravel())[1:]  # Exclude background
    for i, size in enumerate(sizeCC, start=1):
        print(f"CC {i}: {size} pixels")

    tifffile.imwrite(os.path.join("post",stackName+"_pred.tif"), volume.astype(np.uint8)) 

    print("--- BIGGEST ---")
    # keep only the biggest
    labelCC[labelCC != np.argmax(sizeCC)+1] = 0

    # Compute size of each connected component
    sizeCC = np.bincount(labelCC.ravel())[1:]  # Exclude background
    for i, size in enumerate(sizeCC, start=1):
        print(f"CC {i}: {size} pixels")

    tifffile.imwrite(os.path.join("post", stackName + "_pred_one.tif"), (labelCC > 0).astype(np.uint8) * 255)

    ##############################

    binary_volume = (volume > 0).astype(np.uint8)

    # Create a new slice (background or empty slice of the same height and width)
    first_slice = np.zeros((1, binary_volume.shape[1], binary_volume.shape[2]), dtype=np.uint8)  # Empty slice
    last_slice = np.zeros((1, binary_volume.shape[1], binary_volume.shape[2]), dtype=np.uint8)  # Empty slice

    # Add slices to the volume (prepend and append)
    binary_volume = np.concatenate([first_slice, binary_volume, last_slice], axis=0)

    print("--- PRED CLOSED ---")
    struct_element = generate_binary_structure(rank=3, connectivity=2)
    closed_volume = binary_closing(binary_volume, structure=struct_element, iterations=1) # IF I WANT TO add iteration, I need to add slices

    # Remove the first and last slices after closing
    closed_volume = closed_volume[1:-1]

    labelCC, numCC = label(closed_volume)

    # Compute size of each connected component
    sizeCC = np.bincount(labelCC.ravel())[1:]  # Exclude background
    for i, size in enumerate(sizeCC, start=1):
        print(f"CC {i}: {size} pixels")

    tifffile.imwrite(os.path.join("post",stackName+"_close.tif"), closed_volume.astype(np.uint8)*255)     

    print("--- BIGGEST CLOSED ---")
    # keep only the biggest
    labelCC[labelCC != np.argmax(sizeCC)+1] = 0

    # Compute size of each connected component
    sizeCC = np.bincount(labelCC.ravel())[1:]  # Exclude background
    for i, size in enumerate(sizeCC, start=1):
        print(f"CC {i}: {size} pixels")

    stack_to_save = (labelCC > 0).astype(np.uint8) * 255
    print(stack_to_save.shape)
    tifffile.imwrite(os.path.join("post", stackName + "_close_one.tif"), stack_to_save)
    # tifffile.imwrite(os.path.join("post", stackName + "_close_one.tif"), (labelCC > 0).astype(np.uint8) * 255)
