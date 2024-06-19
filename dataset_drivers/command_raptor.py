import glob
import os
import cv2
import numpy as np

def load_jpg_images_to_array(folder_path):
    """
    Load a sequence of JPG images from a folder into a NumPy array.

    Args:
        folder_path (str): Path to the folder containing JPG images.

    Returns:
        numpy.ndarray: NumPy array containing the loaded images with shape (N x W x H x 3).
    """
    # Get a list of JPG files in the folder
    jpg_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    # Initialize an empty list to store the images
    image_list = []

    # Loop through the JPG files and load each image
    for i, jpg_file in enumerate(jpg_files[::15]):
        image_path = os.path.join(folder_path, jpg_file)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))

        # Append the image to the list
        image_list.append(img)

    # Convert the list of images to a NumPy array
    image_array = np.array(image_list)

    return image_array



print('HI')
egtea = '/mnt/raptor/datasets/EgoProcel_zijia/frames/EPIC/'
hi = glob.glob(egtea + '/*')

print(hi)
for folder in hi:
    name = folder.split('/')[-1]
    array = load_jpg_images_to_array(folder)
    np.save('./data/egoprocel-skip15/frames/EPIC-Tents/' + name, array)
    print('SAVED')
