import glob
import os
import cv2
import numpy as np
from utils.plotter import validate_folder
from pprint import pprint as pp

frames_folder = '/mnt/raptor/datasets/EgoProcel_zijia/frames/'


def load_jpg_images_to_array(folder_path, skip_rate=15, first_frame=0, last_frame=None):
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
    for i, jpg_file in enumerate(jpg_files[::skip_rate][first_frame:last_frame]):
        image_path = os.path.join(folder_path, jpg_file)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))

        # Append the image to the list
        image_list.append(img)

    # Convert the list of images to a NumPy array
    image_array = np.array(image_list)

    return image_array

if __name__ == '__main__':
    task_to_frame_folder = {
        'pc_disassembly': '/mnt/raptor/datasets/EgoProcel_zijia/frames/PC/disassemble',
        'pc_assembly': '/mnt/raptor/datasets/EgoProcel_zijia/frames/PC/assemble',
        'MECCANO': '/mnt/raptor/datasets/EgoProcel_zijia/frames/MECCANO',
        'CMU-MMAC.Salad': '/mnt/raptor/datasets/EgoProcel_zijia/frames/CMU_Kitchens/Salad',
        'CMU-MMAC.Eggs': '/mnt/raptor/datasets/EgoProcel_zijia/frames/CMU_Kitchens/Eggs',
        'CMU-MMAC.Brownie': '/mnt/raptor/datasets/EgoProcel_zijia/frames/CMU_Kitchens/Brownie',
        'CMU-MMAC.Pizza': '/mnt/raptor/datasets/EgoProcel_zijia/frames/CMU_Kitchens/Pizza',
        'CMU-MMAC.Sandwich': '/mnt/raptor/datasets/EgoProcel_zijia/frames/CMU_Kitchens/Sandwich',
        'EPIC-Tents': '/mnt/raptor/datasets/EgoProcel_zijia/frames/EPIC',
        'EGTEA_Gaze+.BaconAndEggs': '/mnt/raptor/datasets/EgoProcel_zijia/frames/egtea',
        'EGTEA_Gaze+.Cheeseburger': '/mnt/raptor/datasets/EgoProcel_zijia/frames/egtea',
        'EGTEA_Gaze+.TurkeySandwich': '/mnt/raptor/datasets/EgoProcel_zijia/frames/egtea',
        'EGTEA_Gaze+.PastaSalad': '/mnt/raptor/datasets/EgoProcel_zijia/frames/egtea',
        'EGTEA_Gaze+.Pizza': '/mnt/raptor/datasets/EgoProcel_zijia/frames/egtea',
        'EGTEA_Gaze+.ContinentalBreakfast': '/mnt/raptor/datasets/EgoProcel_zijia/frames/egtea',
        'EGTEA_Gaze+.GreekSalad': '/mnt/raptor/datasets/EgoProcel_zijia/frames/egtea',
    }

    for task_folder in glob.glob('./data/egoprocel-skip15/embeddings/*'):
        task = task_folder.split('/')[-1]
        if task != 'EPIC-Tents':
            continue
        print()
        print(task)
        frame_folder = task_to_frame_folder[task]
        the_task_out_folder = f'./data/egoprocel-skip15/frames/{task}'
        validate_folder(the_task_out_folder)
        for video_name in ['.'.join(x.split('/')[-1].split('.')[:-1]) for x in glob.glob(task_folder + '/*')]:
            if task.startswith('CMU'):
                video_filepath = f'{frame_folder}/{"_".join(video_name.split("_")[:2])}_Video/{video_name}'
            elif task.startswith('EPIC'):
                video_filepath = f'{frame_folder}/{video_name.split(".")[0]}'
                video_filepath = glob.glob(video_filepath + '/*')[0]
            else:
                video_filepath = f'{frame_folder}/{video_name}'
            assert os.path.isdir(video_filepath)
            video = load_jpg_images_to_array(video_filepath)
            print(f'{the_task_out_folder}/{video_name}.npy')
            np.save(f'{the_task_out_folder}/{video_name}.npy', video)
        # if task.startswith('CMU'):
        # og_frame_folder = task_to_frame_folder[task]


    # frames_folder = '/mnt/raptor/datasets/EgoProcel_zijia/frames/'
    # csvs = {'csvs': [], 'dirs': {}}
    # recursive_csv_search(frames_folder, csvs)
    # pp(frames_folder)

