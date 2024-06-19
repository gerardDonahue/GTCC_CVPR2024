"""
    This file shows how to make a random synthetic dataset that follows the structure of the egoprocel.json file in this folder. 

    The data being generated here is random, to help users what the dataloader code requires.
"""
import json
import os
import numpy as np


with open('./egoprocel.json', 'r') as file:
    data_structure = json.load(file)

tasks = data_structure.keys()
print(tasks)
path_to_synthetic_simplefeatures = './egoprocel/features'
# os.makedirs(path_to_synthetic_simplefeatures)
path_to_synthetic_features = './egoprocel/frames'
# os.makedirs(path_to_synthetic_features)

feature_dimensionality = 4

for task in tasks:
    print(f'Making synthetic datafiles for {task} task')

    for handle in data_structure[task]['handles']:
        random_video_length = np.random.choice(list(range(10,30)))
        simple_features = np.random.random((random_video_length, feature_dimensionality))
        features = np.random.random((random_video_length, feature_dimensionality, 14, 14))
        np.save(f'{path_to_synthetic_simplefeatures}/{handle}.npy', simple_features)
        np.save(f'{path_to_synthetic_features}/{handle}.npy', features)

