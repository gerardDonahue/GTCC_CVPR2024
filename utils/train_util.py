import glob
from easydict import EasyDict as edict
import json 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.model_singleprong import Resnet50Encoder, StackingEncoder, NaiveEncoder
from models.model_multiprong import MultiProngAttDropoutModel
from utils.logging import configure_logging_format

logger = configure_logging_format()



def save_dict_to_json_file(dictionary, filepath):
    with open(filepath, 'w') as fp:
        json.dump(dictionary, fp, indent=4)


def get_npy_shape_from_file(file_path):
    with open(file_path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f)
    return shape


def get_base_model_deets(config_obj):
    if type(config_obj) == dict:
        config_obj = edict(config_obj)
    architecture = config_obj.BASEARCH.ARCHITECTURE
    if architecture == 'temporal_stacking':
        return StackingEncoder, config_obj.BASEARCH.TEMPORAL_STACKING_ARCH
    elif architecture == 'naive':
        return NaiveEncoder, config_obj.BASEARCH.NAIVE_ARCH
    elif architecture == 'resnet50':
        return Resnet50Encoder, config_obj.BASEARCH.Resnet50_ARCH
    else:
        logger.error("Bad Architecture Value, check CONFIG object")
        exit(1)

def flatten_dataloader(dl):
    return [pair for batch in list(iter(dl)) for pair in list(zip(batch[0], batch[1]))]


def ckpt_restore_mprong(path, num_heads, dropout=False, device='cpu'):
    # Additional information
    checkpoint = torch.load(path, map_location="cpu")
    config_obj = checkpoint['config']
    base_model_class, base_model_params = get_base_model_deets(config_obj)
    if 'drop_layers' in config_obj['ARCHITECTURE'].keys():
        model = MultiProngAttDropoutModel(
            base_model_class=base_model_class,
            base_model_params=base_model_params,
            output_dimensionality=config_obj['OUTPUT_DIMENSIONALITY'],
            num_heads=num_heads,
            dropping=dropout,
            attn_layers=config_obj['ARCHITECTURE']['attn_layers'],
            drop_layers=config_obj['ARCHITECTURE']['drop_layers'],
        ).to(device)
    else:
        model = MultiProngAttDropoutModel(
            base_model_class=base_model_class,
            base_model_params=base_model_params,
            output_dimensionality=config_obj['OUTPUT_DIMENSIONALITY'],
            num_heads=num_heads,
            dropping=False,
            attn_layers=config_obj['ARCHITECTURE']['attn_layers'],
        ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config_obj['LEARNING_RATE'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (
        model,
        optimizer,
        checkpoint['epoch'],
        checkpoint['loss'],
        config_obj
    )


def ckpt_restore_sprong(path, device='cpu'):
    # Additional information
    checkpoint = torch.load(path, map_location="cpu")
    config_obj = checkpoint['config']
    base_model_class, base_model_params = get_base_model_deets(config_obj)
    model = base_model_class(**base_model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config_obj['LEARNING_RATE'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (
        model,
        optimizer,
        checkpoint['epoch'],
        checkpoint['loss'],
        config_obj
    )



def get_data_subfolder_and_extension(architecture):
    """
        Based on the architecture, return the subfolder where the data files will be and the file-entension of the data files.
    """
    if architecture == 'resnet50':
        return 'frames', 'npy'
    else:
        return 'features', 'npy'


def get_config_for_folder(folder):
    try:
        with open(folder + '/config.json', 'r') as json_file:
            config = edict(json.load(json_file))
    except Exception as e:
        print(glob.glob(folder + '/*'))
        folder_to_check = glob.glob(folder + '/*')[0]
        with open(folder_to_check + '/config.json', 'r') as json_file:
            config = edict(json.load(json_file))
    return config

