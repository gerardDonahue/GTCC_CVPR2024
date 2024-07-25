import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import imageio
import glob
import random
import os
import pandas as pd
import time
import json
from utils.collate_functions import jsondataset_collate_fn
from utils.logging import configure_logging_format
from utils.train_util import get_data_subfolder_and_extension, get_npy_shape_from_file


logger = configure_logging_format()

def jsondataset_get_train_test(
    task,
    task_json,
    data_folder,
    device,
    split,
    extension='npy',
    data_size=None,
    lazy_loading=False
):
    return (
        JSONDataset(
            task=task,
            task_json=task_json,
            data_folder=data_folder,
            split=[split],
            extension=extension,
            loader_type='train',
            lazy_loading=lazy_loading
        ),
        JSONDataset(
            task=task,        
            task_json=task_json,
            data_folder=data_folder,
            split=[split],
            extension=extension,
            loader_type='test',
            lazy_loading=lazy_loading
        )
    )

class JSONDataset(Dataset):
    def __init__(
        self,
        task,
        task_json,
        data_folder,
        split,
        extension='npy',
        loader_type=True,
        lazy_loading=True
    ):
        """
            This filepath must maintain the following structure
                data_folder/
                    - embeddings/
                        baseball_pitch/
                            0001.npy
                            ...
                        ...
                    - times/
                        baseball_pitch/
                            0001.csv
                            ...
                        ...
        """
        # make sure split is either [train-test-split] or [train-testval-split, testval-split]
        assert type(split) == list and len(split) in [1,2] and all([0 < p < 1 for p in split])
        # make sure if only train/test split is give that loader_type isnt val
        with_val = len(split) == 2
        assert with_val or loader_type != 'val'

        self.data_folder = data_folder + f'/{task}'
        self.task = task
        self.split = split

        N = len(task_json['handles'])
        all_embeddings_files = glob.glob(f'{data_folder}/*')
        num_vids_total = len(task_json['handles'])
        train_end_idx = round(num_vids_total * split[0])
        test_start_idx = round(train_end_idx + (num_vids_total-train_end_idx) * split[1]) if with_val else train_end_idx
        all_handle_indices = list(range(N))
        if loader_type == 'train':
            active_hdl_indices = all_handle_indices[:train_end_idx]
        elif loader_type == 'val':
            assert with_val
            active_hdl_indices = all_handle_indices[train_end_idx:test_start_idx]
        elif loader_type == 'test':
            active_hdl_indices = all_handle_indices[test_start_idx:num_vids_total]
        else:
            logger.error("Bad loader type (should be in ['train', 'val', 'test'])")
            exit(1)


        datas = []
        times = []
        names = []
        self.action_set = set()
        for i, (handle, action_sequence, start_times, end_times) in enumerate(zip(
            task_json['handles'], task_json['hdl_actions'], task_json['hdl_start_times'], task_json['hdl_end_times']
        )):
            for a in action_sequence:
                self.action_set.add(a)

            if i not in active_hdl_indices:
                continue
            assert f'{data_folder}/{handle}.{extension}' in all_embeddings_files, f'File {handle}.{extension} not in {data_folder} folder'
            # log the data filename
            data = f'{data_folder}/{handle}.{extension}'
            times_dict = {'step': action_sequence, 'start_frame': [int(t) for t in start_times], 'end_frame': [int(t) for t in end_times], 'name': handle}
            N = get_npy_shape_from_file(data)[0]
            
            if N > times_dict['end_frame'][-1] - 1:
                times_dict['end_frame'][-1] = N-1
            elif N < times_dict['end_frame'][-1] - 1:
                while N < times_dict['end_frame'][-1] - 1:
                    if N > times_dict['start_frame'][-1]+1:
                        times_dict['end_frame'][-1] = N-1
                        break
                    else:
                        times_dict['start_frame'] = times_dict['start_frame'][:-1]
                        times_dict['end_frame'] = times_dict['end_frame'][:-1]

            # add the times data dict
            datas.append(data if lazy_loading else np.load(data))
            times.append(times_dict)
            names.append(task + '_' + handle)

        # logger.info("Embeddings folder and time label folder contain same file, times are in order, moving on")
        self.times = times
        self.data_label_name = list(zip(datas, times, names))
        if loader_type == 'test':
            random.seed(1)
        random.shuffle(self.data_label_name)

    def __len__(self):
        """
            gives the length of the dataset
        """
        return len(self.data_label_name)
    
    def __getitem__(self, index):
        """
            responsible for returning the 'index'^th data and label from wherever.
        """
        if type(self.data_label_name[index][0]) == str:
            return self.data_label_name[index][0], self.data_label_name[index][1], self.data_label_name[index][2]
        else:
            return torch.from_numpy(self.data_label_name[index][0]), self.data_label_name[index][1], self.data_label_name[index][2]


def data_json_labels_handles(dset_json_folder, dset_name='egoprocel'):
    subset_specifier = None
    if dset_name in ['cmu', 'egtea']:
        subset_specifier = dset_name
        dset_name = 'egoprocel'

    data_json = f"{dset_json_folder}/{dset_name}.json"
    with open(data_json, 'r') as file:
        d = json.load(file)
        if subset_specifier is None:
            return d
        else:
            return {key: value for key, value in d.items() if subset_specifier in key}


def get_test_dataloaders(tasks, data_structure, config, device):
    batch_size = config.BATCH_SIZE
    data_subfolder_name, datafile_extension = get_data_subfolder_and_extension(architecture=config.BASEARCH.ARCHITECTURE)
    data_folder = f'{config.DATAFOLDER}/{data_subfolder_name}'
    test_dataloaders = {}
    for task in tasks:
        _, test_set = jsondataset_get_train_test(
            task=task,
            task_json=data_structure[task],
            data_folder=data_folder,
            device=device,
            split=config.TRAIN_SPLIT[0],
            extension=datafile_extension,
            data_size=config.DATA_SIZE,
            lazy_loading=config.LAZY_LOAD
        )
        logger.debug(f'{len(test_set)} vids in test set for {task}')
        if batch_size is None:
            batch_size = len(test_set)
        test_dataloaders[task] = DataLoader(test_set, batch_size=batch_size, collate_fn=jsondataset_collate_fn, drop_last=True, shuffle=False)
    return test_dataloaders


def get_train_dataloaders(tasks, data_structure, config, device):
    batch_size = config.BATCH_SIZE
    data_subfolder_name, datafile_extension = get_data_subfolder_and_extension(architecture=config.BASEARCH.ARCHITECTURE, dataset=config.DATASET_NAME)
    data_folder = f'{config.DATAFOLDER}/{data_subfolder_name}'
    test_dataloaders = {}
    for task in tasks:
        _, test_set = jsondataset_get_train_test(
            task=task,
            task_json=data_structure[task],
            data_folder=data_folder,
            device=device,
            split=config.TRAIN_SPLIT[0],
            extension=datafile_extension,
            data_size=config.DATA_SIZE,
            lazy_loading=config.LAZY_LOAD
        )
        # logger.debug(f'{len(test_set)} vids in train set for {task}')
        if batch_size is None:
            batch_size = len(test_set)
        test_dataloaders[task] = DataLoader(test_set, batch_size=batch_size, collate_fn=jsondataset_collate_fn, drop_last=True, shuffle=False)
    return test_dataloaders
