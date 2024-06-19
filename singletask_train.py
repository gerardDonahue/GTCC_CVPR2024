import json

import glob
import torch
from torch.utils.data import DataLoader
from utils.loss_entry import get_loss_function

from utils.os_util import get_env_variable
from utils.train_util import get_data_subfolder_and_extension, save_dict_to_json_file
from utils.logging import configure_logging_format
from utils.plotter import validate_folder
from utils.train_util import (
    get_base_model_deets
)
from utils.collate_functions import jsondataset_collate_fn
from models.json_dataset import jsondataset_get_train_test, data_json_labels_handles
from models.alignment_training_loop import alignment_training_loop
from models.model_multiprong import MultiProngAttDropoutModel
from configs.entry_config import get_generic_config



if __name__ == '__main__':
    # setup logger and pytorch device
    logger = configure_logging_format()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # environment variable
    dset_json_folder = get_env_variable('JSON_DPATH')

    # get configuration for this experiment
    CFG = get_generic_config()

    # initialize folder for logging output
    master_experiment_foldername = CFG.EVAL_PLOTFOLDER + f'/{CFG.EXPERIMENTNAME}'
    validate_folder(master_experiment_foldername)
    save_dict_to_json_file(CFG, master_experiment_foldername + '/config.json')

    # dataset structure json, get DSET variables
    data_structure = data_json_labels_handles(dset_json_folder, dset_name=CFG.DATASET_NAME)
    TASKS = data_structure.keys()
    data_subfolder_name, datafile_extension = get_data_subfolder_and_extension(architecture=CFG.BASEARCH.ARCHITECTURE)
    data_folder = f'{CFG.DATAFOLDER}/{data_subfolder_name}'
    
    # initiate training for each task in the dataset.
    for task in sorted(TASKS):
        # task-specific output folder.
        task_foldername = f'{master_experiment_foldername}/{task}'
        validate_folder(task_foldername)

        # get torch dataset for training and testing.
        train_set, test_set = jsondataset_get_train_test(
            task=task,
            task_json=data_structure[task],
            data_folder=data_folder,
            device=device,
            split=CFG.TRAIN_SPLIT[0],
            extension=datafile_extension,
            data_size=CFG.DATA_SIZE,
            lazy_loading=CFG.LAZY_LOAD
        )
        logger.debug(f'{len(train_set)} vids in train set for {task}')
        logger.debug(f'{len(test_set)} vids in test set for {task}')
        # wrap in torch dataloaders
        train_dataloader = DataLoader(train_set, batch_size=CFG.BATCH_SIZE, collate_fn=jsondataset_collate_fn, drop_last=True, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=CFG.BATCH_SIZE, collate_fn=jsondataset_collate_fn, drop_last=True, shuffle=True)
        logger.debug("Dataloaders successfully obtained.")
        
        # If using MCN
        if CFG.ARCHITECTURE['MCN']:
            base_model_class, base_model_params = get_base_model_deets(CFG)
            model = MultiProngAttDropoutModel(
                base_model_class=base_model_class,
                base_model_params=base_model_params,
                output_dimensionality=CFG.OUTPUT_DIMENSIONALITY,
                classes=[task],
                dropping=CFG.LOSS_TYPE['GTCC'],
                attn_layers=CFG.ARCHITECTURE['attn_layers'],
                drop_layers=CFG.ARCHITECTURE['drop_layers'],
            )
        # if using regular encoder
        else:
            base_model_class, base_model_params = get_base_model_deets(CFG)
            model = base_model_class(**base_model_params)
        
        # initiate training loop
        alignment_training_loop(
            model=model,
            train_dl_dict={task: train_dataloader},
            loss_fn=get_loss_function(CFG),
            foldername=task_foldername,
            CONFIG=CFG,
            test_dl_dict=None if not CFG.DEBUG else {task: test_dataloader}
        )
