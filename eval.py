import glob
import copy
import argparse

import pandas as pd
from models.json_dataset import get_test_dataloaders
import numpy as np
import torch

from utils.ckpt_save import get_ckpt_for_eval
from utils.os_util import get_env_variable
from models.json_dataset import data_json_labels_handles
from utils.evaluation import (
    PhaseProgression,
    PhaseClassification,
    KendallsTau,
    WildKendallsTau,
    EnclosedAreaError,
    OnlineGeoProgressError
)
from utils.plotter import validate_folder
from utils.logging import configure_logging_format
from utils.train_util import get_config_for_folder

# GLOBAL VARS
logger = configure_logging_format()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_json_dict_and_save(ckpt_out_folder, test_object, historical_json, update_json):
    if historical_json[test_object.name] is None:
        historical_json[test_object.name] = {key: [value] for key, value in update_json.items()}
    else:
        for key, value in update_json.items():
            historical_json[test_object.name][key].append(value)
    json_to_save = copy.deepcopy(historical_json[test_object.name])
    json_to_save['task'].append('MEAN')
    for k in json_to_save.keys():
        if k != 'task':
            json_to_save[k].append(np.mean(json_to_save[k]) if None not in json_to_save[k] else None)
    pd.DataFrame(json_to_save).to_csv(f'{ckpt_out_folder}/{test_object}.csv')

def begin_eval_loop_over_tasks(config, folder_to_test, tests_to_run, tasks, test_tasks, test_dl_dict):
    ##########################################
    # for each ckpt in this folder....
    ##########################################
    out_folder = f'{folder_to_test}/EVAL'
    validate_folder(out_folder)
    json_test_result = {test_object.name: None for test_object in tests_to_run}

    # If multitask setting, then grab single model
    multi_prong_bool = config["MULTITASK"]
    if multi_prong_bool:
        if config['ARCHITECTURE']['num_heads'] is None:
            num_heads = len(tasks)
        else:
            num_heads = config['ARCHITECTURE']['num_heads']
        model, epoch, ckpt_handle = get_ckpt_for_eval(
            ckpt_parent_folder=folder_to_test,
            config=config,
            num_heads=num_heads,
            device=device
        )

    for task in sorted(test_tasks):
        logger.info(f'\n{"*" * 40}\n{"*" * 40}\n{task}\n{"*" * 40}\n{"*" * 40}')
        
        if not multi_prong_bool:
            if f'{folder_to_test}/{task}' not in glob.glob(folder_to_test + '/*'):
                continue
            taskfolder_to_test = f'{folder_to_test}/{task}'
            model, epoch, ckpt_handle = get_ckpt_for_eval(
                ckpt_parent_folder=taskfolder_to_test,
                config=config,
                device=device
            )
        ckpt_out_folder = f'{out_folder}/{ckpt_handle}'
        validate_folder(ckpt_out_folder)
        ####################
        # loop tests for this ckpt
        ####################
        for test_object in tests_to_run:
            ####################
            # run the test
            ####################
            logger.info(f'** Beginning Test: {test_object} for {folder_to_test.split("/")[-1]}')
            eval_results_dict = test_object(
                model,
                config,
                epoch,
                {task: test_dl_dict[task]},
                folder_to_test,
                [task]
            )
            if eval_results_dict is not None:
                update_json_dict_and_save(
                    ckpt_out_folder=ckpt_out_folder,
                    test_object=test_object,
                    historical_json=json_test_result,
                    update_json=eval_results_dict
                )
            logger.info(f'** Finished Test: {test_object} for {task}')


if __name__ == '__main__':
    ##########################################
    # quick parser code to specify folder to test.
    parser = argparse.ArgumentParser(description='Please specify the parameters of the experiment.')
    parser.add_argument('-f', '--folder', required=True) 
    args = parser.parse_args()

    tests_to_run = {
        EnclosedAreaError(),
        OnlineGeoProgressError(),
        KendallsTau(),
        WildKendallsTau(),
        PhaseClassification(),
        PhaseProgression(),
    }

    dset_json_folder = get_env_variable('JSON_DPATH')
    folder_to_test = args.folder
    logger.info(f'Beginning test suite for {folder_to_test.split("/")[-1]}')

    ##########################################
    # get config!
    config = get_config_for_folder(folder_to_test)

    ##########################################
    # dataset util form saved json
    data_structure = data_json_labels_handles(dset_json_folder, dset_name=config.DATASET_NAME)
    TASKS = data_structure.keys()
    testTASKS = TASKS # edit if you see fit

    ##########################################
    # get all test dataloaders
    test_dataloaders = get_test_dataloaders(
        tasks=testTASKS,
        data_structure=data_structure,
        config=config,
        device=device
    )

    ##########################################
    # print summary information
    ##########################################
    logger.info(f'Model architecture is {config.BASEARCH.ARCHITECTURE}')
    logger.info(f'Dataset is {config.DATASET_NAME}')
    logger.info(f'Test deck is {tests_to_run}')
    logger.info(f'Folder to run is {folder_to_test}')
    logger.info(f'tasks are {TASKS}')
 
    begin_eval_loop_over_tasks(
        config,
        folder_to_test,
        tests_to_run,
        TASKS,
        testTASKS,
        test_dataloaders
    )
