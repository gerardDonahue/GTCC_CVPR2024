import numpy as np
import glob
import os
import pandas as pd
from pprint import pprint as pp
import shutil
from utils.logging import configure_logging_format
from utils.plotter import validate_folder
logger = configure_logging_format()

annotations_folder = '/mnt/raptor/datasets/EgoProcel_zijia/annotations'

files_r3d_folder = '/mnt/raptor/datasets/EgoProcel_zijia/formatted/r3d'
files_groundtruths_folder = '/mnt/raptor/datasets/EgoProcel_zijia/formatted/groundTruth'
output_times_folder = './data/egoprocel/times'
output_embeddings_folder = './data/egoprocel/embeddings'
validate_folder(output_times_folder)
validate_folder(output_embeddings_folder)

DOWNSAMPLE_RATE = 15

pc_disassemble_files = ['Head_10.npy',  'Head_12.npy',  'Head_15.npy',  'Head_17.npy',  'Head_19.npy',  'Head_20.npy',  'Head_22.npy',  'Head_25.npy',  'Head_27.npy',  'Head_29.npy',  'Head_30.npy',  'Head_31.npy',  'Head_33.npy',  'Head_6.npy',  'Head_8.npy',]
pc_assemble_files = ['Head_11.npy',  'Head_13.npy',  'Head_14.npy',  'Head_16.npy',  'Head_18.npy',  'Head_21.npy',  'Head_24.npy',  'Head_26.npy',  'Head_28.npy',  'Head_32.npy',  'Head_35.npy',  'Head_5.npy',  'Head_7.npy',  'Head_9.npy',]
conditions = {
    '/EGTEA_Gaze+/GreekSalad': lambda x: 'GreekSalad' in x,
    '/EGTEA_Gaze+/ContinentalBreakfast': lambda x: 'ContinentalBreakfast' in x,
    '/EGTEA_Gaze+/Pizza': lambda x: 'Pizza' in x,
    '/EGTEA_Gaze+/PastaSalad': lambda x: 'PastaSalad' in x,
    '/EGTEA_Gaze+/TurkeySandwich': lambda x: 'TurkeySandwich' in x,
    '/EGTEA_Gaze+/Cheeseburger': lambda x: 'Cheeseburger' in x,
    '/EGTEA_Gaze+/BaconAndEggs': lambda x: 'BaconAndEggs' in x,
    '/EPIC-Tents': lambda x: 'tent' in x,
    '/CMU-MMAC/Sandwich': lambda x: 'Brownie' in x,
    '/CMU-MMAC/Pizza': lambda x: 'Pizza' in x and x.startswith('OP'),
    '/CMU-MMAC/Brownie': lambda x: 'Brownie' in x,
    '/CMU-MMAC/Eggs': lambda x: 'Eggs' in x,
    '/CMU-MMAC/Salad': lambda x: 'Salad' in x,
    '/MECCANO': lambda x: x.split('/')[-1].startswith('0'),
    '/pc_assembly': lambda x: x.split('/')[-1] in pc_assemble_files,
    '/pc_disassembly': lambda x: x.split('/')[-1] in pc_disassemble_files,
}

def get_CMU_MMAC_csvs(food):
    assert food in ['Salad', 'Eggs', 'Brownie', 'Pizza', 'Sandwich']
    subfolders = csvs['dirs']['CMU-MMAC']['dirs'][food]['dirs']
    list_csvs = []
    for subfolder in subfolders.keys():
        list_csvs.append(glob.glob("/".join([annotations, 'CMU-MMAC', food, subfolder]) + '/*')[0])
    return list_csvs

def recursive_csv_search(filepath, dict_to_add):
    for item in glob.glob(filepath + '/*'):
        filename = item.split('/')[-1]
        if os.path.isdir(item):
            dict_to_add['dirs'][filename] = {'csvs': [], 'dirs': {}}
            recursive_csv_search(item, dict_to_add=dict_to_add['dirs'][filename])
        elif item.endswith('csv'):
            dict_to_add['csvs'].append(filename)

def get_details(prev_level, dirs_dict):
    for top_level in dirs_dict.keys():
        this_level = '.'.join([prev_level, top_level]) if len(prev_level) > 0 else top_level
        this_dirs_dict = dirs_dict[top_level]['dirs']
        recurse = True
        
        spreadsheets = [annotations + prev_level + '/' + top_level + '/' + name for name in dirs_dict[top_level]['csvs']]
        if any([food == top_level for food in ['Salad', 'Eggs', 'Brownie', 'Pizza', 'Sandwich']]) and 'CMU' in prev_level:
            spreadsheets = get_CMU_MMAC_csvs(top_level)
            recurse = False
        if any([food == top_level for food in []]):
            spreadsheets = get_CMU_MMAC_csvs(top_level)
            recurse = False
        
        if len(spreadsheets) > 0:
            logger.info(f'{this_level}, {len(spreadsheets)}')
            if this_level == 'EGTEA_Gaze+.Pizza':
                copy_times_vids_to_output_folder(task=this_level, spreadsheets=spreadsheets)
        elif recurse:
            get_details(this_level, this_dirs_dict)


def get_tdict_from_txt_file(txt_file, dsample_vidlength):
    tdict = {'step':[], 'start_frame':[], 'end_frame':[]}
    t = 0
    prev_action = None
    with open(txt_file, 'r') as file:
        for line_no, line in enumerate(file):
            if line_no % DOWNSAMPLE_RATE == 0:
                line = line.replace(" ", "").replace("\n", "")
                if line != prev_action:
                    tdict['step'].append(line)
                    tdict['start_frame'].append(t)
                    if t> 0:
                        tdict['end_frame'].append(t)
                    prev_action = line
                else:
                    pass
                t += 1
    tdict['end_frame'].append(dsample_vidlength - 1)
    return tdict



def copy_times_vids_to_output_folder(task, spreadsheets):
    # make the task folder !
    validate_folder(output_times_folder + f'/{task}')
    validate_folder(output_embeddings_folder + f'/{task}')

    for spreadsheet in spreadsheets:
        # GET VIDEO NAME
        video_name = '.'.join(spreadsheet.split('/')[-1].split('.')[:-1])
        if task == 'EPIC-Tents':
            video_name = '.'.join(spreadsheet.split('/')[-1].split('.')[:-3])

        # get embedding and times file
        r3d_file = files_r3d_folder + f'/{video_name}.npy'
        groundTruth_file = files_groundtruths_folder + f'/{video_name}.txt'

        # copy proper files if they exist
        if not os.path.isfile(r3d_file) or not os.path.isfile(groundTruth_file):
            print('FALSE:   ', video_name, task)
            exit(1)
        elif task == 'EGTEA_Gaze+.Pizza':
            # first, copy the features. No need to edit anything for this
            video = np.load(r3d_file)
            dsample = video[::DOWNSAMPLE_RATE]
            tdict = get_tdict_from_txt_file(groundTruth_file, dsample_vidlength=dsample.shape[0])
            np.save(output_embeddings_folder + f'/{task}/{video_name}.npy', dsample)
            pd.DataFrame(tdict).to_csv(output_times_folder + f'/{task}/{video_name}.csv')
    
        

annotations = '/mnt/raptor/datasets/EgoProcel_zijia/annotations'
csvs = {'csvs': [], 'dirs': {}}
recursive_csv_search(annotations, csvs)
get_details('', csvs['dirs'])
