import json 
from pprint import pprint as pp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as st
import glob
import os
from utils.logging import configure_logging_format
from dataset_coin import print_graph

import distance
from itertools import combinations
logger = configure_logging_format()

annotations = '/mnt/raptor/datasets/EgoProcel_zijia/annotations'

csvs = {'annotations': {'csvs': [], 'dirs': {}}}


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
    '/CMU-MMAC/Pizza': lambda x: 'Pizza' in x,
    '/CMU-MMAC/Brownie': lambda x: 'Brownie' in x,
    '/CMU-MMAC/Eggs': lambda x: 'Eggs' in x,
    '/CMU-MMAC/Salad': lambda x: 'Salad' in x,
    '/MECCANO': lambda x: x.split('/')[-1].startswith('0'),
    '/pc_assembly': lambda x: x.split('/')[-1] in pc_assemble_files,
    '/pc_disassembly': lambda x: x.split('/')[-1] in pc_disassemble_files,
}

def get_avg_numframes(key):
    folder = '/mnt/raptor/datasets/EgoProcel_zijia/formatted/features/'
    num_frames = []
    for features in glob.glob(folder + '*'):
        # if 'Head' in features:
        #     print(features)
        if conditions[key](features):
            num_frames.append(np.load(features, mmap_mode='r').shape[0])
    
    return np.mean(num_frames), np.std(num_frames)
    

def string_histogram(strings_list):
    # Create an empty dictionary to store the count of each unique string
    histogram = {}

    # Count the occurrences of each string in the list
    for string in strings_list:
        if string in histogram.keys():
            histogram[string] += 1
        else:
            histogram[string] = 1

    # Print the histogram
    for i, (string, count) in enumerate(histogram.items()):
        print(f"Variation {i} ({count} total): {string}")

def print_the_steps(steps):
    new_steps = []
    for i, step_set in enumerate(steps):
        cur_step = step_set[0]
        these_steps = [cur_step]
        for step in step_set[1:]:
            if step == cur_step:
                continue
            else:
                cur_step = step
                these_steps.append(cur_step)
        new_steps.append(these_steps)
    string_histogram(["->".join([str(step) for step in seq]) for seq in new_steps])
    string_steps = set(["->".join([str(step) for step in seq]) for seq in new_steps])
    new_steps = [[int(l) for l in step_set.split('->')] for step_set in string_steps]


    return new_steps


def calculate_edit_distance(sequences):
    print(sequences)
    num_sequences = len(sequences)
    edit_distances = [[0] * num_sequences for _ in range(num_sequences)]

    for i, j in combinations(range(num_sequences), 2):
        seq1 = sequences[i]
        seq2 = sequences[j]
        edit_distances[i][j] = distance.levenshtein(seq1, seq2) / max(len(seq1), len(seq2))
        edit_distances[j][i] = edit_distances[i][j]

    return np.mean(np.array(edit_distances))


def recursive_csv_search(filepath, dict_to_add):
    for item in glob.glob(filepath + '/*'):
        filename = item.split('/')[-1]
        if os.path.isdir(item):
            dict_to_add['dirs'][filename] = {'csvs': [], 'dirs': {}}
            recursive_csv_search(item, dict_to_add=dict_to_add['dirs'][filename])
        elif item.endswith('csv'):
            dict_to_add['csvs'].append(filename)


def get_CMU_MMAC_csvs(food):
    assert food in ['Salad', 'Eggs', 'Brownie', 'Pizza', 'Sandwich']
    subfolders = csvs['dirs']['CMU-MMAC']['dirs'][food]['dirs']
    list_csvs = []
    for subfolder in subfolders.keys():
        list_csvs.append(glob.glob("/".join([annotations, 'CMU-MMAC', food, subfolder]) + '/*')[0])
    return list_csvs

def read_spreadsheets(csv_list):
    convert_actid = lambda x: int(float(x.split(" ")[0]))
    G = nx.DiGraph()
    durations = []
    percent_noises = []
    all_act_steps = []
    for ff in csv_list:
        csvFile = pd.read_csv(ff, header=None)
        rows = list(csvFile.iterrows())
        act_steps = []
        active_duration = 0
        duration = float(rows[-1][1][1]) - float(rows[0][1][0])
        for i, line in enumerate(range(len(rows))):
            idx = rows[line][0]
            start, end, act = rows[line][1][:3]
            act_name = " ".join(act.split(" ")[1:])
            # if '.' in act.split(" ")[0]:
            #     print(csvFile)
            #     print(float(act.split(" ")[0]))
            #     exit(1)
            act_id = convert_actid(act)
            
            G.add_node(act_id)
            if i < len(rows)-1:
                G.add_edge(act_id, convert_actid(rows[line+1][1][2]))
            
            act_steps.append(act_id)
            active_duration += float(end) - float(start)
        durations.append(duration)
        percent_noises.append(1 - active_duration / duration)
        all_act_steps.append(act_steps)
    avg_duration = np.mean(durations)
    avg_percent_noise = np.mean(percent_noises)
    other_loop_percent = sum([ 1 if (n1,n2) in G.edges() and (n2,n1) in G.edges() else 0 for n1 in G.nodes() for n2 in G.nodes() if n1 != n2]) / (2 * len(G.edges()))
    self_loop_percent = sum([0 if n1 != n2 else 1 for n1, n2 in G.edges]) / len(G.nodes())
    set_var = set(["->".join([str(step) for step in seq]) for seq in all_act_steps])
    num_variations = len(set_var)
    all_variations = all_act_steps
    edit_distance = None
    majority_endstate_percent = st.mode(np.array([int(seq[-1]) for seq in all_act_steps]), keepdims=True).count[0] / len(all_act_steps)
    return majority_endstate_percent, num_variations, self_loop_percent, other_loop_percent, avg_duration, avg_percent_noise, G, edit_distance, all_variations

def get_details(prev_level, dirs_dict):
    for top_level in dirs_dict.keys():
        this_level = '/'.join([prev_level, top_level])
        this_dirs_dict = dirs_dict[top_level]['dirs']
        recurse = True
        
        logger.info(f'*** Testing {top_level} data')
        spreadsheets = [annotations + prev_level + '/' + top_level + '/' + name for name in dirs_dict[top_level]['csvs']]
        
        # Handle CMAC
        if any([food == top_level for food in ['Salad', 'Eggs', 'Brownie', 'Pizza', 'Sandwich']]):
            spreadsheets = get_CMU_MMAC_csvs(top_level)
            recurse = False
        if any([food == top_level for food in []]):
            spreadsheets = get_CMU_MMAC_csvs(top_level)
            recurse = False

        if len(spreadsheets) > 0:
            majority_endstate_percent, num_variations, self_loop_percent, other_loop_percent, avg_duration, avg_percent_noise, G, edit_distance, all_variations = read_spreadsheets(spreadsheets)
            graphs[this_level] = G
            data_details['class'].append(this_level)
            data_details['num_vids'].append(len(spreadsheets))
            data_details['num_action_steps'].append(len(G.nodes()))
            data_details['avg_duration'].append(avg_duration)
            data_details['num_frames_mean'].append(get_avg_numframes(this_level)[0])
            data_details['num_frames_std'].append(get_avg_numframes(this_level)[1])
            data_details['majority_endstate_percent'].append(majority_endstate_percent)
            print(f'\n\n***************** {this_level}')
            variation_no_repeat = print_the_steps(all_variations)
            data_details['num_variations'].append(len(variation_no_repeat))
            data_details['self_loop_percent'].append(self_loop_percent)
            data_details['other_loop_percent'].append(other_loop_percent)
            data_details['avg_percent_noise'].append(avg_percent_noise)
            data_details['edit_distance'].append(calculate_edit_distance(variation_no_repeat))

        elif recurse:
            get_details(this_level, this_dirs_dict)
            
if __name__ == '__main__':
    graphs = {}
    data_details = {
        'class': [], # english name of the task
        'num_vids': [], # number of total vids for this task
        'num_action_steps': [], # number of total vids for this task
        'majority_endstate_percent': [], # percent of vids that end in the most common end state
        'num_variations': [], # number of sequential variations in task completion
        'self_loop_percent': [], # percent of total edges that self-loop (A.K.A repeatable step percentage)
        'other_loop_percent': [], # percent of total edges that are bidirectional (A.K.A alternating step percentage)
        'avg_duration': [], # average seconds for videos in this task
        'num_frames_mean': [], # average seconds for videos in this task
        'num_frames_std': [], # average seconds for videos in this task
        'avg_percent_noise': [], # average seconds for videos in this task
        'edit_distance': [], # average seconds for videos in this task
    }
    csvs = {'csvs': [], 'dirs': {}}
    recursive_csv_search(annotations, csvs)

    get_details('', csvs['dirs'])

    pd.DataFrame(data_details).to_csv('./data/egoprocel/egoprocel_stats_editdistance.csv')

    # for key, graph in graphs.items():
    #     print_graph(graph, f'egoprocel/{key.replace("/", "->")[1:] if "/" in key else key}.png')
