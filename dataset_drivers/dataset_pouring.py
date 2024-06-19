import json 
from pprint import pprint as pp
import time
import numpy as np
import traceback
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy import stats as st
from utils.logging import configure_logging_format
import distance
from itertools import combinations
from ego_get_frames import load_jpg_images_to_array
import torch
from torchvision import models
from dataset_drivers.imagenet_embed import Resnet50_embed
from utils.plotter import validate_folder
logger = configure_logging_format()

# this is from /mnt/raptor/datasets/pouring/pouring_demonstrations/segmentation_groundtruth.py
segmentation_groundtruth = {
    'pouring_001': {'2': [32], '3': [8, 40]},
    'pouring_004': {'2': [27], '3': [11, 32]},
    'pouring_007': {'2': [23], '3': [11, 26]},
    'pouring_009': {'2': [32], '3': [9, 39]},
    'pouring_012': {'2': [20], '3': [7, 23]},
    'pouring_016': {'2': [24], '3': [12, 28]},
    'pouring_002': {'2': [25], '3': [11, 35]},
    'pouring_003': {'2': [32], '3': [12, 38]},
    'pouring_005': {'2': [27], '3': [10, 32]},
    'pouring_006': {'2': [21], '3': [10, 27]},
    'pouring_008': {'2': [18], '3': [10, 23]},
    'pouring_010': {'2': [22], '3': [9, 30]},
    'pouring_011': {'2': [20], '3': [10, 27]},
    'pouring_013': {'2': [22], '3': [10, 29]},
    'pouring_014': {'2': [20], '3': [8, 25]},
    'pouring_015': {'2': [25], '3': [9, 35]},
    'pouring_017': {'2': [18], '3': [9, 20]},
}

if __name__ == '__main__':
    frames_folder = '/mnt/raptor/datasets/pouring/pouring_demonstrations/frames2'


    resnet_obj = Resnet50_embed()

    conv4d_out_folder = './data/pouring/embeddings_resnet50'
    resnet50_out_folder = './data/pouring/embeddings'
    tdict_out_folder2 = './data/pouring/times'
    tdict_out_folder3 = './data/pouring/times_three'
    task_class = 'just_pouring'
    task_conv4d_out_folder = conv4d_out_folder + f'/{task_class}'
    task_resnet50_out_folder = resnet50_out_folder + f'/{task_class}'
    task_tdict_out_folder2 = tdict_out_folder2 + f'/{task_class}'
    task_tdict_out_folder3 = tdict_out_folder3 + f'/{task_class}'
    validate_folder(task_conv4d_out_folder)
    validate_folder(task_resnet50_out_folder)
    validate_folder(task_tdict_out_folder2)
    validate_folder(task_tdict_out_folder3)

    def get_tdict(time_list, vid_len):
        tdict = {'step': [], 'start_frame': [], 'end_frame': []}
        tdict['step'].append(f'step_{0}')
        tdict['start_frame'].append(0)
        for i in range(len(time_list)):
            tdict['step'].append(f'step_{i+1}')
            tdict['start_frame'].append(time_list[i])
            tdict['end_frame'].append(time_list[i]-1)
        tdict['end_frame'].append(vid_len-1)
        return tdict

    # first iterate over each task
    combos = {}
    durations = {}
    for jpg_folder in glob.glob(frames_folder + '/*'):
        s = time.time()
        vid_string = jpg_folder.split('/')[-1]
        result = load_jpg_images_to_array(f'{frames_folder}/{vid_string}', skip_rate=1)
        print(vid_string, result.shape)
        three_time_list = segmentation_groundtruth[vid_string]['3']
        two_time_list = segmentation_groundtruth[vid_string]['2']
        tdict_2 = get_tdict(two_time_list, result.shape[0])
        tdict_3 = get_tdict(three_time_list, result.shape[0])
        
        conv4d_output = resnet_obj.get_output(result, conv4d=True)
        resnet_output = resnet_obj.get_output(result)

        # saving
        np.save(f'{task_conv4d_out_folder}/{vid_string}.npy', conv4d_output)
        np.save(f'{task_resnet50_out_folder}/{vid_string}.npy', resnet_output)
        pd.DataFrame(tdict_2).to_csv(f'{task_tdict_out_folder2}/{vid_string}.csv')
        pd.DataFrame(tdict_3).to_csv(f'{task_tdict_out_folder3}/{vid_string}.csv')

        logger.info(f'id({vid_string}), {task_class} | length is {resnet_output.shape[0]} | {time.time() - s:.2f}')


    #     recipe_type = details['recipe_type']
    #     action_sequence = [int(act_obj['id']) for act_obj in details['annotation']]
    #     amount_of_noise = sum([int(act_obj['segment'][1]) - int(act_obj['segment'][0]) for act_obj in details['annotation']])
    #     if recipe_type not in combos.keys():
    #         combos[recipe_type] = {
    #             'class': details['class'], 
    #             'seqs':[action_sequence], 
    #             'durations': [float(details['duration'])], 
    #             'noise_percent': [amount_of_noise / float(details['duration'])], 
    #             'graph': nx.DiGraph()
    #         }
    #     else:
    #         combos[recipe_type]['seqs'].append(action_sequence)
    #         combos[recipe_type]['durations'].append(float(details['duration']))
    #         combos[recipe_type]['noise_percent'].append(amount_of_noise / float(details['duration']))

    # data_details = {
    #     'task_id': [], # english name of the task
    #     'class': [], # english name of the task
    #     'num_vids': [], # number of total vids for this task
    #     'num_action_steps': [], # number of total vids for this task
    #     'majority_endstate_percent': [], # percent of vids that end in the most common end state
    #     'num_variations': [], # number of sequential variations in task completion
    #     'self_loop_percent': [], # percent of total edges that self-loop (A.K.A repeatable step percentage)
    #     'other_loop_percent': [], # percent of total edges that are bidirectional (A.K.A alternating step percentage)
    #     'avg_duration': [], # average seconds for videos in this task
    #     'avg_percent_noise': [], # average seconds for videos in this task
    #     'edit_distance': [], # average edit distance between seqs
    # }
    # for task, details in combos.items():
    #     # Create graph
    #     G = combos[task]['graph']
    #     nodes = list(set(flatten(combos[task]['seqs'])))
    #     for node in nodes:
    #         G.add_node(int(node))
    #     for seq in combos[task]['seqs']:
    #         for i in range(len(seq)-1):
    #             G.add_edge(int(seq[i]), int(seq[i+1]))
    #     data_details['task_id'].append(task)
    #     data_details['class'].append(details['class'])
    #     data_details['num_vids'].append(len(details['seqs']))
    #     data_details['num_action_steps'].append(len(nodes))
    #     data_details['avg_duration'].append(np.mean(details['durations']))
    #     data_details['majority_endstate_percent'].append(st.mode([seq[-1] for seq in details['seqs']]).count[0] / len(details['seqs']))
    #     data_details['num_variations'].append(len(set(["->".join([str(step) for step in seq]) for seq in details['seqs']])))
    #     data_details['self_loop_percent'].append(sum([0 if n1 != n2 else 1 for n1, n2 in G.edges]) / len(nodes))
    #     data_details['other_loop_percent'].append(sum([ 1 if (n1,n2) in G.edges() and (n2,n1) in G.edges() else 0 for n1 in G.nodes() for n2 in G.nodes() if n1 != n2]) / (2 * len(G.edges())))
    #     data_details['avg_percent_noise'].append(np.mean(details['noise_percent']))
    #     data_details['edit_distance'].append(calculate_edit_distance(details['seqs']))

    # pd.DataFrame(data_details).to_csv('./data/coin/coin_stats_editdistance.csv')

    # # for key in combinations.keys():
    # #     print(key, combinations[key]['graph'].nodes())
    # #     print_graph(combinations[key]['graph'], f'COINGRAPH-{key}.png')

# def calculate_edit_distance(sequences):
#     num_sequences = len(sequences)
#     edit_distances = [[0] * num_sequences for _ in range(num_sequences)]

#     for i, j in combinations(range(num_sequences), 2):
#         seq1 = sequences[i]
#         seq2 = sequences[j]
#         edit_distances[i][j] = distance.levenshtein(seq1, seq2)
#         edit_distances[j][i] = edit_distances[i][j]

#     return np.mean(np.array(edit_distances))
# def flatten(l):
#     return [item for sublist in l for item in sublist]
# def print_graph(graph, filename):
#     plt.clf()
#     pos = nx.spring_layout(graph)
#     nx.draw(graph, pos, node_size=1500, node_color='blue', font_size=8, font_weight='bold', with_labels=True)
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('./plots/graphs/' + filename, format="PNG") 