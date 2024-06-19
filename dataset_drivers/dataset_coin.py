import json 
from pprint import pprint as pp
import time
import numpy as np
import traceback
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
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




if __name__ == '__main__':
    with open('/mnt/raptor/datasets/COIN_zijia/annotations/COIN.json', 'r') as f:
        data = dict(json.load(f))['database']
    frames_folder = '/mnt/raptor/datasets/COIN_zijia/annotations/frames_10fps'

    num_videos = len(data)
    num_tasks = max([int(d["recipe_type"]) for v, d in data.items()])
    num_actions = max([int(act_seg["id"]) for v, d in data.items() for act_seg in d["annotation"]])

    logger.info(f'COIN dataset has {num_videos} total videos')
    logger.info(f'Total number of tasks are {num_tasks}')
    logger.info(f'Total number of action types are {num_actions}')

    resnet_obj = Resnet50_embed()
    skip_rate = 5
    get_frame_number = lambda t: int(t / (skip_rate / 10)) - 1

    conv4d_out_folder = './data/coin-skip5/embeddings_resnet50'
    resnet50_out_folder = './data/coin-skip5/embeddings'
    tdict_out_folder = './data/coin-skip5/times'

    # first iterate over each task
    combos = {}
    durations = {}

    for u, (vid_string, details) in enumerate(list(data.items())):
        try:
            task_class = details['class']
            task_conv4d_out_folder = conv4d_out_folder + f'/{task_class}'
            task_resnet50_out_folder = resnet50_out_folder + f'/{task_class}'
            task_tdict_out_folder = tdict_out_folder + f'/{task_class}'

            validate_folder(task_conv4d_out_folder)
            validate_folder(task_resnet50_out_folder)
            validate_folder(task_tdict_out_folder)

            
            video_start_time = float(details['start'])
            video_end_time = float(details['end'])
            video_start_frame = get_frame_number(video_start_time)   
            video_end_frame = get_frame_number(video_end_time)
            tdict = {'step': [], 'start_frame': [], 'end_frame': []}

            prev_time = 0
            for p, segment in enumerate(details['annotation']):
                s = time.time()
                action_id = segment['id']
                label_name = segment['label']
                start_frame = get_frame_number(float(segment['segment'][0])) - video_start_frame
                end_frame = get_frame_number(float(segment['segment'][1])) - video_start_frame

                if start_frame > prev_time:
                    tdict['step'].append('SIL')
                    tdict['start_frame'].append(prev_time)
                    tdict['end_frame'].append(start_frame-1)
                tdict['step'].append(label_name)
                tdict['start_frame'].append(start_frame)
                tdict['end_frame'].append(end_frame)
                prev_time = end_frame+1
            
            if prev_time < video_end_frame - video_start_frame-1:
                tdict['step'].append('SIL')
                tdict['start_frame'].append(prev_time)
                tdict['end_frame'].append(video_end_frame - video_start_frame-1)

            if task_class in durations.keys():
                durations[task_class].append(tdict['end_frame'][-1])
            else:
                durations[task_class] = [tdict['end_frame'][-1]]
            
            result = load_jpg_images_to_array(f'{frames_folder}/{vid_string}', skip_rate=skip_rate, first_frame=video_start_frame, last_frame=video_end_frame)
            conv4d_output = resnet_obj.get_output(result, conv4d=True)
            resnet_output = resnet_obj.get_output(result)

            np.save(f'{task_conv4d_out_folder}/{vid_string}.npy', conv4d_output)
            np.save(f'{task_resnet50_out_folder}/{vid_string}.npy', resnet_output)
            pd.DataFrame(tdict).to_csv(f'{task_tdict_out_folder}/{vid_string}.csv')
            logger.info(f'id({vid_string}), {task_class} | length is {resnet_output.shape[0]} | {time.time() - s:.2f} | prog({(u)/len(data.items()):.5f})')
        except Exception as e:
            print(f"ISSUE WITH {vid_string}")
            traceback.print_exc()
            continue


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