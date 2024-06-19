import os
import random
import shutil
import matplotlib.pyplot as plt

from scipy.stats import kendalltau
from external_util.ot_pytorch import sink
import torch.nn as nn
from models.json_dataset import data_json_labels_handles
from utils.os_util import get_env_variable
from utils.plotter import validate_folder
from utils.trainers import train_and_evaluate_svm, train_linear_regressor, svm_normalize_embedded_dl
from utils.tensorops import compute_eae_between_dict_vids, contains_non_float_values, flatten_dataloader_and_get_dict, get_average_train_cum_distance, get_cum_matrix, get_target_alignment_with_dict, get_trueprogress, preprocess_batch
from utils.logging import configure_logging_format
import torch
import numpy as np

logger = configure_logging_format()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhaseProgression:
    def __init__(self) -> None:
        self.name = 'phaseprogression'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False):
        for task in tasks:
            embedded_dl = flatten_dataloader_and_get_dict(model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu')
            if normalize:
                embedded_dl = svm_normalize_embedded_dl(embedded_dl=embedded_dl)
            for i, (outputs_dict, tdict) in enumerate(embedded_dl):
                outputs = outputs_dict['outputs']
                X = []
                y = []
                if contains_non_float_values(outputs):
                    continue
                true_prog = get_trueprogress(tdict).detach().cpu().numpy()
                # true_prog = np.arange(o.shape[0]) / (o.shape[0] - 1)
                outputs = outputs.detach().cpu().numpy()
                for k, frame in enumerate(outputs):
                    X.append(frame)
                    y.append(true_prog[k])
            _, r2 = train_linear_regressor(X, y, normalize=normalize)
            r2 = np.mean([max(0, train_linear_regressor(X, y)[1]) for _ in range(200) ])
            if len(tasks) == 1:
                return {'task': task, 'phase_prog': r2}

class PhaseClassification:
    def __init__(self) -> None:
        self.name = 'phaseclassification'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False):
        percents = [.1, .5, 1]
        df = {'task': []}
        for percent in percents:
            df[f'phase_classification_ovo_{percent}'] = []

        for task in tasks:
            # first get set of all actions in the test set
            action_set = set()
            num_videos = 0
            num_data_points = 0
            embedded_dl = flatten_dataloader_and_get_dict(model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu')
            if normalize:
                embedded_dl = svm_normalize_embedded_dl(embedded_dl=embedded_dl)
            for i, (_, tdict) in enumerate(embedded_dl):
                num_videos += 1
                list(map(action_set.add, tdict['step']))
                num_data_points += tdict['end_frame'][-1]+1
            id_to_action = dict(list(zip(range(len(action_set)), action_set)))
            action_to_id = {v: k for k, v in id_to_action.items()}

            X_set = []
            Y_set = []
            for i, (outputs_dict, tdict) in enumerate(embedded_dl):
                outputs = outputs_dict['outputs']
                for t in range(outputs.shape[0]):
                    X = outputs[t]
                    Y = None
                    for step, start, end in zip(tdict['step'], tdict['start_frame'], tdict['end_frame']):
                        if start <= t <= end:
                            Y = action_to_id[step]
                    if not contains_non_float_values(X) and Y is not None:
                        X_set.append(X.clone().detach().cpu().numpy())
                        Y_set.append(Y)
            # shuffle
            if len(X_set) == 0:
                return None

            x_y_pairs = list(zip(X_set, Y_set))
            random.shuffle(x_y_pairs)
            X_set_r, Y_set_r = map(list, list(zip(*x_y_pairs)))

            if contains_non_float_values(X_set_r):
                print("X_set_r")
                exit(1)
            # print(Y_set_r)
            if contains_non_float_values(Y_set_r):
                print(Y_set_r)
                exit(1)
            df['task'] = task
            for percent in percents:
                barrier = round(percent * len(X_set_r))
                trained_classifier, accuracy = train_and_evaluate_svm(X_set_r[:barrier], Y_set_r[:barrier], X_set_r, Y_set_r, normalize=normalize)
                df[f'phase_classification_ovo_{percent}'] = accuracy

            if len(tasks) == 1:
                return df

class KendallsTau:
    def __init__(self) -> None:
        self.name = 'KT'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False):
        for task in tasks:
            sci_ktau_list = []
            embedded_dl = flatten_dataloader_and_get_dict(model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu')
            if normalize:
                embedded_dl = svm_normalize_embedded_dl(embedded_dl=embedded_dl)
            for i, (odict1, _) in enumerate(embedded_dl):
                v1 = odict1['outputs']
                N = v1.shape[0]
                for j, (odict2, _) in enumerate(embedded_dl):
                    if i == j:
                        continue
                    v2 = odict2['outputs']
                    neighbors = torch.cdist(v1, v2).argmin(dim=1).detach().cpu().numpy()
                    ktau_sci = kendalltau(neighbors, np.arange(N)).statistic
                    sci_ktau_list.append(ktau_sci)

            if len(tasks) == 1:
                return {'task': task, 'sci-KT': np.mean(sci_ktau_list)}

class WildKendallsTau:
    def __init__(self) -> None:
        self.name = 'AKT'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False):
        for task in tasks:
            ktau_list = []
            sci_ktau_list = []
            for (inputs, times) in test_dataloaders[task]:
                inputs, times = preprocess_batch(inputs, times, skip_rate=config_obj.SKIP_RATE)
                outputs_dict = model(inputs)
                del inputs
                outputs = outputs_dict['outputs']
                for i, (v1, t1) in enumerate(zip(outputs, times)):
                    N = v1.shape[0]
                    for j, (v2, t2) in enumerate(zip(outputs, times)):
                        if i == j:
                            continue
                        neighbors = torch.cdist(v1, v2).argmin(dim=1).detach().cpu().numpy()
                        true_alignment = get_target_alignment_with_dict(v1.shape[0], v2.shape[0], t1, t2).to(device)
                        indices = np.nonzero(true_alignment).clone().cpu().numpy().T
                        align_options = {}
                        for a, b in zip(indices[0], indices[1]):
                            if a not in align_options.keys():
                                align_options[a] = [b]
                            else:
                                align_options[a].append(b)
                        concord_pairs = 0
                        disconcord_pairs = 0
                        for i in range(N-1):
                            if i not in align_options.keys():
                                continue #SIL
                            p_options = align_options[i]
                            for j in range(i+1, N):
                                if j not in align_options.keys():
                                    continue #SIL
                                q = neighbors[j]
                                concord = False
                                for p in p_options:
                                    if p < q:
                                        concord = True
                                        continue
                                if concord:
                                    concord_pairs += 1
                                else:
                                    disconcord_pairs += 1

                        ktau = (concord_pairs - disconcord_pairs) / ((N * (N-1)) / 2)
                        ktau_list.append(ktau)
            if len(tasks) == 1:
                return {'task': task, 'my-KT': np.mean(ktau_list)}

class EnclosedAreaError:
    def __init__(self) -> None:
        self.name = 'eae'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False):
        results = {'task': [], 'eae': []}
        for task in tasks:
            eae_list = []
            embedded_dl = flatten_dataloader_and_get_dict(model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu')
            if normalize:
                embedded_dl = svm_normalize_embedded_dl(embedded_dl=embedded_dl)
            for i, (odict1, t1) in enumerate(embedded_dl):
                v1 = odict1['outputs']
                for j, (odict2, t2) in enumerate(embedded_dl):
                    if i == j:
                        continue
                    v2 = odict2['outputs']
                    eae = compute_eae_between_dict_vids(v1, v2, t1, t2)
                    if eae is not None:
                        eae_list.append(eae)

            results['task'].append(task)
            results['eae'].append(np.mean(eae_list))
            if len(tasks) == 1:
                return {'task': task, 'eae': np.mean(eae_list)}

class OnlineGeoProgressError:
    def __init__(self) -> None:
        self.name = 'ogpe'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False, plotting=False):
        plot_folder = testfolder  + '/plotting_progress'
        validate_folder(plot_folder)

        for task in tasks:
            taskplot = plot_folder + f'/{task}'
            validate_folder(taskplot)
            dset_json_folder = get_env_variable('JSON_DPATH')
            data_structure = data_json_labels_handles(dset_json_folder, dset_name=config_obj['DATASET_NAME'])
            if os.path.isdir(testfolder + '/ckpt'):
                train_cum_means, train_cum_vars = get_average_train_cum_distance(model, testfolder, data_structure, targ_task=task, skip_rate=config_obj.SKIP_RATE)
            elif os.path.isdir(testfolder + f'/{task}/ckpt'):
                train_cum_means, train_cum_vars = get_average_train_cum_distance(model, testfolder + f'/{task}', data_structure, targ_task=task, skip_rate=config_obj.SKIP_RATE)

            if None in [train_cum_means, train_cum_vars]:
                return None
            if task not in train_cum_means.keys():
                print(f'BTW {task} not in tasks')
                return None

            gpe_list = []
            embedded_dl = flatten_dataloader_and_get_dict(model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu')
            if normalize:
                embedded_dl = svm_normalize_embedded_dl(embedded_dl=embedded_dl)
            for i, (outputs_dict, tdict) in enumerate(embedded_dl):
                outputs = outputs_dict['outputs']
                tdict['end_frame'][-1] = outputs.shape[0]-1
                true_progress = get_trueprogress(tdict)
                pred2_progress = get_cum_matrix(outputs)
                # pred2_progress = pred2_progress / pred2_progress.max() 

                if pred2_progress.sum() == 0:
                    continue
                pred2_progress = pred2_progress / train_cum_means[task]
                gpe = torch.mean(torch.abs(true_progress - pred2_progress))
                gpe_list.append(gpe.item())

                if plotting:
                    plt.clf()
                    a = true_progress.detach().cpu().numpy()
                    b = pred2_progress.detach().cpu().numpy()
                    plt.plot(a, color='green', label='Ground Truth Progress')
                    plt.plot(b, color='blue', label='Online Progress Estimate')
                    plt.fill_between(range(len(a)), a, b, color='red', alpha=0.5, label='Error')
                    # Adjust the font size and frequency of x-axis ticks
                    plt.xticks(np.arange(0, len(a), step=pred2_progress.shape[0] // 5), fontsize=10)

                    # Adjust the font size of y-axis ticks
                    plt.yticks(fontsize=10)

                    # plt.title(f'{loss_plot_name} Progress Plot (EgoProcel - Making Brownies)')
                    plt.xlabel('Time', fontsize=15)  # Adjust the font size for x-axis label
                    plt.ylabel('Progress', fontsize=15)  # Adjust the font size for y-axis label

                    name = ".".join(outputs_dict['name'].split('/')[-1].split('.')[:-1])
                    plt.legend()
                    plt.savefig(f'{taskplot}/{name}.pdf')
            return {'task': task, 'ogpe': np.mean(gpe_list), 'CoV': train_cum_vars[task] / train_cum_means[task]}

