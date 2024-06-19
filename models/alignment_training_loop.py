import json
import random
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models.model_multiprong import logger
from utils.train_util import save_dict_to_json_file
from utils.ckpt_save import ckpt_save
from utils.plotter import validate_folder
from utils.tensorops import preprocess_batch, contains_non_float_values
from utils.loss_functions import TCC_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def alignment_training_loop(
        model,
        train_dl_dict,
        loss_fn: callable,
        foldername: str,
        CONFIG,
        GPU_intensive=False,
        test_dl_dict=None
    ):
    """this takes the config for model training and executes the training job

    Args:
        model (nn.Module): pytorch model to train. must take (input_batch, task) and output a return dict
        train_dl_dict (dict): train split {t: DL for t in tasks}
        loss_fn (callable): give (input, times, epoch) to get loss
        foldername (str): folderpath for plotting and documentation
        CONFIG (easydict): config EASY dict that follows format of ./configs.
        GPU_intensive (bool): whether to be sparing with GPU memory or not

    Returns:
        None
    """
    logger.info(model)

    #################################
    ### write the config dict for documentation
    num_epochs = CONFIG.NUM_EPOCHS
    learning_rate = CONFIG.LEARNING_RATE
    save_dict_to_json_file(CONFIG, foldername + '/config.json')

    #################################
    ### optimizer variable
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate
    )

    #################################
    ### get checkpointing folder
    ckpt_folder = foldername + '/' + 'ckpt'
    validate_folder(ckpt_folder)

    ckpt_save(
        model_t=model,
        optimizer_t=optimizer,
        epoch_t=0,
        loss_t=10000000,
        filename=ckpt_folder + f'/epoch-0.pt',
        config=CONFIG
    )
    train_loss_to_plot = []
    epoch_losses_to_plot = []
    more_epochs_bool = True
    for epoch in range(num_epochs):
        if not more_epochs_bool:
            break
        running_loss = 0
        start = time.time()
        model.train()
        all_sub_batches = _get_all_batches_with_taskid(train_dl_dict)
        time_lengs = []
        for i, (task, (inputs, times)) in enumerate(all_sub_batches):
            # initial housekeeping
            s = time.time()
            torch.cuda.empty_cache()

            # process inputs, send through model
            inputs, times = preprocess_batch(inputs, times, device=device if GPU_intensive else 'cpu', skip_rate=CONFIG.SKIP_RATE)
            output_dict = model(inputs)
            del inputs

            # calculate loss
            loss_dict = loss_fn(output_dict, epoch)

            # check + record loss
            loss = loss_dict['total_loss']
            if contains_non_float_values(loss):
                logger.error(f'Loss was NAN! exiting now')
                more_epochs_bool = False
                break
            del output_dict
            running_loss += loss.item()
            train_loss_to_plot.append(loss.item())

            # step update
            loss.backward()
            del loss
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=.00001, norm_type=2)
            optimizer.step()
            time_lengs.append(time.time() - s)
            print(np.mean(time_lengs) * (len(all_sub_batches) - i), end='\r')

        epoch_losses_to_plot.append(running_loss / (i + 1))
        _simple_loss_plot(
            epoch_losses_to_plot, 
            plot_title='Training Loss over epochs', 
            filename=f'{foldername}/train_loss_epochlevel.png', 
            condition=len(epoch_losses_to_plot) > 0,
            scatter=False
        )
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / (i + 1):.4f} ({time.time() - start:.2f}s)")
        if epoch % 5 == 0 or epoch >= num_epochs-3:
            ckpt_save(
                model_t=model,
                optimizer_t=optimizer,
                epoch_t=epoch,
                loss_t=running_loss / (i + 1),
                filename=ckpt_folder + f'/epoch-{epoch+1}.pt',
                config=CONFIG
            )


def _get_all_batches_with_taskid(dl_dict):
    all_batches = [(task, (inputs, times)) for task, dl in dl_dict.items() for i, (inputs, times) in enumerate(dl)]
    random.shuffle(all_batches)
    return [b for b in all_batches]

def _simple_loss_plot(loss_list, plot_title, filename, condition, scatter=False):
    if condition:
        plt.clf()
        (plt.scatter if scatter else plt.plot)([i for i in range(len(loss_list))], loss_list)
        plt.title(plot_title)
        plt.savefig(filename)