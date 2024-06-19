import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE
import scipy.stats as st
from scipy.spatial.distance import cdist

from utils.tensorops import preprocess_batch, contains_non_float_values
from utils.math import pca


def plot_gmm_informative(og_distribution, gauss, total_spread, filename):
    p = og_distribution.detach().cpu().numpy()
    gauss = gauss.detach().cpu().numpy()
    total_spread = total_spread.detach().cpu().numpy()
    if contains_non_float_values(total_spread) or contains_non_float_values(og_distribution) or contains_non_float_values(gauss):
        return

    hi = (gauss.T @ total_spread).T

    # downsample
    if p.shape[0] > 100:
        p = p[::p.shape[0] // 100]
        hi = hi[::hi.shape[0] // 100]
        gauss = gauss[:, ::gauss.shape[1] // 100]

    fig, axes = plt.subplots(2 + 1, 1)
    categories = [i for i in range(p.shape[0])]
    axes[0].bar(categories, p)
    axes[0].set_title(f'Outgoing distribution, α', fontsize="20")
    axes[1].bar(categories, hi)
    axes[1].set_title(f'K-modal gaussian, p', fontsize="20")

    plot_spread = total_spread / total_spread.max()
    for i in range(total_spread.shape[0]):
        dist = gauss[i]
        axes[2].bar(categories, dist, alpha=plot_spread[i], color='black')
        axes[2].set_title(f'p mixture components plotted', fontsize="20")

    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.tight_layout()

    fig.savefig(filename)
    print('plotted!')


def hetero_tsne(seq_list, filename, perplexity=30, learning_rate=200, n_iter=1000, random_state=None, drop_list=None):
    """
    Visualize the t-SNE embeddings of a list of numpy sequences.

    Args:
        seq_list (list of numpy arrays): A list of numpy arrays, where each array represents a sequence.
        perplexity (float): The perplexity parameter for t-SNE.
        learning_rate (float): The learning rate for t-SNE.
        n_iter (int): The maximum number of iterations for t-SNE.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        None.
    """
    if drop_list is not None:
        assert len(drop_list) == 2 and len(seq_list) == 2
    # Concatenate the sequences into a single array
    plt.clf()
    seq_concat = np.concatenate(seq_list)

    # Compute the t-SNE embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity)

    # PCA then tSNE
    seq_concat = pca(seq_concat, n_components=8)
    tsne_embed = tsne.fit_transform(seq_concat)

    # Plot the embeddings
    offset = 0
    for i, seq in enumerate(seq_list):
        n_points = seq.shape[0]
        x = tsne_embed[offset:offset+n_points, 0]
        y = tsne_embed[offset:offset+n_points, 1]
        plt.plot(x, y, color=f"C{i}", linewidth=1)   
        plt.scatter(x, y, s=10, color=f"black", marker='.')
        if drop_list is not None:
            assert seq.shape[0] == len(drop_list[i])
            plt.scatter(x[drop_list[i]], y[drop_list[i]], s=20, color=f"purple", marker='s')
            plt.scatter(x[drop_list[i]], y[drop_list[i]], s=20, color=f"brown", marker='s')
        plt.plot([x[0]], [y[0]], color=f"black", marker='^', markersize=10)
        plt.plot([x[-1]], [y[-1]], color=f"black", marker='*', markersize=10)
        offset += n_points
    plt.savefig(filename)

def tSNE(datas, filename="tSNE", plot_tsne=True, neighborhood=20, plotting=True):
    print(f'Data shape: {datas.shape}, filename: {filename}, plot_tsne: {plot_tsne}')
    def plot_sequences(X,  axes: list): 
        '''
            axes: list of axis to display tSNE plots to
            plt_range: (minx, maxx, miny, maxy)
        '''
        counter = 0
        i = 0
        for seq in X:
            x = seq.T[0]
            y = seq.T[1]
            for ax in axes:
                ax.plot(x, y, color=f"C{i}", linewidth=1)
                ax.scatter(x, y, color=f"black", marker='.')
                ax.plot([x[0]], [y[0]], color=f"black", marker='^', markersize=10)
                ax.plot([x[-1]], [y[-1]], color=f"black", marker='*', markersize=10)
                ax.plot([x[0]], [y[0]], color=f"C{i}", marker='^', markersize=5)
                ax.plot([x[-1]], [y[-1]], color=f"C{i}", marker='*', markersize=5)

            i += 1
            counter += 20
    def init_plots():
        plt.clf()
        fig, axis = plt.subplots(2, 2)
        fig.set_figwidth(20)    
        fig.set_figheight(10)   
        return fig, axis
    def plot_states_and_heatmap(states, pltrange, all_axes, scatter_ax, end_states):
        # first, the heatmap
        mode = plot_grid_based_heatmap(states, pltrange, all_axes)
        mark = '*' if end_states else '^'
        # next, scatter
        x = [point[0] for point in states if np.linalg.norm(point - mode) < neighborhood]
        y = [point[1] for point in states if np.linalg.norm(point - mode) < neighborhood]
        scatter_ax.scatter(x, y, color='black', marker=mark, s=40)
        scatter_ax.plot([mode[0]], [mode[1]], color='red', marker=mark, markersize=2)
        return mode
        
    # some preprocessing
    lengths = [len(data) for data in datas]
    OG_X = np.array(datas)

    X = OG_X[0]
    for data in OG_X[1:]:
        X = np.vstack((X, np.array(data)))

    # either plotting tSNE or just 2-dimensional PCA
    X = pca(X, n_components=min(X.shape[1], 8) if plot_tsne else 2)
    if plot_tsne:
        for perp in [30]:
            _, axis = init_plots() 

            # get the tsne data for plotting
            tsne_fnc = TSNE(perplexity=perp)
            tsne = tsne_fnc.fit_transform(X)
            OG_tsne = tsne.reshape(OG_X.shape[0], OG_X.shape[1], 2)
            plot_range = (np.min(tsne, axis=0)[0] - 1, np.max(tsne, axis=0)[0] + 1, np.min(tsne, axis=0)[1] - 1, np.max(tsne, axis=0)[1] + 1)

            # plot the same tSNE on both
            start_mode = plot_states_and_heatmap(get_all_start_states_2d(OG_tsne), plot_range, [axis[0, 0], axis[1, 0]], scatter_ax=axis[1, 0], end_states=False)
            end_mode = plot_states_and_heatmap(get_all_goal_states_2d(OG_tsne), plot_range, [axis[0, 1], axis[1, 1]], scatter_ax=axis[1, 1], end_states=True)
            
            plot_sequences(
                [
                    seq for seq in OG_tsne \
                        if np.linalg.norm(seq[0] - start_mode) < neighborhood and np.linalg.norm(seq[-1] - end_mode) < neighborhood], [axis[0, 0], axis[0, 1]
                ]
            )

            # finally save the plot to a file
            if plotting:
                plt.savefig(f'{filename}_perp={perp}.png')
    else:
        _, axis = init_plots() 
        plot_range = (np.min(X, axis=0)[0] - 1, np.max(X, axis=0)[0] + 1, np.min(X, axis=0)[1] - 1, np.max(X, axis=0)[1] + 1)

        # plot start/end states
        OG_XX = X.reshape(OG_X.shape[0], OG_X.shape[1], 2)
        start_mode = plot_states_and_heatmap(get_all_start_states_2d(OG_XX), plot_range, [axis[0, 0], axis[1, 0]], scatter_ax=axis[1, 0], end_states=False)
        end_mode = plot_states_and_heatmap(get_all_goal_states_2d(OG_XX), plot_range, [axis[0, 1], axis[1, 1]], scatter_ax=axis[1, 1], end_states=True)

        # plot sequences
        plot_sequences(
            [
                seq for seq in OG_XX \
                    if np.linalg.norm(seq[0] - start_mode) < neighborhood and np.linalg.norm(seq[-1] - end_mode) < neighborhood], [axis[0, 0], axis[0, 1]
            ]
        )

        # finally save the plot to a file
        if plotting:
            plt.savefig(f'{filename}.png')

def plot_grid_based_heatmap(data, range_vals, axes1):
    xmin, xmax, ymin, ymax = range_vals
    x = data[:, 0]
    y = data[:, 1]
    
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)

    # getting max probability coordinates
    mode = data[np.argmax([kernel.evaluate(value) for value in data])]


    f = np.reshape(kernel(positions).T, xx.shape)
    # cfset = ax.contourf(xx, yy, f, cmap='Blues')
    for ax in axes1:
        ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    return mode


def get_all_start_states_2d(XX):
    idx = 0
    start_states = []
    for array in XX:
        start_states.append(array[0])
    return np.array(start_states)

def get_all_goal_states_2d(XX):
    idx = 0
    start_states = []
    for array in XX:
        start_states.append(array[-1])
    return np.array(start_states)


def validate_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def plot_model_tsne(model, dataloader, filepath, task, drop=False):
    for i, (inputs, times) in enumerate(dataloader):
        inputs, times = preprocess_batch(inputs, times)
        outputs = [output.clone().detach().cpu().numpy() for output in model(inputs)['outputs']]
        del inputs
        if drop:
            drop12 = [False for i in range(outputs[0].shape[0])]
            drop21 = [False for i in range(outputs[1].shape[0])]
            for drop_vec, t1, t2 in [(drop12, times[0], times[1]), (drop21, times[1], times[0])]:
                for step, start, end in zip(t1['step'], t1['start_frame'], t1['end_frame']):
                    for t in range(start, end):
                        is_SIL = step == 'SIL'
                        is_RED = step not in t2['step']
                        is_bg = is_SIL or is_RED
                        drop_vec[t] = is_bg
            hetero_tsne(outputs[:2], filename=filepath, drop_list=[drop12, drop21])
        else:
            hetero_tsne(outputs, filename=filepath, drop_list=None)
        break

