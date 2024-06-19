import glob
import torch
from utils.train_util import ckpt_restore_mprong, ckpt_restore_sprong


def ckpt_save(
        model_t,
        optimizer_t,
        epoch_t,
        loss_t,
        filename,
        config
    ):
    """
        Creates checkpoint with necessary data.
    """
    # Additional information
    ckpt_dict = {
        'config': config,
        'epoch': epoch_t,
        'loss': loss_t,
        'model_state_dict': model_t.state_dict(),
        'optimizer_state_dict': optimizer_t.state_dict(),
    }
    torch.save(ckpt_dict, filename)


def get_ckpt_MCN(folder, num_heads, device, dropout=False):
    ckpts = glob.glob(folder + f'/ckpt/*')
    try:
        best_ckpt_file = ckpts[-1]
        ckpt_handle = ".".join(best_ckpt_file.split('/')[-1].split('.')[:-1])
        model, _, epoch, _, _ = ckpt_restore_mprong(
            best_ckpt_file,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        return model, epoch, ckpt_handle
    except Exception as e:
        return None, None, None


def get_ckpt_basic(folder, device):
    ckpts = glob.glob(folder + f'/ckpt/*')
    try:
        best_ckpt_file = ckpts[-1]
        ckpt_handle = ".".join(best_ckpt_file.split('/')[-1].split('.'))
        model, _, epoch, _, _ = ckpt_restore_sprong(
            best_ckpt_file,
            device=device
        )
        return model, epoch, ckpt_handle
    except Exception as e:
        return None, None, None


def get_ckpt_for_eval(ckpt_parent_folder, config, device, num_heads=None):
    """
        if task_list is none, we have singleprong architecture, else we have MCN
    """
    if config.ARCHITECTURE['MCN']:
        model, epoch, ckpt_handle = get_ckpt_MCN(
            ckpt_parent_folder,
            num_heads,
            device,
            dropout=config.LOSS_TYPE['GTCC'],
        )
    else:
        model, epoch, ckpt_handle = get_ckpt_basic(
            ckpt_parent_folder,
            device
        )

    if model is None:
        return None, None, None
    else:
        model.eval()
        return model, epoch, ckpt_handle