from pprint import pprint as pp

from configs.generic_config import CONFIG as BASE_CONFIG
from utils.os_util import get_env_variable
from utils.parser_util import prog_argparser

data_path = get_env_variable('DATASET_PATH')
output_path = get_env_variable('OUTPUT_PATH')


def get_generic_config(multi_task_setting=False, delta=None, n_components=None):
    # parse command line args
    args_given = prog_argparser()

    # save args to local vars
    version = args_given['version']
    dataset = args_given['dataset']
    loss_type = args_given['loss_type']
    base_arch = args_given['base_arch']
    lr = args_given['lr']
    batch_size = args_given['batch_size']
    epochs = args_given['epochs']
    output_dimensions = args_given['output_dimensions']
    message = args_given['message']
    MCN = args_given['MCN']
    DEBUG = args_given['debug']
    output_foldername = 'multi-task-setting' if multi_task_setting else 'single-task-setting'

    # validate the parsed values
    validate_parsed_values(
        version,
        dataset,
        loss_type,
        base_arch,
        lr,
        batch_size,
        epochs,
        output_dimensions,
        message,
        MCN
    )

    # set the base config object, where None values must be addressed
    CONFIG = BASE_CONFIG
    CONFIG.MULTITASK = multi_task_setting
    
    # dataset related items
    CONFIG.DATASET_NAME = dataset
    if dataset in ['cmu', 'egtea']:
        CONFIG.DATAFOLDER = f'{data_path}/egoprocel'
    else:
        CONFIG.DATAFOLDER = f'{data_path}/{dataset}'

    # set the debug flag
    CONFIG.DEBUG = DEBUG

    # set the architectural information
    CONFIG.ARCHITECTURE['MCN'] = MCN
    CONFIG.BASEARCH.ARCHITECTURE = base_arch

    # set the loss function type
    CONFIG.LOSS_TYPE[loss_type] = True

    # if using GTCC, set the delta value
    if loss_type == 'GTCC':
        CONFIG.GTCC_PARAMS['delta'] = default_delta_for_dset(dataset) if delta is None else delta
        CONFIG.GTCC_PARAMS['n_components'] = default_K_for_dset(dataset) if n_components is None else n_components
        CONFIG.GTCC_PARAMS['gamma'] = default_gamma_for_dset(dataset) if n_components is None else n_components

    # if not using MCN, sprong network holds the drop network
    if 'GTCC' in loss_type and not MCN:
        CONFIG.BASEARCH.TEMPORAL_STACKING_ARCH['dropping'] = True
        CONFIG.BASEARCH.TEMPORAL_STACKING_ARCH['output_dimensions'] = int(output_dimensions)

    # set split based on dataset
    if dataset in ['coin', 'egoprocel', 'cmu', 'egtea']:
        CONFIG.TRAIN_SPLIT = [.65, .5]
    else:
        CONFIG.TRAIN_SPLIT = [.75, .5]

    # int values for training
    CONFIG.LEARNING_RATE = float(lr)
    CONFIG.BATCH_SIZE = int(batch_size)
    CONFIG.NUM_EPOCHS = int(epochs)
    CONFIG.OUTPUT_DIMENSIONALITY = int(output_dimensions)

    # output folders related items
    CONFIG.EVAL_PLOTFOLDER = f'{output_path}/{output_foldername}'
    CONFIG.VERSION = version
    CONFIG.EXPERIMENTNAME = f'V{CONFIG.VERSION}___{loss_type}_{dataset}{f".{message}" if message is not None else ""}'

    return CONFIG

def default_delta_for_dset(dataset_name):
    if dataset_name in ['egoprocel', 'cmu', 'egtea']: # here list in-the-wild datasets.
        return 0.2
    else:
        return 0.5

def default_K_for_dset(dataset_name):
    if dataset_name in ['egoprocel', 'cmu', 'egtea']: # here list in-the-wild datasets.
        return 15
    else:
        return 5

def default_gamma_for_dset(dataset_name):
    if dataset_name in ['egoprocel', 'cmu', 'egtea']: # here list in-the-wild datasets.
        return 0.95
    else:
        return 1

def validate_parsed_values(version, dataset, loss_type, base_arch, lr, batch_size, epochs, output_dimensions, message, MCN):
    assert loss_type in ['tcc', 'GTCC', 'LAV', 'VAVA'], 'parser_check: Wrong loss_type'
    assert base_arch in ['temporal_stacking', 'resnet50', 'naive'], 'parser_check: Wrong base_arch'
    assert dataset in ['egoprocel', 'cmu', 'egtea'], 'parser_check: Wrong dataset'
    assert type(version) == str, 'parser_check: Wrong version'
    assert type(float(lr)) == float, 'parser_check: Wrong lr'
    assert type(int(batch_size)) == int, 'parser_check: Wrong batch_size'
    assert type(int(epochs)) == int, 'parser_check: Wrong epochs'
    assert type(int(output_dimensions)) == int, 'parser_check: Wrong output_dimensions'
    assert type(MCN) == bool, 'parser_check: Wrong MCN'
    assert type(message) == str or message is None, 'parser_check: Wrong message'
