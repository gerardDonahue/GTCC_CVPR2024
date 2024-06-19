import argparse


def prog_argparser():
    parser = argparse.ArgumentParser(description='Please specify the parameters of the experiment.')
    
    # mandatory arg for version
    parser.add_argument('version')
    # specify sprong or MCN (sprong only for multi-task setting.)
    parser.add_argument('--mcn', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # settings to specify optionally
    parser.add_argument('-lr', '--learning_rate', default='0.0001') 
    parser.add_argument('-bs', '--batch_size', default='2') 
    parser.add_argument('-ep', '--epochs', default='50') 
    parser.add_argument('-od', '--output_dimensions', default='128') 
    parser.add_argument('-m', '--message', default=None) 

    # Create a mutually exclusive group for the METHOD/LOSS
    loss_type = parser.add_mutually_exclusive_group(required=True)
    loss_type.add_argument('--TCC',  '--tcc', nargs='?', action='store', const='tcc', dest='loss_type', help='Loss type is TCC')
    loss_type.add_argument('--LAV',  '--lav', nargs='?', action='store', const='LAV', dest='loss_type', help='Loss type is LAV')
    loss_type.add_argument('--VAVA',  '--vava', nargs='?', action='store', const='VAVA', dest='loss_type', help='Loss type is VAVA')
    loss_type.add_argument('--GTCC',  '--gtcc', nargs='?', action='store', const='GTCC', dest='loss_type', help='Loss type is GTCC')

    # Create a mutually exclusive group for the ARCH
    base_arch = parser.add_mutually_exclusive_group(required=True)
    base_arch.add_argument('--temporal_stacking',  '--tstack', nargs='?', action='store', const='temporal_stacking', dest='base_arch', help='Arch is temporal_stacking')
    base_arch.add_argument('--naive',  '--naive', nargs='?', action='store', const='naive', dest='base_arch', help='Arch is naive')
    base_arch.add_argument('--resnet50',  '--rnet50', nargs='?', action='store', const='resnet50', dest='base_arch', help='Arch is resnet50')

    # Create a mutually exclusive group for the DSET
    dataset_name = parser.add_mutually_exclusive_group(required=True)
    dataset_name.add_argument('--penn_action',  '--penn', nargs='?', action='store', const='penn-action', dest='dataset', help='Dataset is penn-action')
    dataset_name.add_argument('--pouring',  '--pour', nargs='?', action='store', const='pouring', dest='dataset', help='Dataset is pouring')
    dataset_name.add_argument('--egoprocel',  '--ego', nargs='?', action='store', const='egoprocel', dest='dataset', help='Dataset is egoprocel-skip15')
    dataset_name.add_argument('--coin',  '--coin', nargs='?', action='store', const='coin', dest='dataset', help='Dataset is coin')
    dataset_name.add_argument('--cmu',  '--cmu', nargs='?', action='store', const='cmu', dest='dataset', help='Dataset is cmu')
    dataset_name.add_argument('--egtea',  '--egtea', nargs='?', action='store', const='egtea', dest='dataset', help='Dataset is egtea')

    args = parser.parse_args()
    return {
        'version': args.version,
        'dataset': args.dataset,
        'loss_type': args.loss_type,
        'base_arch': args.base_arch,
        'lr': args.learning_rate,
        'message': args.message,
        'MCN': args.mcn,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'output_dimensions': args.output_dimensions,
        'debug': args.debug,
    }