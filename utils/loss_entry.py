from utils.loss_functions import GTCC_loss, TCC_loss, LAV_loss, VAVA_loss

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loss_function(config_obj):
    loss_booldict = config_obj.LOSS_TYPE
    TCC_ORIGINAL_PARAMS = config_obj.TCC_ORIGINAL_PARAMS
    GTCC_PARAMS = config_obj.GTCC_PARAMS
    LAV_PARAMS = config_obj.LAV_PARAMS
    VAVA_PARAMS = config_obj.VAVA_PARAMS
    GEO_PARAMS = config_obj.GEO_PARAMS
    def _alignment_loss_fn(output_dict_list, epoch):
        if type(output_dict_list) != list:
            output_dict_list = [output_dict_list]
        ################################
        # Dict for returning loss results.
        ################################
        loss_return_dict = {}
        ################################
        # set some starter values
        ################################
        loss_return_dict['total_loss'] = torch.tensor(0).float().to(device)
        for loss_term, verdict in loss_booldict.items():
            if verdict:
                loss_return_dict[loss_term + '_loss'] = torch.tensor(0).float().to(device)

        ################################
        # for each batch output.....
        ################################
        for output_dict in output_dict_list:
            if len(output_dict['outputs']) < 2:
                continue
            # check each loss term, should we add?? verdict will tell
            for loss_term, verdict in loss_booldict.items():
                if verdict:
                    coefficient = 1
                    if loss_term == 'GTCC':
                        specific_loss = GTCC_loss(
                            output_dict['outputs'],
                            dropouts=output_dict['dropouts'],
                            epoch=epoch,
                            **GTCC_PARAMS
                        )
                    elif loss_term == 'tcc':
                        specific_loss = TCC_loss(
                            output_dict['outputs'], **TCC_ORIGINAL_PARAMS
                        )
                    elif loss_term == 'LAV':
                        specific_loss = LAV_loss(
                            output_dict['outputs'], **LAV_PARAMS
                        )
                    elif loss_term == 'VAVA':
                        specific_loss = VAVA_loss(
                            output_dict['outputs'], global_step=epoch, **VAVA_PARAMS
                        )
                    else:
                        print(f"BAD LOSS TERM: {loss_term}, {verdict}")
                        exit(1)
                        
                    loss_return_dict[loss_term + '_loss'] += specific_loss
                    loss_return_dict['total_loss'] += coefficient * specific_loss
        return loss_return_dict
    return _alignment_loss_fn