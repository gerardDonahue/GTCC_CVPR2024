import torch
import torch.nn as nn
import numpy as np

from utils.logging import configure_logging_format
from utils.model_util import get_linear_layers_w_activations

logger = configure_logging_format()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Resnet50Encoder(nn.Module):
    def __init__(
            self,
            temporal_depth,
            dropping=False,
            in_size=1024,
        ):
        super(Resnet50Encoder, self).__init__()
        self.device = device
        self.dropping = dropping
        self.k = temporal_depth
        self.input_dimension = in_size
        self.conv_embedding, self.linear_embedding = self.get_embeddermodel(in_size=in_size)
        if dropping:
            self.dropout = nn.Sequential(
                nn.Linear(128, 512),  # Adjust the architecture as needed
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, 128)
            ).to(device)

    def get_embeddermodel(self, in_size):
        layers1 = nn.Sequential(
            nn.Conv3d(in_channels=in_size, out_channels=in_size, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_size, out_channels=512, kernel_size=3, padding='same'),
            nn.ReLU()
        ).to(device)
        layers2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        ).to(device)
        return layers1, layers2



    def forward(self, videos):
        """
            This forward function takes in inputs and returns those inputs as embedded outputs
                this function handles different length inputs
            - takes in list of tensors [(vid(i)_length, input_dimensions) for i in batch_size]
            - outputs list of tensors [(embedded_vid(i)_length, output_dimensions) for i in batch_size]
        """
        outputs = []
        dropouts = []
        for i, sequence in enumerate(videos):
            # get the base model output for each frame
            base_output = sequence.to(device)
            zero_arrays = torch.zeros((self.k,) + (self.input_dimension, 14, 14), dtype=base_output.dtype, device=base_output.device)
            base_output = torch.cat((zero_arrays, base_output), dim=0)
            # temporal stacking
            this_video = torch.Tensor().long().to(device)
            for t in range(self.k, base_output.shape[0]):
                stack = base_output[t - self.k:t, :].permute(1, 0, 2, 3)
                conv5 = self.conv_embedding(stack)
                spatio = nn.MaxPool3d(kernel_size=conv5.shape[1:])(conv5)
                embedding = self.linear_embedding(spatio.squeeze())
                this_video = torch.cat((this_video, embedding[None, :]), 0)
            outputs.append(this_video)
            if self.dropping:
                dout = self.dropout(this_video).mean(dim=0)
                dropouts.append(dout)
        
        if self.dropping:
            return {'outputs': outputs, 'dropouts': dropouts}
        else:
            return {'outputs': outputs}

class StackingEncoder(nn.Module):
    """
        Quicker model which works well. 
    """
    def __init__(
            self,
            temporal_depth,
            conv_num_channels,
            output_dimensions=128,
            input_dimensions=2048,
            dropping=False,
            drop_layers = [512, 128, 256],
        ):
        super(StackingEncoder, self).__init__()
        assert output_dimensions < conv_num_channels and output_dimensions in [512, 256, 128, 64, 32, 16, 8, 4, 2]
        self.device = device
        self.dropping = dropping

        self.get_conv = lambda in_size, out_size, kernel_size : nn.Conv1d(
            in_channels=in_size,
            out_channels=out_size,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            bias=False
        )
        self.input_dimensions = input_dimensions
        self.conv_num_channels = conv_num_channels
        self.temporal_depth = temporal_depth
        self.output_dimensions = output_dimensions

        self.convolutional_layers = nn.Sequential(
            *self.get_convolutional_layers()
        ).to(device)
        if dropping:
            self.dropout = nn.Sequential(
                nn.Linear(self.output_dimensions, drop_layers[0]),  # Adjust the architecture as needed
                *get_linear_layers_w_activations(drop_layers, activation_at_end=True),
                nn.Linear(drop_layers[-1], self.output_dimensions + 1)
            ).to(device)


    def get_descending_layers(self):
        end = self.output_dimensions
        cur = self.conv_num_channels
        output = []
        while cur >= end:
            output.append(round(cur))
            cur = cur / 2
        return output

    def get_convolutional_layers(self):
        layers = self.get_descending_layers()
        min_numlayers = 7
        cur_channels = self.input_dimensions
        next_channels = layers[0]
        layer_list = []
        layer_num = 0
        while True:
            layer = self.get_conv(cur_channels, next_channels, self.temporal_depth if layer_num == 0 else 3)
            layer_list.append(layer)
            layer_num += 1
            cur_channels = next_channels
            if cur_channels <= self.output_dimensions and layer_num >= min_numlayers:
                break
            elif cur_channels == self.output_dimensions and layer_num < min_numlayers:
                next_channels = cur_channels
            else:
                next_channels = layers[layer_num]
            layer_list.append(nn.ReLU())
        return layer_list

    def forward(self, videos):
        """
            This forward function takes in inputs and returns those inputs as embedded outputs
                this function handles different length inputs
            - takes in list of tensors [(vid(i)_length, input_dimensions) for i in batch_size]
            - outputs list of tensors [(embedded_vid(i)_length, output_dimensions) for i in batch_size]
        """
        outputs = []
        dropouts = []
        for i, sequence in enumerate(videos):
            sequence = sequence.to(device)
            sequence = sequence.permute(1,0)
            this_video = self.convolutional_layers(sequence.to(device)).permute(1,0)
            outputs.append(this_video)
            if self.dropping:
                dout = self.dropout(this_video).mean(dim=0)
                dropouts.append(dout)
        
        if self.dropping:
            return {'outputs': outputs, 'dropouts': dropouts}
        else:
            return {'outputs': outputs}


class NaiveEncoder(nn.Module):
    def __init__(
            self,
            layers,
            output_dimensions=128,
            drop_layers=None,
        ):
        super(NaiveEncoder, self).__init__()
        self.device = device
        self.dropping = drop_layers is not None
        self.output_dimensions = output_dimensions
        if drop_layers is not None:
            assert type(drop_layers) == list and all(type(x) == int for x in drop_layers)

        self.layers =nn.Sequential(
                nn.Linear(2048, layers[0]),  # Adjust the architecture as needed
                *get_linear_layers_w_activations(drop_layers, activation_at_end=True),
                nn.Linear(layers[-1], self.output_dimensions)
            ).to(device)

        if self.dropping:
            self.dropout = nn.Sequential(
                nn.Linear(self.output_dimensions, drop_layers[0]),  # Adjust the architecture as needed
                *get_linear_layers_w_activations(drop_layers, activation_at_end=True),
                nn.Linear(drop_layers[-1], self.output_dimensions + 1)
            ).to(device)


    def forward(self, videos):
        """
            This forward function takes in inputs and returns those inputs as embedded outputs
                this function handles different length inputs
            - takes in list of tensors [(vid(i)_length, input_dimensions) for i in batch_size]
            - outputs list of tensors [(embedded_vid(i)_length, output_dimensions) for i in batch_size]
        """
        outputs = []
        dropouts = []
        for i, sequence in enumerate(videos):
            this_video = self.layers(sequence.to(device))
            outputs.append(this_video)
            if self.dropping:
                dout = self.dropout(this_video).mean(dim=0)
                dropouts.append(dout)
        
        if self.dropping:
            return {'outputs': outputs, 'dropouts': dropouts}
        else:
            return {'outputs': outputs}
