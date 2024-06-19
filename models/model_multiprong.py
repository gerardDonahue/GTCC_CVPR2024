
import torch
import torch.nn as nn

from utils.model_util import get_linear_layers_w_activations
from utils.logging import configure_logging_format

logger = configure_logging_format()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiProngAttDropoutModel(nn.Module):
    def __init__(
        self, 
        base_model_class,
        base_model_params,
        output_dimensionality,
        num_heads,
        dropping=False,
        attn_layers = [512, 256],
        drop_layers = [512, 128, 256],
    ):
        super(MultiProngAttDropoutModel, self).__init__()
        self.num_classes = num_heads
        self.dropping = dropping
        self.output_dimensionality = output_dimensionality
        # shared base model
        self.base_model = base_model_class(**base_model_params)
        # all prongs
        self.head_models = nn.ModuleList({
            HeadModel(output_dimensionality, class_name=head_id) for head_id in range(num_heads)
        })
        # Attention Mechanism
        self.attention_layer = nn.Sequential(
            nn.Linear(output_dimensionality * self.num_classes, attn_layers[0]),  # Adjust the architecture as needed
            # nn.Dropout(p=0.2),
            *get_linear_layers_w_activations(attn_layers, activation_at_end=True, activation=nn.Tanh()),
            nn.Linear(attn_layers[-1], self.num_classes),
            nn.Softmax(dim=1)  # Apply softmax to get attention weights
        ).to(device)
        if self.dropping:
            self.dropout = nn.Sequential(
                nn.Linear(output_dimensionality, drop_layers[0]),  # Adjust the architecture as needed
                *get_linear_layers_w_activations(drop_layers, activation_at_end=True, activation=nn.ReLU()),
                nn.Linear(drop_layers[-1], output_dimensionality + 1)
            ).to(device)

    def forward(self, videos):
        general_features = self.base_model(videos)['outputs']
        prong_outputs = [prong(general_features) for prong in self.head_models]
        prong_outputs = list(map(list, zip(*prong_outputs))) # list transpose to get (batch, video_heads)
        outputs = []
        attentions = []
        dropouts = []
        for prong_output in prong_outputs:
            T = prong_output[0].shape[0]
            prong_output_t = torch.stack(prong_output, dim=0)
            concatenated_prongs = torch.stack(prong_output, dim=0).view(T, -1)
            attention_weights = self.attention_layer(concatenated_prongs)
            weighted_combination = prong_output_t.permute(2,1,0) * attention_weights
            combined_embedding = weighted_combination.sum(dim=2).T
            outputs.append(combined_embedding)
            attentions.append(attention_weights)
            if self.dropping:
                dout = self.dropout(combined_embedding).mean(dim=0)
                dropouts.append(dout)

        if self.dropping:
            return {'outputs': outputs, 'attentions': attentions, 'dropouts': dropouts}
        else:
            return {'outputs': outputs, 'attentions': attentions}

class HeadModel(nn.Module):
    def __init__(self, output_dimensionality, class_name, layers=[512, 128, 256]):
        super(HeadModel, self).__init__()
        self.class_name = class_name
        self.fc_layers = nn.Sequential(
                nn.Linear(output_dimensionality, layers[0]),  # Adjust the architecture as needed
                *get_linear_layers_w_activations(layers, activation_at_end=True, activation=nn.ReLU()),
                nn.Linear(layers[-1], output_dimensionality)
            ).to(device)
        
    def forward(self, x):
        outputs = []
        for i, sequence in enumerate(x):
            this_video = self.fc_layers(sequence)
            # print(this_video.shape, sequence.shape)
            outputs.append(this_video)
        return outputs
