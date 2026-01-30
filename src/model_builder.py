import torch
import torch.nn as nn

class HiggsNet(nn.Module):
    def __init__(self, model_config: dict):
        super(HiggsNet, self).__init__()
        
        input_size = model_config['input_shape_factor']
        hidden_nodes = model_config['hidden_layer_nodes'] 
        
        layers = []
        
        in_features = input_size
        
        for out_features in hidden_nodes:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
            
        layers.append(nn.Linear(in_features, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

