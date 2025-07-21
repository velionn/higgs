import torch
import torch.nn as nn

class HiggsNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: int, hidden_neurons: int):
        super(HiggsNet, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(nn.ReLU())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_neurons, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

