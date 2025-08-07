import torch
import torch.nn as nn

class HiggsNet(nn.Module):
    def __init__(self, model_config: dict):
        super(HiggsNet, self).__init__()
        
        input_size = model_config['input_shape_factor']
        # Ambil daftar jumlah neuron dari config
        hidden_nodes = model_config['hidden_layer_nodes'] 
        
        layers = []
        
        # Tentukan ukuran input untuk lapisan pertama
        in_features = input_size
        
        # Loop melalui daftar hidden_nodes untuk membangun setiap hidden layer
        for out_features in hidden_nodes:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            # Ukuran input untuk lapisan berikutnya adalah ukuran output dari lapisan saat ini
            in_features = out_features
            
        # Layer output terakhir
        # Ukuran inputnya adalah jumlah neuron dari hidden layer terakhir
        layers.append(nn.Linear(in_features, 1))
        
        # Gabungkan semua layer menjadi satu model sekuensial
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

