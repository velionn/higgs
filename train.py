import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAUROC

from src.data_loader import get_dataloaders
from src.model_builder import HiggsNet

def train_model(config_path: Path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    train_loader, valid_loader, _ = get_dataloaders(config)

    model_config = config['model']
    training_config = config['training']
    
    model = HiggsNet(model_config).to(device)
    
    print("Arsitektur Model:")
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    auc_metric = BinaryAUROC().to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = training_config['early_stopping_patience']
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    model_save_path = models_dir / f"{model_config['name']}_best.pt"

    for epoch in range(training_config['epochs']):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        auc_metric.reset()
        with torch.no_grad():
            for features, labels in valid_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                auc_metric.update(outputs, labels.int())

        avg_val_loss = val_loss / len(valid_loader)
        val_auc = auc_metric.compute().item()

        print(f"Epoch {epoch+1}/{training_config['epochs']} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Val loss membaik. Menyimpan model ke {model_save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Tidak ada perbaikan selama {patience} epoch. Menghentikan pelatihan.")
            break
            
    print("Pelatihan selesai.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Latih model PyTorch untuk dataset HIGGS.")
    parser.add_argument('--config', type=str, required=True, help="Path menuju file konfigurasi YAML.")
    args = parser.parse_args()
    train_model(config_path=Path(args.config))

