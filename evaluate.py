import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import json
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from src.data_loader import get_dataloaders
from src.model_builder import HiggsNet

def evaluate_model(config_path: Path):
    print(f"Memuat konfigurasi dari: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    _, _, test_loader = get_dataloaders(config)

    model_config = config['model']
    
    model = HiggsNet(model_config).to(device)

    model_save_path = Path('models') / f"{model_config['name']}_best.pt"
    if not model_save_path.exists():
        print(f"Error: File model tidak ditemukan di {model_save_path}")
        return

    print(f"Memuat bobot model dari: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    criterion = nn.BCEWithLogitsLoss()
    auc_metric = BinaryAUROC().to(device)
    acc_metric = BinaryAccuracy().to(device)
    
    model.eval() 
    test_loss = 0.0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            auc_metric.update(outputs, labels.int())
            acc_metric.update(outputs, labels.int())

    avg_test_loss = test_loss / len(test_loader)
    test_auc = auc_metric.compute().item()
    test_acc = acc_metric.compute().item()

    metrics = {
        'loss': avg_test_loss,
        'auc': test_auc,
        'accuracy': test_acc
    }

    print("\n--- Hasil Evaluasi Akhir pada Test Set ---")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC (Area Under Curve): {metrics['auc']:.4f}")
    print("-----------------------------------------")

    results_dir = Path('results/metrics')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file_path = results_dir / f"{model_config['name']}.json"
    
    with open(results_file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Hasil evaluasi telah disimpan di: {results_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluasi model PyTorch pada dataset HIGGS.")
    parser.add_argument('--config', type=str, required=True, help="Path menuju file konfigurasi YAML dari model yang akan dievaluasi.")
    args = parser.parse_args()
    evaluate_model(config_path=Path(args.config))

