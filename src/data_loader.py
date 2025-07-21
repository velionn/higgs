import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path

COLUMN_NAMES = [
    'label', 'lepton_pt', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 'missing_energy_phi',
    'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b_tag',
    'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b_tag',
    'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b_tag',
    'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b_tag',
    'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
]
RAW_FEATURES = COLUMN_NAMES[1:22]
HIGH_LEVEL_FEATURES = COLUMN_NAMES[22:]

class HIGGSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].unsqueeze(-1)

def get_dataloaders(config: dict):
    data_config = config['data']
    training_config = config['training']
    
    file_path = Path(data_config['file_path'])
    feature_set = data_config['feature_set']
    batch_size = training_config['batch_size']

    df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES, nrows=data_config.get('max_samples'))
    
    if feature_set == 'raw':
        features = RAW_FEATURES
    elif feature_set == 'high':
        features = HIGH_LEVEL_FEATURES
    else: # 'all'
        features = RAW_FEATURES + HIGH_LEVEL_FEATURES

    X = df[features].values
    y = df['label'].values

    n_train = 10000000
    n_valid = 500000
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_valid, y_valid = X[n_train:n_train+n_valid], y[n_train:n_train+n_valid]
    X_test, y_test = X[n_train+n_valid:], y[n_train+n_valid:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    if 'num_train_samples' in data_config:
        num_samples = data_config['num_train_samples']
        X_train, y_train = X_train[:num_samples], y_train[:num_samples]
    if 'num_valid_samples' in data_config:
        num_samples = data_config['num_valid_samples']
        X_valid, y_valid = X_valid[:num_samples], y_valid[:num_samples]

    train_dataset = HIGGSDataset(X_train, y_train)
    valid_dataset = HIGGSDataset(X_valid, y_valid)
    test_dataset = HIGGSDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, valid_loader, test_loader

