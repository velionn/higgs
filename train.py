# train.py

import yaml
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Impor pemuat data kita dari folder src
from src.data_loader import load_higgs_data

def train_model(config_path: Path):
    """
    Fungsi utama untuk melatih model berdasarkan file konfigurasi.
    """
    # --- 1. Memuat Konfigurasi ---
    print(f"Memuat konfigurasi dari: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ekstrak konfigurasi untuk kemudahan akses
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']

    # --- 2. Memuat Data ---
    # Path ke file data dibangun dari root proyek
    data_file = Path(data_config['file_path'])
    X_train, y_train, X_valid, y_valid, _, _ = load_higgs_data(
        file_path=data_file,
        feature_set=data_config['feature_set']
    )

    if 'num_train_samples' in data_config:
        num_samples = data_config['num_train_samples']
        print(f"Mengurangi data latih menjadi {num_samples} sampel untuk pengujian ringan.")
        X_train = X_train[:num_samples]
        y_train = y_train[:num_samples]

    if 'num_valid_samples' in data_config:
        num_samples = data_config['num_valid_samples']
        print(f"Mengurangi data validasi menjadi {num_samples} sampel untuk pengujian ringan.")
        X_valid = X_valid[:num_samples]
        y_valid = y_valid[:num_samples]

    print("Data latih dan validasi berhasil dimuat dan disesuaikan.")

    # --- 3. Membangun Model ---
    print(f"Membangun model: {model_config['name']}")
    
    model = Sequential()
    # Menambahkan layer input secara eksplisit
    model.add(Input(shape=(model_config['input_shape_factor'],)))

    # Menambahkan hidden layer sesuai jumlah di konfigurasi
    for i in range(model_config['layers']):
        model.add(Dense(
            units=model_config['neurons'],
            activation=model_config['activation'],
            name=f'hidden_layer_{i+1}'
        ))

    # Menambahkan layer output untuk klasifikasi biner
    # Menggunakan aktivasi sigmoid untuk menghasilkan probabilitas (0-1)
    model.add(Dense(units=1, activation='sigmoid', name='output_layer'))

    model.summary() # Mencetak ringkasan arsitektur model

    # --- 4. Kompilasi Model ---
    print("Kompilasi model...")
    model.compile(
        optimizer=training_config['optimizer'],
        loss=training_config['loss'],
        metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy']
    )

    # --- 5. Menyiapkan Callbacks ---
    # Callbacks adalah alat bantu yang dijalankan selama pelatihan
    
    # Path untuk menyimpan model terbaik
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True) # Buat folder jika belum ada
    best_model_path = models_dir / f"{model_config['name']}_best.keras"

    # ModelCheckpoint: Menyimpan hanya model dengan val_loss terbaik
    checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # EarlyStopping: Menghentikan pelatihan jika tidak ada kemajuan
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=training_config['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )

    # --- 6. Melatih Model ---
    print("Memulai pelatihan model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        validation_data=(X_valid, y_valid),
        callbacks=[checkpoint, early_stopping]
    )
    print("Pelatihan selesai.")

if __name__ == '__main__':
    # Menyiapkan parser untuk argumen dari command line
    parser = argparse.ArgumentParser(description="Latih model deep learning untuk dataset HIGGS.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path menuju file konfigurasi YAML."
    )
    args = parser.parse_args()

    # Memulai proses pelatihan dengan path konfigurasi yang diberikan
    train_model(config_path=Path(args.config))

