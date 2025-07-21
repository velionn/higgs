# evaluate.py

import argparse
import json
from pathlib import Path
import tensorflow as tf

# Impor pemuat data kita dari folder src
from src.data_loader import load_higgs_data

def evaluate_model(model_path: Path, feature_set: str):
    """
    Fungsi untuk mengevaluasi model yang sudah dilatih pada test set.

    Args:
        model_path (Path): Path menuju file model .keras yang sudah disimpan.
        feature_set (str): Set fitur yang digunakan saat melatih model
                           ('all', 'raw', atau 'only').
    """
    if not model_path.exists():
        print(f"Error: File model tidak ditemukan di {model_path}")
        return

    print(f"Mengevaluasi model: {model_path.name}")
    print(f"Menggunakan set fitur: '{feature_set}'")

    # --- 1. Memuat Data Uji ---
    # Kita hanya memerlukan X_test dan y_test, jadi kita abaikan sisanya
    # dengan menggunakan underscore (_)
    data_file = Path('data/raw/HIGGS.csv')
    _, _, _, _, X_test, y_test = load_higgs_data(
        file_path=data_file,
        feature_set=feature_set
    )
    print("Data uji berhasil dimuat.")

    # --- 2. Memuat Model yang Sudah Dilatih ---
    print("Memuat model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return
    
    print("Model berhasil dimuat.")

    # --- 3. Mengevaluasi Model pada Data Uji ---
    print("Memulai evaluasi pada test set...")
    # model.evaluate() akan mengembalikan nilai loss dan metrik
    # yang kita definisikan saat kompilasi (loss, auc, accuracy)
    results = model.evaluate(X_test, y_test, verbose=1)
    
    # Membuat dictionary yang rapi untuk hasil
    metrics = {
        'loss': results[0],
        'auc': results[1],
        'accuracy': results[2]
    }
    
    print("\n--- Hasil Evaluasi Akhir ---")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC (Area Under Curve): {metrics['auc']:.4f}")
    print("----------------------------")

    # --- 4. Menyimpan Hasil ---
    # Membuat folder results/metrics jika belum ada
    results_dir = Path('results/metrics')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Nama file hasil akan sama dengan nama model, tetapi dengan ekstensi .json
    results_file_path = results_dir / f"{model_path.stem}.json"
    
    with open(results_file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Hasil evaluasi telah disimpan di: {results_file_path}")


if __name__ == '__main__':
    # Menyiapkan parser untuk argumen dari command line
    parser = argparse.ArgumentParser(description="Evaluasi model yang sudah dilatih pada test set HIGGS.")
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help="Path menuju file model .keras yang akan dievaluasi (contoh: models/higgs_1_layer_raw_full_best.keras)."
    )
    parser.add_argument(
        '--feature-set',
        type=str,
        required=True,
        choices=['all', 'raw', 'only'],
        help="Set fitur yang digunakan untuk melatih model ini."
    )
    args = parser.parse_args()

    # Memulai proses evaluasi
    evaluate_model(model_path=Path(args.model_path), feature_set=args.feature_set)

