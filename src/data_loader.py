import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# --- Konstanta untuk keterbacaan kode ---
# Nama kolom sesuai dengan dokumentasi dataset HIGGS
# Kolom pertama adalah label, diikuti oleh 28 fitur
COLUMN_NAMES = [
    'label',
    # 21 Fitur Tingkat Rendah (raw features)
    'lepton_pt', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 'missing_energy_phi',
    'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b_tag',
    'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b_tag',
    'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b_tag',
    'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b_tag',
    # 7 Fitur Tingkat Tinggi (high-level features)
    'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
]

# Definisikan kelompok fitur untuk memudahkan pemilihan
RAW_FEATURES = COLUMN_NAMES[1:22]
HIGH_LEVEL_FEATURES = COLUMN_NAMES[22:]

def load_higgs_data(file_path: Path, feature_set: str = 'all'):
    """
    Memuat, membagi, dan memproses dataset HIGGS dari file CSV.

    Fungsi ini mereplikasi logika dari paper asli untuk membagi data
    dan memilih set fitur yang berbeda untuk eksperimen.

    Args:
        file_path (Path): Path object menuju file HIGGS.csv.
        feature_set (str): Set fitur yang akan digunakan.
                           Pilihan: 'all', 'raw', 'only'.
                           - 'all': Menggunakan semua 28 fitur.
                           - 'raw': Hanya menggunakan 21 fitur tingkat rendah.
                           - 'only': Hanya menggunakan 7 fitur tingkat tinggi.

    Returns:
        tuple: Tuple berisi 6 array NumPy:
               (X_train, y_train, X_valid, y_valid, X_test, y_test)
               yang sudah distandardisasi.
    """
    print(f"Memuat data dari: {file_path}")
    # Membaca file CSV menggunakan pandas. Karena tidak ada header, kita berikan nama kolom.
    df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES)
    print("Data berhasil dimuat.")

    # --- 1. Pemilihan Fitur ---
    print(f"Memilih set fitur: '{feature_set}'")
    if feature_set == 'raw':
        features = RAW_FEATURES
    elif feature_set == 'only':
        features = HIGH_LEVEL_FEATURES
    elif feature_set == 'all':
        features = RAW_FEATURES + HIGH_LEVEL_FEATURES
    else:
        raise ValueError("feature_set harus salah satu dari: 'all', 'raw', atau 'only'")

    # Pisahkan fitur (X) dan label (y)
    X = df[features]
    y = df['label']

    # --- 2. Pembagian Data (Sesuai Paper) ---
    # Paper secara spesifik menggunakan jumlah baris tetap untuk setiap set.
    # 10,000,000 untuk training
    # 500,000 untuk validation
    # 500,000 untuk testing
    n_train = 10000000
    n_valid = 500000
    
    print("Membagi data menjadi set train, validasi, dan tes...")
    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]

    X_valid = X.iloc[n_train : n_train + n_valid]
    y_valid = y.iloc[n_train : n_train + n_valid]

    X_test = X.iloc[n_train + n_valid : n_train + n_valid + 500000] # sisanya
    y_test = y.iloc[n_train + n_valid : n_train + n_valid + 500000]

    # --- 3. Standardisasi Fitur ---
    # Ini adalah langkah pra-pemrosesan yang sangat penting untuk model deep learning.
    # Kita membuat scaler (objek untuk standardisasi).
    print("Melakukan standardisasi fitur...")
    scaler = StandardScaler()

    # PENTING: Fit scaler HANYA pada data latih (train set) untuk mencegah kebocoran data (data leakage).
    scaler.fit(X_train)

    # Terapkan transformasi yang sama ke semua set data.
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    print("Standardisasi selesai.")

    # Mengembalikan data sebagai NumPy array, format yang umum digunakan oleh TensorFlow/PyTorch.
    return (
        X_train_scaled, y_train.to_numpy(),
        X_valid_scaled, y_valid.to_numpy(),
        X_test_scaled, y_test.to_numpy()
    )


# --- Contoh Penggunaan ---
# Blok ini hanya akan berjalan jika Anda menjalankan file ini secara langsung (python src/data_loader.py)
# Ini sangat berguna untuk menguji apakah fungsi kita bekerja dengan benar.
if __name__ == '__main__':
    # Pastikan Anda sudah menempatkan HIGGS.csv di dalam folder 'data/raw/'
    # Path relatif dari root proyek
    data_file_path = Path(__file__).parent.parent / 'data' / 'raw' / 'HIGGS.csv'

    if not data_file_path.exists():
        print("-" * 50)
        print(f"ERROR: File data tidak ditemukan di {data_file_path}")
        print("Silakan unduh dataset dari UCI dan letakkan di sana.")
        print("Link: https://archive.ics.uci.edu/dataset/280/higgs")
        print("-" * 50)
    else:
        print("\n--- Menguji pemuat data dengan set fitur 'all' ---")
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_higgs_data(
            file_path=data_file_path,
            feature_set='all'
        )
        print(f"Bentuk X_train: {X_train.shape}") # Harusnya (10000000, 28)
        print(f"Bentuk y_train: {y_train.shape}")
        print(f"Bentuk X_valid: {X_valid.shape}") # Harusnya (500000, 28)
        print(f"Bentuk X_test: {X_test.shape}")   # Harusnya (500000, 28)

        print("\n--- Menguji pemuat data dengan set fitur 'raw' ---")
        X_train_raw, _, _, _, _, _ = load_higgs_data(
            file_path=data_file_path,
            feature_set='raw'
        )
        print(f"Bentuk X_train_raw: {X_train_raw.shape}") # Harusnya (10000000, 21)

        print("\n--- Menguji pemuat data dengan set fitur 'only' ---")
        X_train_only, _, _, _, _, _ = load_higgs_data(
            file_path=data_file_path,
            feature_set='only'
        )
        print(f"Bentuk X_train_only: {X_train_only.shape}") # Harusnya (10000000, 7)
