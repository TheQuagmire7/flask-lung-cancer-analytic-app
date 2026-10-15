import sqlite3
import os
import glob
from datetime import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "lung_cancer.db")
CSV_FILE = os.path.join(BASE_DIR, "lung_cancer.csv")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")

os.makedirs(BACKUP_DIR, exist_ok=True)


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_dataframe(df):
    """
    Menyamakan nama kolom dataset dengan format yang digunakan aplikasi.
    """

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.upper()
    )

    return df


def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()

    # =========================================================
    # PATIENTS
    # =========================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT UNIQUE NOT NULL,
            patient_name TEXT NOT NULL,
            gender INTEGER,
            age INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =========================================================
    # TRAINING DATA
    # =========================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender INTEGER,
            age INTEGER,
            smoking INTEGER,
            yellow_fingers INTEGER,
            anxiety INTEGER,
            peer_pressure INTEGER,
            chronic_disease INTEGER,
            fatigue INTEGER,
            allergy INTEGER,
            wheezing INTEGER,
            alcohol_consuming INTEGER,
            coughing INTEGER,
            shortness_of_breath INTEGER,
            swallowing_difficulty INTEGER,
            chest_pain INTEGER,
            lung_cancer INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =========================================================
    # TESTING DATA
    # =========================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS testing_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender INTEGER,
            age INTEGER,
            smoking INTEGER,
            yellow_fingers INTEGER,
            anxiety INTEGER,
            peer_pressure INTEGER,
            chronic_disease INTEGER,
            fatigue INTEGER,
            allergy INTEGER,
            wheezing INTEGER,
            alcohol_consuming INTEGER,
            coughing INTEGER,
            shortness_of_breath INTEGER,
            swallowing_difficulty INTEGER,
            chest_pain INTEGER,
            lung_cancer INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =========================================================
    # PREDICTIONS
    # =========================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            patient_name TEXT,
            logistic_prediction INTEGER,
            logistic_probability REAL,
            random_forest_prediction INTEGER,
            random_forest_probability REAL,
            overall_result TEXT,
            average_probability REAL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)

    # =========================================================
    # MODEL EVALUATION
    # =========================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_evaluation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # =========================================================
    # APPLICATION SETTINGS
    # =========================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL
        )
    """)

    default_settings = {
        "risk_threshold": "50",
        "rows_per_page": "10",
        "admin_name": "Administrator"
    }

    for key, value in default_settings.items():
        cursor.execute(
            "INSERT OR IGNORE INTO app_settings (setting_key, setting_value) VALUES (?, ?)",
            (key, value)
        )

    conn.commit()
    conn.close()


def convert_value(value):
    """
    Mengubah nilai YES/NO menjadi 1/0.
    Nilai numerik 1/2 dari dataset tetap dipertahankan.
    """

    if pd.isna(value):
        return 0

    if isinstance(value, str):
        value = value.strip().upper()

        if value == "YES":
            return 1

        if value == "NO":
            return 0

        if value == "M":
            return 1

        if value == "F":
            return 0

    try:
        return int(float(value))
    except:
        return 0


REQUIRED_COLUMNS = [
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
    "LUNG_CANCER"
]

FEATURE_COLUMNS = [col for col in REQUIRED_COLUMNS if col != "LUNG_CANCER"]

FEATURE_COLUMNS_DB = [
    "gender",
    "age",
    "smoking",
    "yellow_fingers",
    "anxiety",
    "peer_pressure",
    "chronic_disease",
    "fatigue",
    "allergy",
    "wheezing",
    "alcohol_consuming",
    "coughing",
    "shortness_of_breath",
    "swallowing_difficulty",
    "chest_pain"
]

DATASET_INSERT_QUERY = """
    INSERT INTO training_data (
        gender,
        age,
        smoking,
        yellow_fingers,
        anxiety,
        peer_pressure,
        chronic_disease,
        fatigue,
        allergy,
        wheezing,
        alcohol_consuming,
        coughing,
        shortness_of_breath,
        swallowing_difficulty,
        chest_pain,
        lung_cancer
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class DatasetImportError(Exception):
    """Dilempar saat validasi/format dataset CSV gagal."""
    pass


def _row_to_values(row):
    return (
        int(row["GENDER"]),
        int(row["AGE"]),
        int(row["SMOKING"]),
        int(row["YELLOW_FINGERS"]),
        int(row["ANXIETY"]),
        int(row["PEER_PRESSURE"]),
        int(row["CHRONIC_DISEASE"]),
        int(row["FATIGUE"]),
        int(row["ALLERGY"]),
        int(row["WHEEZING"]),
        int(row["ALCOHOL_CONSUMING"]),
        int(row["COUGHING"]),
        int(row["SHORTNESS_OF_BREATH"]),
        int(row["SWALLOWING_DIFFICULTY"]),
        int(row["CHEST_PAIN"]),
        int(row["LUNG_CANCER"])
    )


def _prepare_ml_dataframe(df):
    """
    Validasi kolom dan encoding dataframe mentah menjadi dataframe
    siap pakai untuk training/testing (nilai numerik saja).
    Melempar DatasetImportError jika kolom tidak lengkap.
    """

    df = normalize_dataframe(df)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise DatasetImportError(
            "Kolom dataset tidak lengkap. Kolom yang hilang: "
            + ", ".join(missing_columns)
        )

    ml_df = df[REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        ml_df[column] = ml_df[column].apply(convert_value)

    if ml_df["LUNG_CANCER"].nunique() < 2:
        raise DatasetImportError(
            "Dataset harus memiliki minimal 2 kelas LUNG_CANCER "
            "(YES dan NO) agar bisa dibagi menjadi data training "
            "dan testing."
        )

    return ml_df


def _split_dataset(ml_df):
    from sklearn.model_selection import train_test_split

    X = ml_df[FEATURE_COLUMNS]
    y = ml_df["LUNG_CANCER"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    training_df = X_train.copy()
    training_df["LUNG_CANCER"] = y_train

    testing_df = X_test.copy()
    testing_df["LUNG_CANCER"] = y_test

    return training_df, testing_df


def inspect_dataset(df):
    """
    Memeriksa dataframe CSV yang baru diupload TANPA mengubah database.
    Dipakai untuk menampilkan preview sebelum user konfirmasi upload.
    """

    normalized = normalize_dataframe(df.copy())

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in normalized.columns
    ]

    result = {
        "valid": not missing_columns,
        "missing_columns": missing_columns,
        "total_rows": len(normalized),
        "class_counts": {},
        "preview_rows": [],
        "can_train": False
    }

    if missing_columns:
        return result

    encoded_target = normalized["LUNG_CANCER"].apply(convert_value)
    counts = encoded_target.value_counts().to_dict()

    result["class_counts"] = {
        "Positive (LUNG_CANCER = 1)": int(counts.get(1, 0)),
        "Negative (LUNG_CANCER = 0)": int(counts.get(0, 0))
    }
    result["can_train"] = len(counts) >= 2

    preview_df = normalized[REQUIRED_COLUMNS].head(10)
    result["preview_rows"] = preview_df.to_dict(orient="records")

    return result


def backup_current_dataset():
    """
    Menyimpan salinan (snapshot) dataset training + testing yang SEDANG
    ada di database ke folder backups/ sebelum diganti/dihapus.
    Mengembalikan None jika tidak ada data untuk dibackup.
    """

    conn = get_db_connection()

    training_rows = conn.execute(
        f"SELECT {', '.join(FEATURE_COLUMNS_DB)}, lung_cancer "
        "FROM training_data"
    ).fetchall()

    testing_rows = conn.execute(
        f"SELECT {', '.join(FEATURE_COLUMNS_DB)}, lung_cancer "
        "FROM testing_data"
    ).fetchall()

    conn.close()

    if not training_rows and not testing_rows:
        return None

    records = []

    for row in training_rows:
        record = dict(row)
        record["dataset_split"] = "training"
        records.append(record)

    for row in testing_rows:
        record = dict(row)
        record["dataset_split"] = "testing"
        records.append(record)

    backup_df = pd.DataFrame(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset_backup_{timestamp}.csv"
    path = os.path.join(BACKUP_DIR, filename)

    backup_df.to_csv(path, index=False)

    return {
        "filename": filename,
        "path": path,
        "rows": len(backup_df)
    }


def list_backups(limit=10):
    """
    Mengembalikan daftar file backup dataset terbaru.
    """

    pattern = os.path.join(BACKUP_DIR, "dataset_backup_*.csv")
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime, reverse=True)

    backups = []

    for path in files[:limit]:
        filename = os.path.basename(path)
        stat = os.stat(path)

        try:
            row_count = sum(1 for _ in open(path, "r")) - 1
        except Exception:
            row_count = None

        backups.append({
            "filename": filename,
            "rows": row_count,
            "size_kb": round(stat.st_size / 1024, 1),
            "created_at": datetime.fromtimestamp(
                stat.st_mtime
            ).strftime("%d %b %Y, %H:%M")
        })

    return backups


def clear_dataset():
    """
    Menghapus seluruh data pada tabel training_data dan testing_data
    tanpa memasukkan data baru.
    """

    conn = get_db_connection()

    try:
        conn.execute("DELETE FROM training_data")
        conn.execute("DELETE FROM testing_data")
        conn.execute(
            "DELETE FROM sqlite_sequence WHERE name IN "
            "('training_data', 'testing_data')"
        )
        conn.commit()
    finally:
        conn.close()


def replace_dataset_from_dataframe(df, save_as_default_csv=False):
    """
    Mengganti seluruh isi tabel training_data dan testing_data
    dengan dataset baru dari dataframe CSV yang diberikan.

    Pembagian data dibuat sama dengan proses model:
    test_size=0.2, random_state=42, stratify=y

    Mengembalikan dict berisi statistik hasil import.
    Melempar DatasetImportError jika dataset tidak valid.
    """

    ml_df = _prepare_ml_dataframe(df)
    training_df, testing_df = _split_dataset(ml_df)

    conn = get_db_connection()

    try:
        # Hapus dataset lama
        conn.execute("DELETE FROM training_data")
        conn.execute("DELETE FROM testing_data")
        conn.execute(
            "DELETE FROM sqlite_sequence WHERE name IN "
            "('training_data', 'testing_data')"
        )

        # Masukkan dataset baru
        for _, row in training_df.iterrows():
            conn.execute(DATASET_INSERT_QUERY, _row_to_values(row))

        testing_insert_query = DATASET_INSERT_QUERY.replace(
            "training_data",
            "testing_data"
        )

        for _, row in testing_df.iterrows():
            conn.execute(testing_insert_query, _row_to_values(row))

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    if save_as_default_csv:
        try:
            df.to_csv(CSV_FILE, index=False)
        except Exception as e:
            print("Gagal menyimpan CSV sebagai default dataset:", e)

    return {
        "total": len(ml_df),
        "training": len(training_df),
        "testing": len(testing_df)
    }


def import_dataset():
    """
    Memasukkan dataset CSV default ke tabel training_data dan
    testing_data. Hanya berjalan sekali saat database masih kosong.
    """

    if not os.path.exists(CSV_FILE):
        print("CSV tidak ditemukan:", CSV_FILE)
        return

    conn = get_db_connection()

    training_count = conn.execute(
        "SELECT COUNT(*) FROM training_data"
    ).fetchone()[0]

    testing_count = conn.execute(
        "SELECT COUNT(*) FROM testing_data"
    ).fetchone()[0]

    conn.close()

    # Jangan import ulang jika database sudah berisi data
    if training_count > 0 or testing_count > 0:
        return

    try:
        df = pd.read_csv(CSV_FILE)
        stats = replace_dataset_from_dataframe(df)

        print("======================================")
        print("DATABASE INITIALIZATION")
        print("======================================")
        print("Dataset berhasil dimasukkan.")
        print("Total data      :", stats["total"])
        print("Training data   :", stats["training"])
        print("Testing data    :", stats["testing"])
        print("======================================")

    except Exception as e:
        print("Gagal import dataset:")
        print(e)


def initialize_database():
    """
    Menjalankan seluruh proses database.
    """

    create_tables()
    import_dataset()