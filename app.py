from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_from_directory
)

import math
import csv
from io import StringIO
import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from werkzeug.utils import secure_filename

from database import (
    initialize_database,
    get_db_connection,
    replace_dataset_from_dataframe,
    clear_dataset,
    inspect_dataset,
    backup_current_dataset,
    list_backups,
    REQUIRED_COLUMNS,
    BACKUP_DIR,
    DatasetImportError
)


app = Flask(__name__)

# Secret key digunakan untuk flash message
app.secret_key = "lung-cancer-research-system-2026"


# ============================================================
# DATABASE
# ============================================================

initialize_database()


# ============================================================
# KONSTANTA DATA
# ============================================================

FEATURES = [
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
    "chest_pain",
    "lung_cancer"
]


FEATURE_LABELS = {
    "gender": "Gender",
    "age": "Age",
    "smoking": "Smoking",
    "yellow_fingers": "Yellow Fingers",
    "anxiety": "Anxiety",
    "peer_pressure": "Peer Pressure",
    "chronic_disease": "Chronic Disease",
    "fatigue": "Fatigue",
    "allergy": "Allergy",
    "wheezing": "Wheezing",
    "alcohol_consuming": "Alcohol Consuming",
    "coughing": "Coughing",
    "shortness_of_breath": "Shortness of Breath",
    "swallowing_difficulty": "Swallowing Difficulty",
    "chest_pain": "Chest Pain",
    "lung_cancer": "Lung Cancer"
}


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "logistic": os.path.join(BASE_DIR, "logistic_regression.pkl"),
    "random_forest": os.path.join(BASE_DIR, "random_forest.pkl"),
    "scaler": os.path.join(BASE_DIR, "scaler.pkl"),
    "features": os.path.join(BASE_DIR, "feature_columns.pkl"),
}

ML_FEATURES = FEATURES[:-1]


def train_models():
    """Train Logistic Regression dan Random Forest dari training_data."""
    conn = get_db_connection()
    rows = conn.execute(
        f"SELECT {', '.join(FEATURES)} FROM training_data"
    ).fetchall()
    conn.close()

    if len(rows) < 2:
        raise ValueError("Data training minimal 2 record diperlukan.")

    df = pd.DataFrame([dict(row) for row in rows])
    X = df[ML_FEATURES].astype(float)
    y = df["lung_cancer"].astype(int)

    if y.nunique() < 2:
        raise ValueError("Data training harus memiliki kelas Lung Cancer 0 dan 1.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logistic = LogisticRegression(max_iter=1000, random_state=42)
    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    logistic.fit(X_scaled, y)
    random_forest.fit(X, y)

    joblib.dump(logistic, MODEL_FILES["logistic"])
    joblib.dump(random_forest, MODEL_FILES["random_forest"])
    joblib.dump(scaler, MODEL_FILES["scaler"])
    joblib.dump(ML_FEATURES, MODEL_FILES["features"])

    return len(df)


def load_models():
    missing = [
        name for name, path in MODEL_FILES.items()
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "File model belum tersedia: " + ", ".join(missing)
        )

    return (
        joblib.load(MODEL_FILES["logistic"]),
        joblib.load(MODEL_FILES["random_forest"]),
        joblib.load(MODEL_FILES["scaler"]),
    )


def evaluate_models():
    """Evaluasi kedua model menggunakan seluruh testing_data."""
    conn = get_db_connection()
    rows = conn.execute(
        f"SELECT {', '.join(FEATURES)} FROM testing_data"
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError("Testing data belum tersedia.")

    df = pd.DataFrame([dict(row) for row in rows])
    X = df[ML_FEATURES].astype(float)
    y = df["lung_cancer"].astype(int)

    logistic, random_forest, scaler = load_models()

    predictions = {
        "Logistic Regression": logistic.predict(scaler.transform(X)),
        "Random Forest": random_forest.predict(X),
    }

    results = []
    conn = get_db_connection()
    conn.execute("DELETE FROM model_evaluation")

    for name, pred in predictions.items():
        result = {
            "model_name": name,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1_score": f1_score(y, pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        }

        results.append(result)

        conn.execute(
            """
            INSERT INTO model_evaluation
            (model_name, accuracy, precision, recall, f1_score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                name,
                result["accuracy"],
                result["precision"],
                result["recall"],
                result["f1_score"],
            )
        )

    conn.commit()
    conn.close()

    return results


# ============================================================
# HELPER
# ============================================================

def get_int_form_value(name, default=0):
    """
    Mengambil nilai integer dari form.
    """

    value = request.form.get(name, "").strip()

    if value == "":
        return default

    try:
        return int(float(value))
    except ValueError:
        return default


def get_dataset_values():
    """
    Mengambil seluruh nilai fitur dari form.
    """

    values = []

    for feature in FEATURES:
        values.append(
            get_int_form_value(feature)
        )

    return values



# ============================================================
# APPLICATION SETTINGS
# ============================================================

DEFAULT_SETTINGS = {
    "risk_threshold": "50",
    "rows_per_page": "10",
    "admin_name": "Administrator"
}


def get_setting(key, default=None):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT setting_value FROM app_settings WHERE setting_key = ?",
        (key,)
    ).fetchone()
    conn.close()
    if row:
        return row["setting_value"]
    return DEFAULT_SETTINGS.get(key, default)


def set_setting(key, value):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO app_settings (setting_key, setting_value)
        VALUES (?, ?)
        ON CONFLICT(setting_key)
        DO UPDATE SET setting_value = excluded.setting_value
        """,
        (key, str(value))
    )
    conn.commit()
    conn.close()


@app.context_processor
def inject_global_settings():
    return {"admin_name": get_setting("admin_name", "Administrator")}


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    conn = get_db_connection()

    total_patients = conn.execute(
        "SELECT COUNT(*) FROM patients"
    ).fetchone()[0]

    total_training = conn.execute(
        "SELECT COUNT(*) FROM training_data"
    ).fetchone()[0]

    total_testing = conn.execute(
        "SELECT COUNT(*) FROM testing_data"
    ).fetchone()[0]

    total_predictions = conn.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    high_risk = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE overall_result = 'High Risk'
        """
    ).fetchone()[0]

    low_risk = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE overall_result = 'Low Risk'
        """
    ).fetchone()[0]

    conn.close()

    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        total_training=total_training,
        total_testing=total_testing,
        total_predictions=total_predictions,
        high_risk=high_risk,
        low_risk=low_risk
    )


# ============================================================
# PATIENT - READ
# ============================================================

@app.route("/patients")
def patients():

    search = request.args.get(
        "search",
        ""
    ).strip()

    page = request.args.get(
        "page",
        1,
        type=int
    )

    per_page = int(get_setting('rows_per_page', 10))

    if page < 1:
        page = 1

    conn = get_db_connection()

    if search:

        count_query = """
            SELECT COUNT(*)
            FROM patients
            WHERE patient_name LIKE ?
               OR patient_code LIKE ?
        """

        search_value = f"%{search}%"

        total = conn.execute(
            count_query,
            (search_value, search_value)
        ).fetchone()[0]

        offset = (page - 1) * per_page

        patients = conn.execute(
            """
            SELECT *
            FROM patients
            WHERE patient_name LIKE ?
               OR patient_code LIKE ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (
                search_value,
                search_value,
                per_page,
                offset
            )
        ).fetchall()

    else:

        total = conn.execute(
            "SELECT COUNT(*) FROM patients"
        ).fetchone()[0]

        offset = (page - 1) * per_page

        patients = conn.execute(
            """
            SELECT *
            FROM patients
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (
                per_page,
                offset
            )
        ).fetchall()

    conn.close()

    total_pages = max(
        1,
        math.ceil(total / per_page)
    )

    if page > total_pages:
        page = total_pages

    return render_template(
        "patients.html",
        patients=patients,
        search=search,
        page=page,
        total_pages=total_pages,
        total=total
    )


# ============================================================
# PATIENT - CREATE
# ============================================================

@app.route(
    "/patients/add",
    methods=["GET", "POST"]
)
def add_patient():

    if request.method == "POST":

        patient_code = request.form.get(
            "patient_code",
            ""
        ).strip()

        patient_name = request.form.get(
            "patient_name",
            ""
        ).strip()

        gender = request.form.get(
            "gender",
            ""
        )

        age = request.form.get(
            "age",
            ""
        )

        # Validasi
        if not patient_code or not patient_name:

            flash(
                "Patient code dan patient name wajib diisi.",
                "danger"
            )

            return render_template(
                "patient_form.html",
                mode="add",
                patient=None
            )

        try:
            gender = int(gender)
            age = int(age)

        except ValueError:

            flash(
                "Gender dan Age harus berupa angka yang valid.",
                "danger"
            )

            return render_template(
                "patient_form.html",
                mode="add",
                patient=None
            )

        conn = get_db_connection()

        try:

            conn.execute(
                """
                INSERT INTO patients (
                    patient_code,
                    patient_name,
                    gender,
                    age
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    patient_code,
                    patient_name,
                    gender,
                    age
                )
            )

            conn.commit()

            flash(
                "Data pasien berhasil ditambahkan.",
                "success"
            )

            return redirect(
                url_for("patients")
            )

        except Exception as e:

            if "UNIQUE" in str(e).upper():

                flash(
                    "Patient code sudah digunakan.",
                    "danger"
                )

            else:

                flash(
                    f"Gagal menambahkan data: {e}",
                    "danger"
                )

        finally:
            conn.close()

    return render_template(
        "patient_form.html",
        mode="add",
        patient=None
    )


# ============================================================
# PATIENT - EDIT
# ============================================================

@app.route(
    "/patients/edit/<int:patient_id>",
    methods=["GET", "POST"]
)
def edit_patient(patient_id):

    conn = get_db_connection()

    patient = conn.execute(
        """
        SELECT *
        FROM patients
        WHERE id = ?
        """,
        (patient_id,)
    ).fetchone()

    if not patient:

        conn.close()

        flash(
            "Data pasien tidak ditemukan.",
            "danger"
        )

        return redirect(
            url_for("patients")
        )

    if request.method == "POST":

        patient_code = request.form.get(
            "patient_code",
            ""
        ).strip()

        patient_name = request.form.get(
            "patient_name",
            ""
        ).strip()

        gender = request.form.get(
            "gender",
            ""
        )

        age = request.form.get(
            "age",
            ""
        )

        try:

            gender = int(gender)
            age = int(age)

            conn.execute(
                """
                UPDATE patients
                SET
                    patient_code = ?,
                    patient_name = ?,
                    gender = ?,
                    age = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    patient_code,
                    patient_name,
                    gender,
                    age,
                    patient_id
                )
            )

            conn.commit()

            flash(
                "Data pasien berhasil diperbarui.",
                "success"
            )

            conn.close()

            return redirect(
                url_for("patients")
            )

        except Exception as e:

            flash(
                f"Gagal memperbarui data: {e}",
                "danger"
            )

    conn.close()

    return render_template(
        "patient_form.html",
        mode="edit",
        patient=patient
    )


# ============================================================
# PATIENT - DELETE
# ============================================================

@app.route(
    "/patients/delete/<int:patient_id>",
    methods=["POST"]
)
def delete_patient(patient_id):

    conn = get_db_connection()

    conn.execute(
        """
        DELETE FROM patients
        WHERE id = ?
        """,
        (patient_id,)
    )

    conn.commit()
    conn.close()

    flash(
        "Data pasien berhasil dihapus.",
        "success"
    )

    return redirect(
        url_for("patients")
    )


# ============================================================
# TRAINING DATA
# ============================================================

@app.route("/training")
def training():

    return dataset_page(
        "training"
    )


# ============================================================
# TESTING DATA
# ============================================================

@app.route("/testing")
def testing():

    return dataset_page(
        "testing"
    )


# ============================================================
# DATASET PAGE HELPER
# ============================================================

def dataset_page(dataset_type):

    search = request.args.get(
        "search",
        ""
    ).strip()

    risk_filter = request.args.get(
        "risk",
        "all"
    )

    page = request.args.get(
        "page",
        1,
        type=int
    )

    per_page = 10

    if page < 1:
        page = 1

    table_name = (
        "training_data"
        if dataset_type == "training"
        else "testing_data"
    )

    conn = get_db_connection()

    conditions = []
    parameters = []

    # Search
    if search:

        conditions.append(
            """
            CAST(id AS TEXT) LIKE ?
            OR CAST(age AS TEXT) LIKE ?
            """
        )

        search_value = f"%{search}%"

        parameters.extend([
            search_value,
            search_value
        ])

    # Risk filter
    if risk_filter in ["0", "1"]:

        conditions.append(
            "lung_cancer = ?"
        )

        parameters.append(
            int(risk_filter)
        )

    where_clause = ""

    if conditions:

        where_clause = (
            "WHERE " +
            " AND ".join(
                f"({condition})"
                for condition in conditions
            )
        )

    total = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        {where_clause}
        """,
        parameters
    ).fetchone()[0]

    offset = (page - 1) * per_page

    rows = conn.execute(
        f"""
        SELECT *
        FROM {table_name}
        {where_clause}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """,
        parameters + [
            per_page,
            offset
        ]
    ).fetchall()

    conn.close()

    total_pages = max(
        1,
        math.ceil(total / per_page)
    )

    if page > total_pages:
        page = total_pages

    title = (
        "Training Data"
        if dataset_type == "training"
        else "Testing Data"
    )

    return render_template(
        "dataset.html",
        dataset_type=dataset_type,
        page_title=title,
        rows=rows,
        search=search,
        risk_filter=risk_filter,
        page=page,
        total_pages=total_pages,
        total=total
    )


# ============================================================
# TRAINING - ADD
# ============================================================

@app.route(
    "/training/add",
    methods=["GET", "POST"]
)
def add_training():

    return add_dataset_record(
        "training"
    )


# ============================================================
# TESTING - ADD
# ============================================================

@app.route(
    "/testing/add",
    methods=["GET", "POST"]
)
def add_testing():

    return add_dataset_record(
        "testing"
    )


# ============================================================
# ADD DATASET HELPER
# ============================================================

def add_dataset_record(dataset_type):

    table_name = (
        "training_data"
        if dataset_type == "training"
        else "testing_data"
    )

    if request.method == "POST":

        values = get_dataset_values()

        placeholders = ",".join(
            ["?"] * len(FEATURES)
        )

        columns = ",".join(
            FEATURES
        )

        conn = get_db_connection()

        try:

            conn.execute(
                f"""
                INSERT INTO {table_name}
                ({columns})
                VALUES ({placeholders})
                """,
                values
            )

            conn.commit()

            flash(
                f"Data {dataset_type} berhasil ditambahkan.",
                "success"
            )

            return redirect(
                url_for(dataset_type)
            )

        except Exception as e:

            flash(
                f"Gagal menambahkan data: {e}",
                "danger"
            )

        finally:

            conn.close()

    return render_template(
        "dataset_form.html",
        mode="add",
        dataset_type=dataset_type,
        row=None
    )


# ============================================================
# TRAINING - EDIT
# ============================================================

@app.route(
    "/training/edit/<int:record_id>",
    methods=["GET", "POST"]
)
def edit_training(record_id):

    return edit_dataset_record(
        "training",
        record_id
    )


# ============================================================
# TESTING - EDIT
# ============================================================

@app.route(
    "/testing/edit/<int:record_id>",
    methods=["GET", "POST"]
)
def edit_testing(record_id):

    return edit_dataset_record(
        "testing",
        record_id
    )


# ============================================================
# EDIT DATASET HELPER
# ============================================================

def edit_dataset_record(
    dataset_type,
    record_id
):

    table_name = (
        "training_data"
        if dataset_type == "training"
        else "testing_data"
    )

    conn = get_db_connection()

    row = conn.execute(
        f"""
        SELECT *
        FROM {table_name}
        WHERE id = ?
        """,
        (record_id,)
    ).fetchone()

    if not row:

        conn.close()

        flash(
            "Data tidak ditemukan.",
            "danger"
        )

        return redirect(
            url_for(dataset_type)
        )

    if request.method == "POST":

        values = get_dataset_values()

        set_clause = ", ".join(
            f"{feature} = ?"
            for feature in FEATURES
        )

        try:

            conn.execute(
                f"""
                UPDATE {table_name}
                SET {set_clause}
                WHERE id = ?
                """,
                values + [record_id]
            )

            conn.commit()

            flash(
                f"Data {dataset_type} berhasil diperbarui.",
                "success"
            )

            conn.close()

            return redirect(
                url_for(dataset_type)
            )

        except Exception as e:

            flash(
                f"Gagal memperbarui data: {e}",
                "danger"
            )

    conn.close()

    return render_template(
        "dataset_form.html",
        mode="edit",
        dataset_type=dataset_type,
        row=row
    )


# ============================================================
# TRAINING - DELETE
# ============================================================

@app.route(
    "/training/delete/<int:record_id>",
    methods=["POST"]
)
def delete_training(record_id):

    return delete_dataset_record(
        "training",
        record_id
    )


# ============================================================
# TESTING - DELETE
# ============================================================

@app.route(
    "/testing/delete/<int:record_id>",
    methods=["POST"]
)
def delete_testing(record_id):

    return delete_dataset_record(
        "testing",
        record_id
    )


# ============================================================
# DELETE DATASET HELPER
# ============================================================

def delete_dataset_record(
    dataset_type,
    record_id
):

    table_name = (
        "training_data"
        if dataset_type == "training"
        else "testing_data"
    )

    conn = get_db_connection()

    conn.execute(
        f"""
        DELETE FROM {table_name}
        WHERE id = ?
        """,
        (record_id,)
    )

    conn.commit()
    conn.close()

    flash(
        f"Data {dataset_type} berhasil dihapus.",
        "success"
    )

    return redirect(
        url_for(dataset_type)
    )


# ============================================================
# TRAIN MODEL
# ============================================================

@app.route("/train-model", methods=["POST"])
def train_model():
    try:
        total = train_models()
        evaluate_models()

        flash(
            f"Model berhasil dilatih menggunakan {total} data training "
            "dan dievaluasi menggunakan testing data.",
            "success"
        )

        return redirect(url_for("evaluation"))

    except Exception as e:
        flash(f"Training model gagal: {e}", "danger")
        return redirect(url_for("dashboard"))


# ============================================================
# PREDICTION
# ============================================================

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    conn = get_db_connection()

    patients = conn.execute(
        """
        SELECT id, patient_code, patient_name, gender, age
        FROM patients
        ORDER BY patient_name
        """
    ).fetchall()

    conn.close()

    result = None
    form_data = {}

    if request.method == "POST":

        try:
            patient_id = request.form.get("patient_id", "").strip()
            patient = None

            if patient_id:
                conn = get_db_connection()

                patient = conn.execute(
                    "SELECT * FROM patients WHERE id = ?",
                    (int(patient_id),)
                ).fetchone()

                conn.close()

            # Jika pasien dipilih, Gender dan Age otomatis diambil dari
            # data pasien. Nilai ini juga dipakai saat prediksi agar
            # konsisten dengan data pasien di database.
            if patient:
                form_data["gender"] = int(patient["gender"]) if patient["gender"] is not None else 0
                form_data["age"] = int(patient["age"]) if patient["age"] is not None else 0

            for feature in ML_FEATURES:
                if feature not in ("gender", "age"):
                    form_data[feature] = get_int_form_value(feature)

            values = [form_data[feature] for feature in ML_FEATURES]

            # Model/scaler dilatih dengan nama kolom uppercase.
            # Gunakan feature_columns.pkl agar urutan dan nama fitur
            # selalu sama dengan saat training.
            feature_columns = joblib.load(MODEL_FILES["features"])
            X = pd.DataFrame([values], columns=feature_columns)

            logistic, random_forest, scaler = load_models()

            logistic_x = scaler.transform(X)

            logistic_prediction = int(
                logistic.predict(logistic_x)[0]
            )

            logistic_probability = (
                float(logistic.predict_proba(logistic_x)[0][1])
                * 100
            )

            random_forest_prediction = int(
                random_forest.predict(X)[0]
            )

            random_forest_probability = (
                float(random_forest.predict_proba(X)[0][1])
                * 100
            )

            average_probability = (
                logistic_probability +
                random_forest_probability
            ) / 2

            overall_result = (
                "High Risk"
                if average_probability >= float(get_setting("risk_threshold", 50))
                else "Low Risk"
            )

            patient_name = (
                patient["patient_name"]
                if patient
                else "Manual Prediction"
            )

            conn = get_db_connection()

            conn.execute(
                """
                INSERT INTO predictions (
                    patient_id,
                    patient_name,
                    logistic_prediction,
                    logistic_probability,
                    random_forest_prediction,
                    random_forest_probability,
                    overall_result,
                    average_probability
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(patient_id) if patient_id else None,
                    patient_name,
                    logistic_prediction,
                    logistic_probability,
                    random_forest_prediction,
                    random_forest_probability,
                    overall_result,
                    average_probability
                )
            )

            conn.commit()
            conn.close()

            result = {
                "patient_name": patient_name,
                "logistic_prediction": logistic_prediction,
                "logistic_probability": round(
                    logistic_probability, 2
                ),
                "random_forest_prediction": random_forest_prediction,
                "random_forest_probability": round(
                    random_forest_probability, 2
                ),
                "average_probability": round(
                    average_probability, 2
                ),
                "overall_result": overall_result,
            }

        except Exception as e:
            flash(f"Prediction gagal: {e}", "danger")

    return render_template(
        "prediction.html",
        patients=patients,
        features=ML_FEATURES,
        feature_labels=FEATURE_LABELS,
        form_data=form_data,
        result=result,
        page_title="Prediction"
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():
    conn = get_db_connection()

    predictions = conn.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY prediction_date DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions,
        page_title="Prediction History"
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

@app.route("/evaluation")
def evaluation():

    try:
        results = evaluate_models()

        conn = get_db_connection()

        evaluations = conn.execute(
            """
            SELECT *
            FROM model_evaluation
            ORDER BY id DESC
            """
        ).fetchall()

        conn.close()

    except Exception as e:

        results = []
        evaluations = []

        flash(
            f"Evaluasi belum dapat dilakukan: {e}",
            "warning"
        )

    return render_template(
        "evaluation.html",
        evaluations=evaluations,
        results=results,
        page_title="Model Evaluation"
    )


# ============================================================
# REPORT
# ============================================================

@app.route("/reports")
def reports():
    conn = get_db_connection()

    total_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_training = conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
    total_testing = conn.execute("SELECT COUNT(*) FROM testing_data").fetchone()[0]
    total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    high_risk = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE overall_result = 'High Risk'"
    ).fetchone()[0]
    low_risk = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE overall_result = 'Low Risk'"
    ).fetchone()[0]

    evaluations = conn.execute(
        """
        SELECT model_name, accuracy, precision, recall, f1_score, created_at
        FROM model_evaluation
        ORDER BY id DESC
        """
    ).fetchall()

    recent_predictions = conn.execute(
        """
        SELECT patient_name, logistic_probability,
               random_forest_probability, average_probability,
               overall_result, prediction_date
        FROM predictions
        ORDER BY prediction_date DESC
        LIMIT 10
        """
    ).fetchall()

    conn.close()

    return render_template(
        "reports.html",
        page_title="Reports",
        total_patients=total_patients,
        total_training=total_training,
        total_testing=total_testing,
        total_predictions=total_predictions,
        high_risk=high_risk,
        low_risk=low_risk,
        evaluations=evaluations,
        recent_predictions=recent_predictions,
        risk_threshold=float(get_setting("risk_threshold", 50))
    )


@app.route("/reports/export")
def export_report():
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT prediction_date, patient_name,
               logistic_probability, random_forest_probability,
               average_probability, overall_result
        FROM predictions
        ORDER BY prediction_date DESC
        """
    ).fetchall()
    conn.close()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Prediction Date", "Patient Name",
        "Logistic Probability (%)", "Random Forest Probability (%)",
        "Average Probability (%)", "Overall Result"
    ])

    for row in rows:
        writer.writerow([
            row["prediction_date"], row["patient_name"],
            row["logistic_probability"], row["random_forest_probability"],
            row["average_probability"], row["overall_result"]
        ])

    return app.response_class(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=lung_cancer_prediction_report.csv"}
    )


# ============================================================
# DATASET UPLOAD / RESET
# ============================================================

ALLOWED_DATASET_EXTENSIONS = {"csv"}
PENDING_UPLOAD_PATH = os.path.join(BASE_DIR, "pending_dataset_upload.csv")


def allowed_dataset_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_DATASET_EXTENSIONS
    )


def _retrain_after_dataset_change():
    """
    Melatih ulang kedua model dan mengevaluasinya.
    Mengembalikan (berhasil: bool, pesan: str)
    """

    try:
        total_trained = train_models()
        evaluate_models()
        return True, (
            f"Model berhasil dilatih ulang menggunakan "
            f"{total_trained} data training."
        )
    except Exception as e:
        return False, f"Retrain model gagal: {e}"


@app.route("/dataset/upload", methods=["POST"])
def upload_dataset():
    file = request.files.get("dataset_file")

    if not file or file.filename == "":
        flash("Pilih file CSV terlebih dahulu.", "danger")
        return redirect(url_for("settings"))

    filename = secure_filename(file.filename)

    if not allowed_dataset_file(filename):
        flash("Format file tidak didukung. Gunakan file .csv.", "danger")
        return redirect(url_for("settings"))

    try:
        df = pd.read_csv(file)
    except Exception as e:
        flash(f"Gagal membaca file CSV: {e}", "danger")
        return redirect(url_for("settings"))

    info = inspect_dataset(df)

    if not info["valid"]:
        flash(
            "Dataset gagal diimport. Kolom yang hilang: "
            + ", ".join(info["missing_columns"]),
            "danger"
        )
        return redirect(url_for("settings"))

    if not info["can_train"]:
        flash(
            "Dataset tidak valid: kolom LUNG_CANCER harus memiliki "
            "kedua kelas (positif dan negatif) agar bisa dibagi "
            "menjadi data training dan testing.",
            "danger"
        )
        return redirect(url_for("settings"))

    # Simpan file sementara sampai user konfirmasi di halaman preview
    df.to_csv(PENDING_UPLOAD_PATH, index=False)
    session["pending_dataset_filename"] = filename

    return render_template(
        "dataset_preview.html",
        page_title="Preview Dataset",
        original_filename=filename,
        total_rows=info["total_rows"],
        class_counts=info["class_counts"],
        preview_rows=info["preview_rows"],
        preview_columns=REQUIRED_COLUMNS
    )


@app.route("/dataset/upload/confirm", methods=["POST"])
def confirm_dataset_upload():
    if not os.path.exists(PENDING_UPLOAD_PATH):
        flash(
            "Tidak ada dataset yang menunggu konfirmasi. "
            "Silakan upload ulang.",
            "danger"
        )
        return redirect(url_for("settings"))

    try:
        df = pd.read_csv(PENDING_UPLOAD_PATH)

        # Backup dataset lama sebelum diganti
        backup_info = backup_current_dataset()

        stats = replace_dataset_from_dataframe(
            df,
            save_as_default_csv=True
        )

        message = (
            "Dataset berhasil diperbarui. Total {total} baris "
            "({training} training, {testing} testing).".format(**stats)
        )

        if backup_info:
            message += (
                f" Dataset lama dibackup sebagai {backup_info['filename']}."
            )

        flash(message, "success")

        # Retrain otomatis menggunakan dataset baru
        success, retrain_message = _retrain_after_dataset_change()
        flash(retrain_message, "success" if success else "danger")

    except DatasetImportError as e:
        flash(f"Dataset gagal diimport: {e}", "danger")
        return redirect(url_for("settings"))

    except Exception as e:
        flash(f"Gagal memproses dataset: {e}", "danger")
        return redirect(url_for("settings"))

    finally:
        if os.path.exists(PENDING_UPLOAD_PATH):
            os.remove(PENDING_UPLOAD_PATH)
        session.pop("pending_dataset_filename", None)

    return redirect(url_for("evaluation"))


@app.route("/dataset/upload/cancel", methods=["POST"])
def cancel_dataset_upload():
    if os.path.exists(PENDING_UPLOAD_PATH):
        os.remove(PENDING_UPLOAD_PATH)

    session.pop("pending_dataset_filename", None)
    flash("Upload dataset dibatalkan.", "success")
    return redirect(url_for("settings"))


@app.route("/dataset/clear", methods=["POST"])
def clear_dataset_route():
    try:
        backup_info = backup_current_dataset()
        clear_dataset()

        message = "Seluruh dataset training dan testing berhasil dihapus."

        if backup_info:
            message += (
                f" Data lama dibackup sebagai {backup_info['filename']}."
            )
        else:
            message += " (Tidak ada data untuk dibackup.)"

        flash(message, "success")

    except Exception as e:
        flash(f"Gagal menghapus dataset: {e}", "danger")

    return redirect(url_for("settings"))


@app.route("/dataset/backups/<path:filename>")
def download_backup(filename):
    safe_name = secure_filename(filename)
    full_path = os.path.join(BACKUP_DIR, safe_name)

    if not os.path.exists(full_path):
        flash("File backup tidak ditemukan.", "danger")
        return redirect(url_for("settings"))

    return send_from_directory(
        BACKUP_DIR,
        safe_name,
        as_attachment=True
    )


# ============================================================
# SETTINGS
# ============================================================

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        try:
            threshold = float(request.form.get("risk_threshold", "50"))
            rows_per_page = int(request.form.get("rows_per_page", "10"))
            admin_name_value = request.form.get("admin_name", "").strip()

            if not 1 <= threshold <= 99:
                raise ValueError("Risk threshold harus berada antara 1 sampai 99.")
            if rows_per_page not in (5, 10, 20, 50):
                raise ValueError("Jumlah baris per halaman tidak valid.")
            if not admin_name_value:
                raise ValueError("Nama administrator tidak boleh kosong.")

            set_setting("risk_threshold", threshold)
            set_setting("rows_per_page", rows_per_page)
            set_setting("admin_name", admin_name_value)

            flash("Pengaturan berhasil disimpan.", "success")
            return redirect(url_for("settings"))

        except Exception as e:
            flash(f"Pengaturan gagal disimpan: {e}", "danger")

    model_status = {
        "Logistic Regression": os.path.exists(MODEL_FILES["logistic"]),
        "Random Forest": os.path.exists(MODEL_FILES["random_forest"]),
        "Standard Scaler": os.path.exists(MODEL_FILES["scaler"]),
        "Feature Columns": os.path.exists(MODEL_FILES["features"])
    }

    conn = get_db_connection()
    training_count = conn.execute(
        "SELECT COUNT(*) FROM training_data"
    ).fetchone()[0]
    testing_count = conn.execute(
        "SELECT COUNT(*) FROM testing_data"
    ).fetchone()[0]
    conn.close()

    return render_template(
        "settings.html",
        page_title="Settings",
        risk_threshold=get_setting("risk_threshold", 50),
        rows_per_page=get_setting("rows_per_page", 10),
        admin_name=get_setting("admin_name", "Administrator"),
        model_status=model_status,
        database_path="lung_cancer.db",
        training_count=training_count,
        testing_count=testing_count,
        dataset_backups=list_backups()
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
