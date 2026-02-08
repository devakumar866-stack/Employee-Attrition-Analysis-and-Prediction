# --- Single-row paste -> predict (self-contained page) ---

import os
import io
import csv
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Single Employee Attrition Check", page_icon="🧭", layout="centered")
st.header("Paste a Single Employee Row → Attrition Risk")

# 1) Load your trained model/pipeline
MODEL_PATH = "models/attrition_model.joblib"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Make sure the file exists.")
    st.stop()
model = joblib.load(MODEL_PATH)

# 2) RAW column order from your CSV (without 'Attrition')
#    Edit only if your CSV order is different.
COLUMN_ORDER = [
    "Age",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "Over18",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "StandardHours",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

st.caption("Paste exactly one row (no header). Tabs or commas are fine. Example:")
st.code(
    "32\tTravel_Frequently\t1005\tResearch & Development\t2\t2\tLife Sciences\t1\t8\t4\tMale\t79\t3\t1\tLaboratory Technician\t4\tSingle\t3068\t11864\t0\tY\tNo\t13\t3\t3\t80\t0\t8\t2\t2\t7\t7\t3\t6",
    language="text",
)

raw_line = st.text_area("Paste the row here:", height=90, placeholder="Paste one line from your CSV/Excel…")
threshold = st.slider("Decision threshold (High risk if probability ≥ threshold)", 0.05, 0.95, 0.40, 0.01)

# ---------- helpers ----------
def autodetect_delimiter(line: str):
    if "\t" in line: return "\t"
    if "," in line: return ","
    return None  # whitespace fallback

def parse_single_row(line: str):
    delim = autodetect_delimiter(line)
    if delim:
        reader = csv.reader(io.StringIO(line), delimiter=delim)
        values = next(reader)
    else:
        values = line.split()
    return [v.strip() for v in values]

def maybe_drop_attrition(values: list[str]) -> list[str]:
    # If the user pasted an 'Attrition' field as 2nd column (Yes/No), drop it.
    if len(values) == len(COLUMN_ORDER) + 1 and values[1].lower() in {"yes", "no", "y", "n"}:
        return values[:1] + values[2:]
    return values

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Convert numerics where possible; keep categoricals as strings
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            df[c] = df[c].astype(str).str.strip()
    return df

def load_training_feature_names(m):
    cols = getattr(m, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    # fallback to json saved during training
    json_path = "models/feature_columns.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None

def build_inference_matrix(row_df: pd.DataFrame, m):
    train_cols = load_training_feature_names(m)
    if train_cols is None:
        # If we can't find the model's training columns, try raw directly (may fail)
        return row_df, None, (
            "Training feature list not found. "
            "Save it as models/feature_columns.json (from X_train.columns) or use a model that exposes feature_names_in_."
        )

    # Detect if the model expects one-hot columns (presence of '_' or raw cols missing)
    expects_one_hot = any("_" in c for c in train_cols) or any(c not in row_df.columns for c in train_cols)

    if expects_one_hot:
        X = pd.get_dummies(row_df, drop_first=False)
        X = X.reindex(columns=train_cols, fill_value=0)
    else:
        # Model expects raw columns in specific order
        missing = [c for c in train_cols if c not in row_df.columns]
        if missing:
            return None, train_cols, f"Missing raw columns expected by model: {missing}"
        X = row_df[train_cols]

    return X, train_cols, None
# -----------------------------

if st.button("Predict"):
    if not raw_line.strip():
        st.error("Please paste a single row first.")
        st.stop()

    values = parse_single_row(raw_line.strip())
    values = maybe_drop_attrition(values)

    if len(values) != len(COLUMN_ORDER):
        st.error(
            f"Column count mismatch.\nExpected {len(COLUMN_ORDER)} values (no header), "
            f"but got {len(values)}.\n\n"
            "Tip: Ensure the row matches your CSV column order and does not include 'Attrition'."
        )
        st.stop()

    row_df = pd.DataFrame([values], columns=COLUMN_ORDER)
    row_df = coerce_types(row_df)

    # Build matrix the model expects (handles one-hot + alignment)
    X_infer, train_cols, err = build_inference_matrix(row_df, model)
    if err:
        st.error(err)
        st.stop()

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_infer)[0][1]
            label = "High risk" if proba >= threshold else "Low risk"
            st.success(f"Prediction: **{label}**")
            st.write(f"Probability of Attrition: **{proba:.2%}**")
        else:
            pred = model.predict(X_infer)[0]
            label = "High risk" if str(pred).lower() in {"1", "yes", "true"} else "Low risk"
            st.success(f"Prediction: **{label}**")
            st.caption("Model has no predict_proba; showing class only.")

        with st.expander("Debug: model vs input columns"):
            st.write("Model expects total columns:", 0 if train_cols is None else len(train_cols))
            st.write("Raw columns provided:", list(row_df.columns))
            st.write("Encoded/aligned columns sent to model (first 25):", list(X_infer.columns)[:25])

        with st.expander("See parsed input row"):
            st.dataframe(row_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")