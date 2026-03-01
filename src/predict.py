import joblib
import pandas as pd
from pathlib import Path

# Load saved model and column structure

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Build correct paths
model_path = BASE_DIR / "models" / "model.pkl"
columns_path = BASE_DIR / "models" / "columns.pkl"

# Load saved model and columns
model = joblib.load(model_path)
columns = joblib.load(columns_path)


import joblib
import pandas as pd
from pathlib import Path

# Get project root
BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "model.pkl"
columns_path = BASE_DIR / "models" / "columns.pkl"

model = joblib.load(model_path)
columns = joblib.load(columns_path)


def predict(input_dict):

    df = pd.DataFrame([input_dict])

    # Ordinal encoding
    df['person_education'] = df['person_education'].map({
        'Doctorate': 5,
        'Master': 4,
        'Bachelor': 3,
        'High School': 2,
        'Associate': 1
    })

    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({
        'Yes': 1,
        'No': 0
    })

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability