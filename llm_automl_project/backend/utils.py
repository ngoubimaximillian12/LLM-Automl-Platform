import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import smtplib
from email.message import EmailMessage
import os

def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

def get_target_column(df: pd.DataFrame) -> str:
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature and one target column.")
    return df.columns[-1]

def print_dataset_info(df: pd.DataFrame) -> dict:
    info = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict()
    }
    return info

def impute_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return pd.DataFrame(X_imputed, columns=X.columns)

def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def select_top_features(X: pd.DataFrame, y: pd.Series, k: int = 5) -> pd.DataFrame:
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_new = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()]
    return pd.DataFrame(X_new, columns=selected_cols)

def validate_dataset(df: pd.DataFrame) -> list:
    issues = []
    if df.isnull().sum().sum() > 0:
        issues.append("Missing values detected.")
    if df.shape[0] < 50:
        issues.append("Dataset may be too small for reliable modeling.")
    if df.select_dtypes(include='object').shape[1] > 0:
        issues.append("Non-numeric columns detected. Consider encoding.")
    return issues

def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        return X
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
    X = X.drop(columns=categorical_cols)
    X = pd.concat([X, encoded_df], axis=1)
    return X

def send_email_report(to_email: str, subject: str, body: str, attachment_path: str):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("EMAIL_ADDRESS")
    msg["To"] = to_email
    msg.set_content(body)
    with open(attachment_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(attachment_path)
        msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_ADDRESS"), os.getenv("EMAIL_PASSWORD"))
        smtp.send_message(msg)
