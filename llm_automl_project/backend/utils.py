import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()  # Ensure environment variables are loaded

# ✅ Flexible Loader for 10+ formats
def load_dataset(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext == ".tsv":
            return pd.read_csv(file_path, sep="\t")
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif ext == ".json":
            return pd.read_json(file_path)
        elif ext == ".xml":
            return pd.read_xml(file_path)
        elif ext == ".parquet":
            return pd.read_parquet(file_path)
        elif ext == ".feather":
            return pd.read_feather(file_path)
        elif ext == ".sav":
            import pyreadstat
            df, _ = pyreadstat.read_sav(file_path)
            return df
        elif ext == ".dta":
            return pd.read_stata(file_path)
        elif ext == ".txt":
            return pd.read_csv(file_path, delimiter=None)
        elif ext == ".html":
            tables = pd.read_html(file_path)
            return tables[0] if tables else pd.DataFrame()
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        raise ValueError(f"❌ Error loading dataset: {e}")

# ✅ Get last column as default target
def get_target_column(df: pd.DataFrame) -> str:
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature and one target column.")
    return df.columns[-1]

# ✅ Dataset metadata
def print_dataset_info(df: pd.DataFrame) -> dict:
    return {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict()
    }

# ✅ Missing value handler
def impute_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return pd.DataFrame(X_imputed, columns=X.columns)

# ✅ Scaler
def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

# ✅ Feature selector
def select_top_features(X: pd.DataFrame, y: pd.Series, k: int = 5) -> pd.DataFrame:
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_new = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()]
    return pd.DataFrame(X_new, columns=selected_cols)

# ✅ Validate dataset
def validate_dataset(df: pd.DataFrame) -> list:
    issues = []
    if df.isnull().sum().sum() > 0:
        issues.append("⚠️ Missing values detected.")
    if df.shape[0] < 50:
        issues.append("⚠️ Dataset may be too small for reliable modeling.")
    if df.select_dtypes(include='object').shape[1] > 0:
        issues.append("⚠️ Non-numeric columns present. Consider encoding.")
    return issues

# ✅ Encode categoricals
def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        return X
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
    X = X.drop(columns=categorical_cols)
    X = pd.concat([X, encoded_df], axis=1)
    return X

# ✅ Email EDA Report
def send_email_report(to_email: str, subject: str, body: str, attachment_path: str):
    try:
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
        print(f"✅ EDA report emailed to {to_email}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to send email: {e}")
