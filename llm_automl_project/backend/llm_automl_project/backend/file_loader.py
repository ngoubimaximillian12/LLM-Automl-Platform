import pandas as pd
import os

def load_file(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    elif ext == ".json":
        return pd.read_json(file_path)
    elif ext == ".parquet":
        return pd.read_parquet(file_path)
    elif ext == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    elif ext == ".txt":
        return pd.read_csv(file_path, delimiter=None)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
