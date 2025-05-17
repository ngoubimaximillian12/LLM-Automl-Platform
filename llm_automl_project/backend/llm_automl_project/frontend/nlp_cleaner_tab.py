import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from profanity_check import predict
from langdetect import detect
from scipy.stats import entropy, kurtosis, skew
from sklearn.impute import SimpleImputer

# ✅ Optional LLM fallback
try:
    from backend.llm_generator import deepseek_fallback
except:
    deepseek_fallback = None

st.set_page_config(layout="wide")
st.subheader("🧹 NLP Data Cleaner, Profiler, and Validator")

DATA_DIR = "data"
files = [f for f in os.listdir(DATA_DIR) if f.endswith((".csv", ".json", ".xlsx"))]

if not files:
    st.warning("No uploaded datasets found.")
    st.stop()

selected_file = st.selectbox("Choose a file", files)

# Load data
if selected_file.endswith(".csv"):
    df = pd.read_csv(os.path.join(DATA_DIR, selected_file))
elif selected_file.endswith(".xlsx"):
    df = pd.read_excel(os.path.join(DATA_DIR, selected_file))
elif selected_file.endswith(".json"):
    df = pd.read_json(os.path.join(DATA_DIR, selected_file))

text_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include='number').columns.tolist()

# ---------- Auto Data Profiling ----------
st.markdown("### 📊 Data Profiling")
st.json({
    "Missing Values": int(df.isnull().sum().sum()),
    "Duplicates": int(df.duplicated().sum()),
    "Numeric Columns": len(num_cols),
    "Text Columns": len(text_cols),
    "Memory Usage (MB)": round(df.memory_usage().sum() / 1e6, 2)
})

if num_cols:
    st.markdown("### 📈 Skewness & Kurtosis")
    st.json({
        col: {"skewness": float(skew(df[col].dropna())), "kurtosis": float(kurtosis(df[col].dropna()))}
        for col in num_cols if df[col].nunique() > 1
    })

# ---------- Custom Rule Checker ----------
st.markdown("### 🧾 Custom Rule Checker")
rule_warnings = []

if "email" in df.columns:
    invalids = (~df["email"].astype(str).str.contains("@")).sum()
    rule_warnings.append(f"Column 'email' has {invalids} invalid addresses.")

if "age" in df.columns:
    negatives = (df["age"] < 0).sum()
    rule_warnings.append(f"Column 'age' has {negatives} negative values.")

for msg in rule_warnings:
    st.warning(msg)

# ---------- LLM Semantic Labeler ----------
if deepseek_fallback:
    st.markdown("### 🤖 Column Description via LLM")
    try:
        preview = df.head(2).to_markdown()
        response = deepseek_fallback(f"Explain the meaning of these columns:\n{preview}")
        st.code(response)
    except Exception as e:
        st.error(f"LLM Description failed: {e}")

# ---------- Text Cleaning ----------
if text_cols:
    st.markdown("## ✨ Text Column Cleaner")
    col = st.selectbox("Text column to clean", text_cols)

    df_cleaned = df.copy()
    cleaned_texts = []
    errors = []

    def detect_offensive(text): return bool(predict([text])[0])
    def correct_grammar(text):
        if deepseek_fallback:
            try:
                return deepseek_fallback(f"Fix grammar and fluency:\n{text}")
            except:
                return text
        return text
    def clean_entities(text): return re.sub(r"\b(Mr\.|Mrs\.|Dr\.|Prof\.|Sir)\s\w+", "<NAME>", text)

    limit = st.slider("Rows to clean", 1, min(500, len(df)), 10)

    if st.button("🚀 Run Cleaning"):
        progress = st.progress(0)
        for i, row in df.head(limit).iterrows():
            text = str(row[col])
            if detect_offensive(text): errors.append(("Offensive", i))
            text = clean_entities(text)
            text = correct_grammar(text)
            cleaned_texts.append(text)
            progress.progress((i + 1) / limit)

        df_cleaned[col + "_cleaned"] = cleaned_texts
        st.success("✅ Cleaning Done")
        st.dataframe(df_cleaned[[col, col + "_cleaned"]])

        st.markdown("### 📑 Cleaning Summary")
        st.json({
            "Rows Processed": limit,
            "Offensive Content": len([e for e in errors if e[0] == "Offensive"]),
            "Entity Tags Rewritten": limit,
        })

        st.download_button("⬇️ Download Cleaned CSV", df_cleaned.to_csv(index=False),
                           file_name="cleaned_data.csv", mime="text/csv")

else:
    st.warning("⚠️ No text columns found for NLP cleaning.")
