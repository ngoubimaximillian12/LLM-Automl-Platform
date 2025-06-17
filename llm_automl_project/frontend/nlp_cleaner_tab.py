import os
import streamlit as st
import pandas as pd
import re
from better_profanity import profanity
from scipy.stats import skew, kurtosis

# Load profanity words once
profanity.load_censor_words()

try:
    from backend.llm_generator import deepseek_fallback
except ImportError:
    deepseek_fallback = None


def run_nlp_cleaner_tab():
    st.subheader("🧹 NLP Data Cleaner, Profiler, and Validator")

    DATA_DIR = "data"
    if not os.path.exists(DATA_DIR):
        st.warning("No uploaded datasets found.")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".csv", ".json", ".xlsx"))]

    if not files:
        st.warning("No uploaded datasets found.")
        return

    selected_file = st.selectbox("Choose a file", files)

    # Load data
    file_path = os.path.join(DATA_DIR, selected_file)
    if selected_file.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif selected_file.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif selected_file.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        st.error("Unsupported file format.")
        return

    text_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    # Data Profiling
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

    # Custom Rule Checker
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

    # LLM Semantic Labeler (optional)
    if deepseek_fallback:
        st.markdown("### 🤖 Column Description via LLM")
        try:
            preview = df.head(2).to_markdown()
            response = deepseek_fallback(f"Explain the meaning of these columns:\n{preview}")
            st.code(response)
        except Exception as e:
            st.error(f"LLM Description failed: {e}")

    # Text Cleaning
    if text_cols:
        st.markdown("## ✨ Text Column Cleaner")
        col = st.selectbox("Text column to clean", text_cols)

        df_cleaned = df.copy()
        cleaned_texts = []
        errors = []

        def detect_offensive(text):
            return profanity.contains_profanity(text)

        def correct_grammar(text):
            if deepseek_fallback:
                try:
                    return deepseek_fallback(f"Fix grammar and fluency:\n{text}")
                except:
                    return text
            return text

        def clean_entities(text):
            return re.sub(r"\b(Mr\.|Mrs\.|Dr\.|Prof\.|Sir)\s\w+", "<NAME>", text)

        limit = st.slider("Rows to clean", 1, min(500, len(df)), 10)

        if st.button("🚀 Run Cleaning"):
            progress = st.progress(0)
            for i, row in df.head(limit).iterrows():
                text = str(row[col])
                if detect_offensive(text):
                    errors.append(("Offensive", i))
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
