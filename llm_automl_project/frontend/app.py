import streamlit as st
st.set_page_config(page_title="LLM AutoML", layout="wide")

import os
import sys
import pandas as pd
import requests


# Detect if running inside Docker (optional, e.g. by env var)
IN_DOCKER = os.getenv("IN_DOCKER", "0") == "1"

# Backend URL - switch based on environment
BACKEND_URL = "http://backend:8000" if IN_DOCKER else "http://127.0.0.1:8000"

# Setup backend path (optional, for local imports)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

deepseek_fallback = None
import_error = None
try:
    from llm_generator import deepseek_fallback
    from eda_email import send_eda_email
    from fairness_charts import plot_fairness_metrics
    from data_preview_tab import show_data_preview
    from nlp_cleaner_tab import run_nlp_cleaner_tab
except ImportError as e:
    import_error = f"❌ Import failed: {e}"

if import_error:
    st.error(import_error)

EDA_DIR = "data/eda_report"

st.title("🤖 LLM AutoML Platform")

tabs = st.tabs([
    "📁 Upload", "📈 Fairness", "📤 Email EDA", "🧠 Fallback",
    "📋 Preview", "🧹 NLP Cleaner"
])

with tabs[0]:
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json", "parquet"])
    df_preview = None

    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            if uploaded_file.name.endswith(".csv"):
                df_preview = pd.read_csv(file_path)
            elif uploaded_file.name.endswith(".xlsx"):
                df_preview = pd.read_excel(file_path)
            elif uploaded_file.name.endswith(".json"):
                df_preview = pd.read_json(file_path)
            elif uploaded_file.name.endswith(".parquet"):
                df_preview = pd.read_parquet(file_path)
            st.success(f"✅ File '{uploaded_file.name}' uploaded.")
            st.dataframe(df_preview.head(100))
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

        if st.button("🚀 Train Model + Generate EDA"):
            with st.spinner("Analyzing & training..."):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/train-model/",
                        params={"file_name": uploaded_file.name},
                        timeout=60
                    )
                    if res.status_code == 200:
                        st.success("✅ Model & EDA ready.")
                        st.json(res.json())
                        if os.path.exists(EDA_DIR):
                            for img in os.listdir(EDA_DIR):
                                if img.endswith(".png"):
                                    st.image(os.path.join(EDA_DIR, img), use_column_width=True)
                    else:
                        st.error(f"❌ Backend error {res.status_code}: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.warning("⚠️ Backend not reachable. Using LLM fallback...")
                    if deepseek_fallback:
                        with open(file_path, "r") as f:
                            head = f.read(3000)
                        try:
                            reply = deepseek_fallback(f"Analyze this dataset:\n{head}")
                            st.markdown(reply)
                        except Exception as e:
                            st.error(f"❌ LLM Fallback failed: {e}")
                    else:
                        st.error("❌ Fallback not available.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

# The rest of your tabs unchanged...
# ...

with tabs[5]:
    st.header("🧹 NLP Cleaner & Profiler")
    run_nlp_cleaner_tab()
