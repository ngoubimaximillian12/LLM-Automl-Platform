import os
import sys
import pandas as pd
import streamlit as st
import requests
from PIL import Image

# ✅ Ensure backend directory is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ✅ Import backend fallback logic and utils
try:
    from llm_generator import deepseek_fallback
    from eda_email import send_eda_email
    from fairness_charts import plot_fairness_metrics
    from data_preview_tab import show_data_preview  # <-- this must exist
except ImportError as e:
    st.error(f"❌ Import failed: {e}")
    deepseek_fallback = None

# 📁 Constants
EDA_DIR = "data/eda_report"
BACKEND_URL = "http://127.0.0.1:8000"

# ----------------------------- 📊 EDA Chart Viewer -----------------------------
def show_eda_images():
    st.subheader("📊 Exploratory Data Analysis Report")
    if not os.path.exists(EDA_DIR):
        st.warning("⚠️ EDA report not found. Please train a model first.")
        return
    for img_file in sorted(os.listdir(EDA_DIR)):
        if img_file.endswith(".png"):
            st.image(os.path.join(EDA_DIR, img_file), caption=img_file, use_column_width=True)

# ----------------------------- 📈 Show Fairness Chart -----------------------------
def show_fairness_chart():
    metrics = {
        "Statistical Parity": 0.18,
        "Equal Opportunity": -0.12,
        "Disparate Impact": 0.75
    }
    path = plot_fairness_metrics(metrics)
    st.image(path, caption="📊 Fairness Metrics")

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="LLM AutoML", layout="wide")
st.title("🤖 LLM AutoML Platform")

tabs = st.tabs(["📁 Upload", "📈 Fairness", "📤 Email EDA", "🧠 Fallback", "📋 Preview"])

# ==========================
# Tab 0: Upload + Train + EDA
# ==========================
with tabs[0]:
    uploaded_file = st.file_uploader("Upload your dataset (CSV, XLSX, JSON, Parquet)", type=["csv", "xlsx", "json", "parquet"])
    df_preview = None

    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Auto-load preview based on type
        try:
            if uploaded_file.name.endswith(".csv"):
                df_preview = pd.read_csv(file_path)
            elif uploaded_file.name.endswith(".xlsx"):
                df_preview = pd.read_excel(file_path)
            elif uploaded_file.name.endswith(".json"):
                df_preview = pd.read_json(file_path)
            elif uploaded_file.name.endswith(".parquet"):
                df_preview = pd.read_parquet(file_path)
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully.")
            st.dataframe(df_preview.head(100))
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

        if st.button("🚀 Train Model + Generate EDA"):
            with st.spinner("Training and analyzing..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/train-model/",
                        params={"file_name": uploaded_file.name},
                        timeout=60
                    )
                    if response.status_code == 200:
                        st.success("✅ Training & EDA complete.")
                        st.json(response.json())
                        show_eda_images()
                    else:
                        st.error(f"❌ Backend error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.warning("⚠️ Backend not available. Trying fallback...")
                    if deepseek_fallback:
                        with open(file_path, "r") as f:
                            csv_head = f.read(3000)
                        try:
                            llm_response = deepseek_fallback(f"Analyze this:\n{csv_head}")
                            st.success("✅ LLM Fallback:")
                            st.markdown(llm_response)
                        except Exception as e:
                            st.error(f"❌ LLM Fallback failed: {e}")
                    else:
                        st.error("⚠️ Fallback unavailable.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

# ==========================
# Tab 1: Fairness
# ==========================
with tabs[1]:
    st.header("📊 Fairness Visualizer")
    show_fairness_chart()

# ==========================
# Tab 2: Email EDA PDF
# ==========================
with tabs[2]:
    st.header("📤 Send EDA Report via Email")
    email = st.text_input("Enter recipient email:")
    if st.button("📨 Send EDA PDF"):
        try:
            result = send_eda_email(email)
            if result:
                st.success(f"✅ EDA report sent to {email}")
            else:
                st.error("❌ Email failed.")
        except Exception as e:
            st.error(f"❌ Failed to send email: {e}")

# ==========================
# Tab 3: LLM Assistant
# ==========================
with tabs[3]:
    st.header("🧠 Ask LLM (DeepSeek Fallback)")
    prompt = st.text_area("Ask something about your ML task:")
    if st.button("🧠 Submit to DeepSeek"):
        if deepseek_fallback:
            st.info("⌛ Waiting for DeepSeek...")
            try:
                reply = deepseek_fallback(prompt)
                st.success("✅ Response:")
                st.markdown(reply)
            except Exception as e:
                st.error(f"❌ Failed to get DeepSeek response: {e}")
        else:
            st.warning("⚠️ DeepSeek not configured.")

# ==========================
# Tab 4: Data Preview
# ==========================
with tabs[4]:
    st.header("📋 Data Preview & Auto Insights")
    show_data_preview()
