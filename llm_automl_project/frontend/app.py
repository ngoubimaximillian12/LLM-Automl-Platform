import os
import streamlit as st
import requests
from PIL import Image
from backend.llm_generator import deepseek_fallback  # <-- Ensure this exists

# Path to EDA chart directory
EDA_DIR = "data/eda_report"

# Backend URL (FastAPI)
BACKEND_URL = "http://127.0.0.1:8000"  # Replace if on a different device

# ----------------------------- Show EDA Charts -----------------------------
def show_eda_images():
    st.subheader("📊 Exploratory Data Analysis Report")
    if not os.path.exists(EDA_DIR):
        st.warning("⚠️ EDA report not found. Please train a model first.")
        return

    image_files = [f for f in os.listdir(EDA_DIR) if f.endswith(".png")]
    if not image_files:
        st.warning("⚠️ No EDA images available to display.")
    else:
        for img_file in image_files:
            image_path = os.path.join(EDA_DIR, img_file)
            st.image(image_path, caption=img_file, use_column_width=True)

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="LLM AutoML", layout="wide")
st.title("🤖 LLM AutoML Platform")

uploaded_file = st.file_uploader("📁 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully.")

    if st.button("🚀 Train Model + Generate EDA"):
        with st.spinner("🔧 Training model and generating EDA..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/train-model/",
                    params={"file_name": uploaded_file.name}
                )

                if response.status_code == 200:
                    st.success("✅ Model training and EDA generation complete.")
                    st.json(response.json())
                    show_eda_images()
                elif response.status_code == 404:
                    st.error("❌ Endpoint '/train-model/' not found on backend.")
                else:
                    st.error(f"❌ Backend error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.warning("⚠️ Backend offline. Using LLM fallback instead...")
                with open(file_path, "r") as f:
                    csv_content = f.read()

                try:
                    llm_response = deepseek_fallback(csv_content)
                    st.success("✅ LLM provided guidance:")
                    st.markdown(llm_response)
                except Exception as e:
                    st.error(f"❌ Fallback failed: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
