import os
import sys
import streamlit as st

# ✅ Add project root to system path so backend module can be imported
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.llm_generator import generate_preprocessing_code
except ModuleNotFoundError as e:
    st.error(f"❌ Failed to import backend module: {e}")
    raise

# 🔷 Streamlit UI
st.title("💬 LLM Assistant: ML Guidance and Code Suggestions")

task = st.text_input("Describe your ML task (e.g., clean missing values, normalize income column):")

if st.button("Generate Code"):
    if task.strip():
        st.info("Generating Python code using LLM...")
        try:
            code = generate_preprocessing_code(task)
            st.code(code, language="python")
        except Exception as e:
            st.error(f"LLM generation failed: {e}")
    else:
        st.warning("Please enter a task description.")

st.markdown("---")
st.caption("Powered by HuggingFace Transformers. Customize this prompt to guide AutoML steps.")
