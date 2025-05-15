import os
import sys
import streamlit as st

# ✅ Add backend path for imports
BACKEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)

# ✅ Try to import backend helpers
try:
    from llm_bias_helper import bias_explainer, inject_code_into_pipeline
except Exception as e:
    st.error(f"❌ Failed to import LLM bias tools: {e}")
    st.stop()

# ✅ Streamlit Title
st.set_page_config(page_title="Bias Insights", layout="centered")
st.title("🧠 LLM-Powered Bias Explanation & Auto-Injection")

# ✅ Simulated audit metrics for demo/testing
simulated_audit = {
    "Statistical Parity Difference": -0.23,
    "Equal Opportunity Difference": 0.14,
    "Disparate Impact": 0.65
}

# ------------------------------ Bias Explanation ------------------------------
st.subheader("1️⃣ Explain Audit Metrics using LLM")
if st.button("🧠 Explain Audit Results"):
    with st.spinner("🧠 Thinking..."):
        try:
            explanation = bias_explainer(simulated_audit)
            st.markdown("### 🔍 LLM Bias Explanation")
            st.write(explanation)
        except Exception as e:
            st.error(f"❌ Could not generate explanation: {e}")

# ------------------------------ Pipeline Code Injection ------------------------------
st.markdown("---")
st.subheader("2️⃣ Inject LLM-Generated Code into Pipeline")

task_desc = st.text_input("🔧 Describe your data preprocessing task (e.g., encode gender column):")

if st.button("⚙️ Generate and Inject Code"):
    if task_desc.strip():
        with st.spinner("🧠 Generating and injecting code..."):
            try:
                file_path = inject_code_into_pipeline(task_desc)
                if os.path.exists(file_path):
                    st.success(f"✅ Code saved to `{file_path}`")
                    with open(file_path, "r") as f:
                        st.code(f.read(), language="python")
                else:
                    st.error("⚠️ Code generation failed or file not created.")
            except Exception as e:
                st.error(f"❌ Injection failed: {e}")
    else:
        st.warning("⚠️ Please enter a task description.")
