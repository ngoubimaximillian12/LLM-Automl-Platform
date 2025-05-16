import streamlit as st
import pandas as pd
import os

def show_model_dashboard():
    st.subheader("📈 Model History & Bias Overview")

    metadata_file = "data/model_metadata.csv"
    feedback_file = "data/feedback_log.csv"

    if os.path.exists(metadata_file):
        df_models = pd.read_csv(metadata_file)
        st.markdown("### ✅ Trained Models")
        st.dataframe(df_models)
    else:
        st.warning("No model metadata available yet.")

    if os.path.exists(feedback_file):
        df_feedback = pd.read_csv(feedback_file)
        st.markdown("### 📝 User Feedback")
        st.dataframe(df_feedback)
    else:
        st.info("No feedback data recorded yet.")
