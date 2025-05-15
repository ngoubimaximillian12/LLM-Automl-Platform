import streamlit as st
import pandas as pd

def render_dashboard():
    st.subheader("📉 Model & Bias History Dashboard")

    history_data = {
        "Date": ["2025-05-01", "2025-05-08", "2025-05-15"],
        "Accuracy": [0.82, 0.78, 0.69],
        "Bias Score": [0.12, 0.15, 0.25],
        "Feedback Samples": [42, 60, 85],
    }

    df = pd.DataFrame(history_data)
    st.line_chart(df.set_index("Date"))
    st.dataframe(df)
