import streamlit as st
from eda_email import email_eda_ui

def show_email_tab():
    st.subheader("📧 Email EDA Report")
    email_eda_ui()
