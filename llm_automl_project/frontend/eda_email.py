import os
import smtplib
import streamlit as st
from email.message import EmailMessage
from dotenv import load_dotenv

# 🔄 Load environment variables
load_dotenv()

# ✅ Email credentials from .env
EMAIL_ADDRESS = os.getenv("SMTP_USER")
EMAIL_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))

# 📄 Path to EDA report PDF
EDA_PDF_PATH = "data/eda_report/eda_report.pdf"


def send_eda_email(recipient: str) -> bool:
    """Send the EDA PDF report to the given recipient email."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        st.error("❌ Email credentials not set. Please check your `.env` file.")
        return False

    if not os.path.exists(EDA_PDF_PATH):
        st.error("❌ EDA report not found. Please train the model first.")
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = "📊 Your EDA Report - LLM AutoML"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = recipient
        msg.set_content(
            "Hello,\n\nAttached is your EDA report generated using the LLM AutoML platform.\n\nBest regards,\nLLM AutoML Team"
        )

        with open(EDA_PDF_PATH, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename="eda_report.pdf")

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        return True

    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        return False


def email_eda_ui():
    """Streamlit UI to collect email and send EDA PDF."""
    with st.expander("📤 Send EDA Report via Email"):
        recipient = st.text_input("Recipient Email")
        if st.button("📨 Send Report"):
            if recipient:
                with st.spinner("Sending report..."):
                    success = send_eda_email(recipient)
                    if success:
                        st.success("✅ Report sent successfully.")
            else:
                st.warning("⚠️ Please enter a valid email address.")
