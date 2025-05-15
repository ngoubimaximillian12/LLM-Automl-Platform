import os
import smtplib
import streamlit as st
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()


# Load secrets from .env or Streamlit secrets
EMAIL_ADDRESS = os.getenv("EMAIL_SENDER") or st.secrets.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") or st.secrets.get("EMAIL_PASSWORD")
EDA_PDF_PATH = "data/eda_report/eda_report.pdf"

def send_eda_email(recipient: str) -> bool:
    """Send the EDA PDF report to the given recipient email."""
    if not os.path.exists(EDA_PDF_PATH):
        st.error("❌ EDA PDF not found. Run training first.")
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = "📊 Your EDA Report from LLM AutoML"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = recipient
        msg.set_content("Hi,\n\nAttached is your EDA report. Happy modeling!\n\n- LLM AutoML Team")

        # Attach PDF
        with open(EDA_PDF_PATH, "rb") as f:
            pdf_data = f.read()
            msg.add_attachment(pdf_data, maintype="application", subtype="pdf", filename="eda_report.pdf")

        # Send email via SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        return True
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        return False

# Streamlit UI to send report
def email_eda_ui():
    st.subheader("📤 Send EDA Report via Email")
    recipient = st.text_input("Recipient Email Address")
    if st.button("📧 Send Report"):
        if recipient:
            with st.spinner("Sending EDA report..."):
                success = send_eda_email(recipient)
                if success:
                    st.success("✅ EDA report sent successfully.")
        else:
            st.warning("⚠️ Please enter a valid email address.")
