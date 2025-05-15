import smtplib
from email.message import EmailMessage
import os

def send_eda_report(recipient_email: str, report_path: str):
    email_sender = os.getenv("EMAIL_SENDER")
    email_password = os.getenv("EMAIL_PASSWORD")

    msg = EmailMessage()
    msg["Subject"] = "📊 EDA Report"
    msg["From"] = email_sender
    msg["To"] = recipient_email

    msg.set_content("Attached is your requested EDA report.")
    with open(report_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename="EDA_Report.pdf")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(email_sender, email_password)
        smtp.send_message(msg)
