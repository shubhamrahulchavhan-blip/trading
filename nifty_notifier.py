"""Standalone Gmail test. Fill the 3 values below (or set them as environment variables), then run:
    python test_email.py
Use a Gmail APP PASSWORD (16 characters), not your normal Gmail password."""
import os, smtplib
from email.message import EmailMessage

EMAIL_FROM = os.getenv("EMAIL_FROM", "your_address@gmail.com")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "abcdefghijklmnop")   # no spaces
EMAIL_TO = os.getenv("EMAIL_TO", "your_address@gmail.com")

msg = EmailMessage()
msg["From"], msg["To"], msg["Subject"] = EMAIL_FROM, EMAIL_TO, "NIFTY notifier - test email"
msg.set_content("If you can read this, your email setup works.")

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(EMAIL_FROM, EMAIL_APP_PASSWORD.replace(" ", ""))
        s.send_message(msg)
    print("SUCCESS: email sent. Check your inbox (and Spam folder).")
except smtplib.SMTPAuthenticationError:
    print("FAILED: login rejected. Use an App password (needs 2-step verification), not your normal password.")
except Exception as e:
    print("FAILED:", e)
