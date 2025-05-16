from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


# ------------------ Configuration ------------------
DATABASE_URL = "sqlite:///./automl.db"  # ✅ Use PostgreSQL for production

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ------------------ Database Models ------------------
class ModelMetadata(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    accuracy = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    filepath = Column(Text)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(Text)
    prediction = Column(String)
    user_correction = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

# ------------------ Initialization ------------------
def init_db():
    """Creates tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

# ------------------ Save Model Info ------------------
def save_model_metadata(name: str, accuracy: float, path: str):
    """Stores model metadata into DB."""
    session = SessionLocal()
    try:
        metadata = ModelMetadata(name=name, accuracy=accuracy, filepath=path)
        session.add(metadata)
        session.commit()
    finally:
        session.close()

# ------------------ Feedback Logging ------------------
def log_feedback(input_data: str, prediction: str, user_correction: str = None):
    """Logs prediction and optional correction from user."""
    session = SessionLocal()
    try:
        entry = Feedback(
            input_data=input_data,
            prediction=prediction,
            user_correction=user_correction
        )
        session.add(entry)
        session.commit()
    finally:
        session.close()

# ------------------ Stats for Retraining ------------------
def get_feedback_stats(threshold: int = 5) -> dict:
    """
    Determines whether enough corrected feedback is present for retraining.

    Returns:
        dict: {
            "should_retrain": bool,
            "feedback_count": int
        }
    """
    session: Session = SessionLocal()
    try:
        count = session.query(Feedback).filter(Feedback.user_correction.isnot(None)).count()
    finally:
        session.close()

    return {
        "should_retrain": count >= threshold,
        "feedback_count": count
    }
