from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./automl.db"  # Use PostgreSQL for production

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

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

def init_db():
    Base.metadata.create_all(bind=engine)

def save_model_metadata(name, accuracy, path):
    session = SessionLocal()
    metadata = ModelMetadata(name=name, accuracy=accuracy, filepath=path)
    session.add(metadata)
    session.commit()
    session.close()

def log_feedback(input_data: str, prediction: str, user_correction: str = None):
    session = SessionLocal()
    entry = Feedback(input_data=input_data, prediction=prediction, user_correction=user_correction)
    session.add(entry)
    session.commit()
    session.close()
