from fastapi import BackgroundTasks
import time
from backend.database import get_feedback_stats
from backend.retrain import retrain_from_feedback

def auto_retrain_task():
    stats = get_feedback_stats()
    if stats["should_retrain"]:
        retrain_from_feedback()

def schedule_daily_monitoring(tasks: BackgroundTasks):
    tasks.add_task(auto_retrain_task)
