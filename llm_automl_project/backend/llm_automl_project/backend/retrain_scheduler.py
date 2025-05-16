from apscheduler.schedulers.background import BackgroundScheduler
from backend.background_tasks import auto_retrain_task

def start_retrain_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(auto_retrain_task, 'interval', hours=24)
    scheduler.start()
