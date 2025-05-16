import os
import sys
from fastapi import BackgroundTasks

# ✅ Fix module path for direct execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ✅ Safe imports after fixing path
try:
    from backend.database import get_feedback_stats
    from backend.retrain import retrain_from_feedback
except ImportError as e:
    print(f"❌ Import error in background_tasks.py: {e}")
    raise

# 🔁 Auto-retraining logic
def auto_retrain_task():
    try:
        stats = get_feedback_stats()
        if stats.get("should_retrain"):
            print("🔁 Triggering auto-retraining from feedback...")
            retrain_from_feedback()
        else:
            print(f"⏳ Retraining skipped. Feedback count: {stats.get('feedback_count')}")
    except Exception as e:
        print(f"❌ Error in auto_retrain_task: {e}")

# 🕓 Background scheduling
def schedule_daily_monitoring(tasks: BackgroundTasks):
    tasks.add_task(auto_retrain_task)
