from celery import Celery
from src.config import settings

celery_app = Celery(
    "mtg_chatbot",
    broker=settings.rabbitmq_url,
    backend="rpc://", 
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

@celery_app.task
def dummy_task():
    return "Task completed"
