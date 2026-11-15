from fastapi import APIRouter
from . import chat, search

router = APIRouter(prefix="/api")

router.include_router(chat.router)
router.include_router(search.router)