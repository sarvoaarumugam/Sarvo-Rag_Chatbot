from fastapi import APIRouter
from src.endpoints.rag_chatbot import router as rag_router

router = APIRouter()

router.include_router(rag_router)


