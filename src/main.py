from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.database import init_db
from src.routers import chat, corrections, contributions, admin

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Maithili AI", version="0.1.0", lifespan=lifespan)

app.include_router(chat.router)
app.include_router(corrections.router)
app.include_router(contributions.router)
app.include_router(admin.router)
