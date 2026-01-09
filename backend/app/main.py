# -*- coding: utf-8 -*-
"""FastAPI 主入口"""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.model_manager import ModelManager
from app.api.endpoints import router as api_router

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型"""
    if not ARTIFACTS_DIR.exists():
        raise RuntimeError(f"artifacts 目录不存在: {ARTIFACTS_DIR}")
    app.state.model = ModelManager(str(ARTIFACTS_DIR))
    print(f"✓ ModelManager 加载成功")
    yield
    app.state.model = None

app = FastAPI(title="心理风险预测系统", version="2.0.0", lifespan=lifespan)

# CORS - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "心理风险预测系统 API v2.0", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": hasattr(app.state, "model") and app.state.model is not None}

