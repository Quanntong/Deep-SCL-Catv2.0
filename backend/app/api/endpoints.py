# -*- coding: utf-8 -*-
"""API 端点"""

import os
import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

# 数据目录
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"

# ========== Schemas ==========
class EPQFactors(BaseModel):
    E: float = 50
    P: float = 50
    N: float = 50
    L: float = 50

class SCL90Factors(BaseModel):
    somatization: float = 1.5
    obsessive_compulsive: float = 1.5
    interpersonal_sensitivity: float = 1.5
    depression: float = 1.5
    anxiety: float = 1.5
    hostility: float = 1.5
    phobic_anxiety: float = 1.5
    paranoid_ideation: float = 1.5
    psychoticism: float = 1.5
    other: float = 1.5

class OnlineRequest(BaseModel):
    answers: List[int]  # 90个答案 (1-5)
    epq: EPQFactors

class ManualRequest(BaseModel):
    scl90: SCL90Factors
    epq: EPQFactors

class PredictionResponse(BaseModel):
    is_risk: bool
    risk_probability: float
    cluster_id: int
    threshold: float
    shap_values: List[Dict[str, Any]] = []

# ========== Endpoints ==========

@router.post("/predict/online", response_model=PredictionResponse)
async def predict_online(request: Request, data: OnlineRequest):
    """在线测评预测：90道题答案 + EPQ"""
    model = request.app.state.model
    try:
        result = model.predict_online(data.answers, data.epq.model_dump())
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.post("/predict/manual", response_model=PredictionResponse)
async def predict_manual(request: Request, data: ManualRequest):
    """手动输入预测：因子分"""
    model = request.app.state.model
    result = model.predict_manual(data.scl90.model_dump(), data.epq.model_dump())
    return result

@router.post("/predict/batch")
async def predict_batch(request: Request, file: UploadFile = File(...), mode: str = Form("balanced")):
    """批量文件预测（mode: strict/balanced/sensitive）"""
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(400, "仅支持 CSV 或 Excel 文件")
    
    content = await file.read()
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_excel(BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"文件读取失败: {e}")
    
    model = request.app.state.model
    results = model.predict_batch(df, mode=mode)
    
    return {
        "results": results,
        "total": len(results),
        "risk_count": sum(1 for r in results if r["is_risk"])
    }

@router.post("/predict/batch/export")
async def export_batch(request: Request, file: UploadFile = File(...), mode: str = Form("balanced")):
    """批量预测并导出 Excel（mode: strict/balanced/sensitive）"""
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(400, "仅支持 CSV 或 Excel 文件")
    
    content = await file.read()
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_excel(BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"文件读取失败: {e}")
    
    # 保留原始数据
    original_df = df.copy()
    
    model = request.app.state.model
    results = model.predict_batch(df, mode=mode)
    
    # 添加预测结果列到原始数据
    original_df['风险状态'] = [('存在风险' if r['is_risk'] else '正常') for r in results]
    original_df['风险概率'] = [f"{r['risk_probability']*100:.1f}%" for r in results]
    original_df['预测挂科数'] = [r.get('predicted_failed_count', 0) for r in results]
    original_df['聚类分组'] = [r['cluster_id'] for r in results]
    
    # 导出为 Excel
    output = BytesIO()
    original_df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': 'attachment; filename=prediction_results.xlsx'}
    )

@router.get("/dashboard/stats")
async def get_dashboard_stats():
    """获取仪表盘真实统计数据"""
    total, high_risk, update_time = 0, 0, "暂无数据"
    
    # 读取所有训练数据文件
    if DATA_DIR.exists():
        for f in DATA_DIR.glob("*.xlsx"):
            try:
                df = pd.read_excel(f)
                total += len(df)
                # 挂科数目 > 0 视为高风险
                if '挂科数目' in df.columns:
                    high_risk += int((df['挂科数目'] > 0).sum())
                # 获取最新文件修改时间
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                update_time = mtime.strftime("%Y-%m-%d %H:%M")
            except:
                pass
    
    ratio = round(high_risk / total * 100, 1) if total > 0 else 0
    return {"total": total, "high_risk": high_risk, "ratio": ratio, "update_time": update_time}

@router.post("/scl90/calculate")
async def calculate_scl90(answers: List[int]):
    """计算 SCL-90 因子分"""
    from backend.app.core.model_manager import ModelManager
    if len(answers) != 90:
        raise HTTPException(400, f"需要90个答案，收到{len(answers)}个")
    
    # 使用静态方法计算
    factors = {}
    from backend.app.core.model_manager import SCL90_FACTOR_ITEMS
    for factor, items in SCL90_FACTOR_ITEMS.items():
        scores = [answers[i-1] for i in items]
        factors[factor] = sum(scores) / len(scores)
    return factors
