# -*- coding: utf-8 -*-
"""推理预测器 - 封装模型加载和预测"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any

class RiskPredictor:
    """风险预测器，封装模型加载和预测接口"""
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """预测并返回完整结果"""
        proba = self.pipeline.predict_proba(X)[:, 1]
        labels = self.pipeline.predict(X)
        
        # 获取聚类标签
        X_scaled = self.pipeline.scaler.transform(X.values if isinstance(X, pd.DataFrame) else X)
        cluster_ids = self.pipeline.kmeans.predict(X_scaled)
        
        return pd.DataFrame({
            "is_risk": labels,
            "risk_probability": proba,
            "cluster_id": cluster_ids
        })
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """单条记录预测"""
        df = pd.DataFrame([features])
        result = self.predict(df)
        return {
            "is_risk": bool(result["is_risk"].iloc[0]),
            "risk_probability": float(result["risk_probability"].iloc[0]),
            "cluster_id": int(result["cluster_id"].iloc[0])
        }
