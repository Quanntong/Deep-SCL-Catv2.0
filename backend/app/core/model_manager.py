# -*- coding: utf-8 -*-
"""ModelManager - 核心模型管理器，处理加载、预处理、预测和SHAP解释"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, List, Any

# SCL-90 题号到因子的映射
SCL90_FACTOR_ITEMS = {
    "somatization": [1,4,12,27,40,42,48,49,52,53,56,58],
    "obsessive_compulsive": [3,9,10,28,38,45,46,51,55,65],
    "interpersonal_sensitivity": [6,21,34,36,37,41,61,69,73],
    "depression": [5,14,15,20,22,26,29,30,31,32,54,71,79],
    "anxiety": [2,17,23,33,39,57,72,78,80,86],
    "hostility": [11,24,63,67,74,81],
    "phobic_anxiety": [13,25,47,50,70,75,82],
    "paranoid_ideation": [8,18,43,68,76,83],
    "psychoticism": [7,16,35,62,77,84,85,87,88,90],
    "other": [19,44,59,60,64,66,89]
}

SCL90_FACTORS = ["somatization", "obsessive_compulsive", "interpersonal_sensitivity", 
                 "depression", "anxiety", "hostility", "phobic_anxiety", 
                 "paranoid_ideation", "psychoticism", "other"]
EPQ_FACTORS = ["E", "P", "N", "L"]
FEATURE_COLS = SCL90_FACTORS + EPQ_FACTORS

COLUMN_MAP = {
    "躯体化": "somatization",
    "强迫症状": "obsessive_compulsive", 
    "人际关系敏感": "interpersonal_sensitivity",
    "抑郁": "depression",
    "焦虑": "anxiety",
    "敌对": "hostility",
    "恐怖": "phobic_anxiety",
    "偏执": "paranoid_ideation",
    "精神病性": "psychoticism",
    "其他": "other",
    "内外向E": "E",
    "神经质N": "N",
    "精神质P": "P",
    "掩饰性L": "L"
}


class ModelManager:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self._load_models()
        self._init_shap()
    
    def _load_models(self):
        models_dir = self.artifacts_dir / "models"
        
        self.scaler = joblib.load(models_dir / "standard_scaler.joblib")
        self.kmeans = joblib.load(models_dir / "kmeans_clusterer.joblib")
        
        # 加载线性模型（新）或 CatBoost（旧）
        clf_path = models_dir / "logistic_classifier.pkl"
        if clf_path.exists():
            self.classifier = joblib.load(clf_path)
            self.full_scaler = joblib.load(models_dir / "full_scaler.joblib")
            self.model_type = "linear"
        else:
            self.classifier = joblib.load(models_dir / "catboost_classifier.joblib")
            self.full_scaler = None
            self.model_type = "catboost"
        
        # 回归器
        reg_path = models_dir / "ridge_regressor.pkl"
        if reg_path.exists():
            self.regressor = joblib.load(reg_path)
        else:
            reg_path = models_dir / "catboost_regressor.joblib"
            self.regressor = joblib.load(reg_path) if reg_path.exists() else None
        
        # 加载配置（包含训练时计算的最优阈值）
        config_path = models_dir / "model_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.thresholds = config.get("thresholds", {
                "strict": 0.60, "balanced": 0.50, "sensitive": 0.35
            })
            self.feature_cols = config.get("feature_cols", FEATURE_COLS + ["cluster"])
        else:
            self.thresholds = {"strict": 0.60, "balanced": 0.50, "sensitive": 0.35}
            self.feature_cols = FEATURE_COLS + ["cluster"]
        
        self.threshold = self.thresholds.get("balanced", 0.50)
    
    def _init_shap(self):
        try:
            if self.model_type == "linear":
                self.explainer = shap.LinearExplainer(self.classifier, np.zeros((1, len(self.feature_cols))))
            else:
                self.explainer = shap.TreeExplainer(self.classifier)
        except:
            self.explainer = None
    
    def calculate_scl90_factors(self, answers: List[int]) -> Dict[str, float]:
        if len(answers) != 90:
            raise ValueError(f"需要90个答案，收到{len(answers)}个")
        factors = {}
        for factor, items in SCL90_FACTOR_ITEMS.items():
            scores = [answers[i-1] for i in items]
            factors[factor] = sum(scores) / len(scores)
        return factors
    
    def preprocess(self, features: Dict[str, float]) -> pd.DataFrame:
        df = pd.DataFrame([features])
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 2.0 if col in SCL90_FACTORS else 50.0
        df = df[FEATURE_COLS].fillna(df.median())
        
        # 添加特征工程（与训练时一致）
        df["scl90_total"] = df[SCL90_FACTORS].mean(axis=1)
        df["high_risk_score"] = (df["depression"] + df["anxiety"] + df["interpersonal_sensitivity"]) / 3
        
        return df
    
    def predict(self, features: Dict[str, float], with_shap: bool = True) -> Dict[str, Any]:
        df = self.preprocess(features)
        
        # SCL-90 标准化 + KMeans
        X_scl90_scaled = self.scaler.transform(df[SCL90_FACTORS])
        cluster_id = int(self.kmeans.predict(X_scl90_scaled)[0])
        
        # 组合特征
        df_with_cluster = df.copy()
        df_with_cluster["cluster"] = cluster_id
        
        # 线性模型需要全特征标准化
        if self.model_type == "linear" and self.full_scaler:
            X_input = self.full_scaler.transform(df_with_cluster)
        else:
            X_input = df_with_cluster
        
        # 分类预测
        proba = self.classifier.predict_proba(X_input)[0, 1]
        
        # 回归预测
        pred_count = 0.0
        if self.regressor:
            pred_count = max(0, round(float(self.regressor.predict(X_input)[0]), 1))
        
        # 使用 balanced 阈值判定
        is_risk = (proba >= self.threshold) or (pred_count >= 0.7)
        
        result = {
            "is_risk": bool(is_risk),
            "risk_probability": float(proba),
            "cluster_id": cluster_id,
            "threshold": float(self.threshold),
            "predicted_failed_count": pred_count
        }
        
        # SHAP
        if with_shap and self.explainer:
            try:
                shap_values = self.explainer.shap_values(X_input)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                result["shap_values"] = [
                    {"feature": name, "value": float(val)}
                    for name, val in zip(self.feature_cols, shap_values[0] if len(shap_values.shape) > 1 else shap_values)
                ]
            except:
                result["shap_values"] = []
        
        return result
    
    def predict_online(self, answers: List[int], epq: Dict[str, float]) -> Dict[str, Any]:
        scl90_factors = self.calculate_scl90_factors(answers)
        return self.predict({**scl90_factors, **epq})
    
    def predict_manual(self, scl90: Dict[str, float], epq: Dict[str, float]) -> Dict[str, Any]:
        return self.predict({**scl90, **epq})
    
    def predict_batch(self, df: pd.DataFrame, mode: str = "balanced") -> List[Dict[str, Any]]:
        # 优化后的阈值配置（基于实际数据分布调优）
        mode_config = {
            "strict": {"prob": 0.58, "count": 0.9},      # 高精确率：prob>=0.58 OR count>=0.9
            "balanced": {"prob": 0.52, "count": 0.7},   # 最优F1：prob>=0.52 OR count>=0.7
            "sensitive": {"prob": 0.46, "count": 0.5}   # 高召回率：prob>=0.46 OR count>=0.5
        }
        config = mode_config.get(mode, mode_config["balanced"])
        
        df = df.rename(columns=COLUMN_MAP)
        results = []
        for _, row in df.iterrows():
            features = {col: row.get(col) for col in FEATURE_COLS}
            r = self.predict(features, with_shap=False)
            
            # 根据模式判定风险：概率 OR 预测挂科数
            r["is_risk"] = (r["risk_probability"] >= config["prob"]) or (r["predicted_failed_count"] >= config["count"])
            r["mode_threshold"] = config["prob"]
            results.append(r)
        return results
