# -*- coding: utf-8 -*-
"""ModelManager - 核心模型管理器，处理加载、预处理、预测和SHAP解释"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, List, Any, Optional

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

# 特征列顺序
SCL90_FACTORS = ["somatization", "obsessive_compulsive", "interpersonal_sensitivity", 
                 "depression", "anxiety", "hostility", "phobic_anxiety", 
                 "paranoid_ideation", "psychoticism", "other"]
EPQ_FACTORS = ["E", "P", "N", "L"]
FEATURE_COLS = SCL90_FACTORS + EPQ_FACTORS

# 中文列名 -> 英文键名映射（用于批量预测）
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
    """模型管理器 - 加载模型、预处理、预测、SHAP解释"""
    
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self._load_models()
        self._init_shap()
    
    def _load_models(self):
        """加载所有模型组件"""
        models_dir = self.artifacts_dir / "models"
        
        # 加载各组件
        self.scaler = joblib.load(models_dir / "standard_scaler.joblib")
        self.kmeans = joblib.load(models_dir / "kmeans_clusterer.joblib")
        self.classifier = joblib.load(models_dir / "catboost_classifier.joblib")
        
        # 加载回归器（可选）
        reg_path = models_dir / "catboost_regressor.joblib"
        self.regressor = joblib.load(reg_path) if reg_path.exists() else None
        
        # 加载阈值（优先从 JSON 配置读取）
        config_path = models_dir / "model_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                self.threshold = config.get("threshold", 0.5)
        else:
            threshold_path = models_dir / "optimal_threshold.joblib"
            self.threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5
    
    def _init_shap(self):
        """初始化 SHAP 解释器"""
        try:
            model = self.classifier.model if hasattr(self.classifier, 'model') else self.classifier
            self.explainer = shap.TreeExplainer(model)
        except:
            self.explainer = None
    
    def calculate_scl90_factors(self, answers: List[int]) -> Dict[str, float]:
        """从90道题答案计算10个因子分"""
        if len(answers) != 90:
            raise ValueError(f"需要90个答案，收到{len(answers)}个")
        
        factors = {}
        for factor, items in SCL90_FACTOR_ITEMS.items():
            scores = [answers[i-1] for i in items]  # 题号从1开始
            factors[factor] = sum(scores) / len(scores)
        return factors
    
    def preprocess(self, features: Dict[str, float]) -> pd.DataFrame:
        """预处理特征，确保列顺序正确"""
        df = pd.DataFrame([features])
        
        # 确保所有列存在，缺失填充中位数
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 2.0 if col in SCL90_FACTORS else 50.0
        
        # 按正确顺序排列
        df = df[FEATURE_COLS]
        
        # 处理 NaN
        df = df.fillna(df.median())
        return df
    
    def predict(self, features: Dict[str, float], with_shap: bool = True) -> Dict[str, Any]:
        """预测风险并返回SHAP值"""
        df = self.preprocess(features)
        
        # 只对 SCL-90 因子标准化（scaler 只训练了这10个特征，保持 DataFrame 格式避免警告）
        X_scl90 = df[SCL90_FACTORS]
        X_scl90_scaled = pd.DataFrame(self.scaler.transform(X_scl90), columns=SCL90_FACTORS)
        
        # KMeans 聚类（只用 SCL-90 因子）
        cluster_id = int(self.kmeans.predict(X_scl90_scaled)[0])
        
        # 组合所有特征 + 聚类标签（用于 CatBoost，需要 DataFrame 格式）
        df_with_cluster = df.copy()
        df_with_cluster["cluster"] = cluster_id
        
        # 预测
        model = self.classifier.model if hasattr(self.classifier, 'model') else self.classifier
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df_with_cluster)[0, 1]
        else:
            proba = model.predict(df_with_cluster)[0]
        
        # 回归预测（预测挂科数）
        pred_count = 0.0
        if self.regressor:
            reg_model = self.regressor.model if hasattr(self.regressor, 'model') else self.regressor
            pred_count = float(reg_model.predict(df_with_cluster)[0])
            pred_count = max(0, round(pred_count, 1))  # 不能为负
        
        # 混合决策逻辑：预测挂科数 >= 0.5 或 概率 >= 阈值 - 0.05（更敏感，减少漏网之鱼）
        is_risk = (pred_count >= 0.5) or (proba >= self.threshold - 0.05)
        
        result = {
            "is_risk": bool(is_risk),
            "risk_probability": float(proba),
            "cluster_id": cluster_id,
            "threshold": float(self.threshold),
            "predicted_failed_count": pred_count
        }
        
        # SHAP 解释
        if with_shap and self.explainer:
            try:
                shap_values = self.explainer.shap_values(df_with_cluster)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # 取正类
                
                feature_names = FEATURE_COLS + ["cluster"]
                result["shap_values"] = [
                    {"feature": name, "value": float(val)}
                    for name, val in zip(feature_names, shap_values[0])
                ]
            except Exception as e:
                result["shap_values"] = []
        
        return result
    
    def predict_online(self, answers: List[int], epq: Dict[str, float]) -> Dict[str, Any]:
        """在线测评预测：90道题答案 + EPQ因子"""
        scl90_factors = self.calculate_scl90_factors(answers)
        features = {**scl90_factors, **epq}
        return self.predict(features)
    
    def predict_manual(self, scl90: Dict[str, float], epq: Dict[str, float]) -> Dict[str, Any]:
        """手动输入预测：直接提供因子分"""
        features = {**scl90, **epq}
        return self.predict(features)
    
    def predict_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """批量预测 - 支持中文列名，返回分类和回归结果"""
        # 应用中文->英文列名映射
        df = df.rename(columns=COLUMN_MAP)
        
        # 验证必要列
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"[警告] 缺少列: {missing}，将使用默认值填充")
        
        results = []
        for idx, row in df.iterrows():
            features = {col: row.get(col) for col in FEATURE_COLS}
            result = self.predict(features, with_shap=False)
            results.append(result)
        return results
