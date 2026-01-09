# -*- coding: utf-8 -*-
"""
SHAP 解释模块

使用 SHAP TreeExplainer 解释模型预测。
"""

import logging
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class ShapExplainer:
    """SHAP 解释器封装"""
    
    def __init__(self, model, feature_names: List[str] = None):
        """
        初始化 SHAP 解释器
        
        Args:
            model: CatBoost 模型实例
            feature_names: 特征名列表
        """
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
    
    def _init_explainer(self):
        """延迟初始化 TreeExplainer"""
        if self._explainer is None:
            logger.info("初始化 SHAP TreeExplainer...")
            self._explainer = shap.TreeExplainer(self.model)
    
    def explain_prediction(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        解释单个或多个样本的预测
        
        Args:
            X: 特征矩阵 (可以是单个样本或多个样本)
            top_k: 返回前 k 个重要特征
            
        Returns:
            List[Dict]: 每个样本的解释结果列表
        """
        self._init_explainer()
        
        # 确保是 2D 数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.asarray(X)
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X_array.shape[1])]
        
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        
        # 计算 SHAP 值
        shap_values = self._explainer.shap_values(X_array)
        
        # 处理二分类情况（取正类的 SHAP 值）
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        results = []
        for i in range(X_array.shape[0]):
            sample_shap = shap_values[i]
            sample_features = X_array[i]
            
            # 按绝对值排序
            sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:top_k]
            
            explanation = {
                "base_value": float(self._explainer.expected_value[1] 
                                   if isinstance(self._explainer.expected_value, list) 
                                   else self._explainer.expected_value),
                "features": []
            }
            
            for idx in sorted_idx:
                explanation["features"].append({
                    "name": feature_names[idx],
                    "value": float(sample_features[idx]),
                    "shap_value": float(sample_shap[idx]),
                    "contribution": "positive" if sample_shap[idx] > 0 else "negative"
                })
            
            results.append(explanation)
        
        return results
    
    def get_feature_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        计算全局特征重要性（基于 SHAP 值的平均绝对值）
        
        Args:
            X: 特征矩阵
            
        Returns:
            pd.DataFrame: 特征重要性排名
        """
        self._init_explainer()
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.asarray(X)
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X_array.shape[1])]
        
        # 计算 SHAP 值
        shap_values = self._explainer.shap_values(X_array)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # 计算平均绝对 SHAP 值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap
        })
        
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("SHAP 解释模块")
    print("使用方法:")
    print("  from ml_core.explainability.shap_explainer import ShapExplainer")
    print("  explainer = ShapExplainer(model, feature_names)")
    print("  result = explainer.explain_prediction(X_sample)")
