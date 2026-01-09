# -*- coding: utf-8 -*-
"""
CatBoost 分类器封装

封装 CatBoostClassifier，提供统一接口。
用于学生挂科风险的二分类预测。
"""

import logging
from typing import Union, Optional, Dict, Any, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .base_model import SupervisedModel

# 配置日志
logger = logging.getLogger(__name__)


class CatBoostWrapper(SupervisedModel):
    """
    CatBoost 分类器封装
    
    封装 CatBoostClassifier，用于学生挂科风险预测。
    支持概率预测和特征重要性获取。
    """
    
    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.1,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 0.5,
        border_count: int = 128,
        random_state: int = 42,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
        name: str = "CatBoostClassifier",
        **kwargs
    ):
        """
        初始化 CatBoost 分类器
        
        Args:
            iterations: 迭代次数（树的数量）
            learning_rate: 学习率
            depth: 树的最大深度
            l2_leaf_reg: L2 正则化系数
            random_strength: 随机强度（用于防止过拟合）
            bagging_temperature: Bagging 温度
            border_count: 数值特征分箱数
            random_state: 随机种子
            early_stopping_rounds: 早停轮数
            verbose: 是否输出训练日志
            name: 模型名称
            **kwargs: 其他 CatBoost 参数
        """
        super().__init__(name=name)
        
        # 保存超参数
        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "border_count": border_count,
            "random_seed": random_state,
            "early_stopping_rounds": early_stopping_rounds,
            "verbose": verbose,
            "loss_function": "Logloss",  # 二分类
            "eval_metric": "AUC",  # 评估指标
            "auto_class_weights": "Balanced",  # 自动处理类别不平衡
            **kwargs
        }
        
        # 内部 CatBoost 实例
        self._model: Optional[CatBoostClassifier] = None
        
        # 特征名
        self.feature_names_: Optional[List[str]] = None
        
        # 训练历史
        self.best_iteration_: Optional[int] = None
        self.best_score_: Optional[float] = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: tuple = None,
        cat_features: List[int] = None
    ) -> "CatBoostWrapper":
        """
        拟合分类器
        
        Args:
            X: 特征矩阵
            y: 目标变量
            eval_set: 验证集 (X_val, y_val)，用于早停
            cat_features: 类别特征的索引列表
            
        Returns:
            self
        """
        # 保存特征名
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = np.asarray(X)
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.asarray(y)
        
        logger.info(f"[{self.name}] 开始训练")
        logger.info(f"[{self.name}] 训练集形状: X={X_array.shape}, y={y_array.shape}")
        logger.info(f"[{self.name}] 目标分布: 0={np.sum(y_array==0)}, 1={np.sum(y_array==1)}")
        
        # 创建 CatBoost 模型
        self._model = CatBoostClassifier(**self.params)
        
        # 准备验证集
        eval_pool = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            eval_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        # 训练
        train_pool = Pool(X_array, y_array, cat_features=cat_features)
        self._model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=True if eval_set else False
        )
        
        # 记录最佳迭代信息
        if eval_set is not None:
            self.best_iteration_ = self._model.get_best_iteration()
            self.best_score_ = self._model.get_best_score()
            logger.info(f"[{self.name}] 最佳迭代: {self.best_iteration_}")
            logger.info(f"[{self.name}] 最佳分数: {self.best_score_}")
        
        self._is_fitted = True
        logger.info(f"[{self.name}] 训练完成")
        
        return self
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        预测类别标签
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测标签 (0 或 1)
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        return self._model.predict(X_array).flatten().astype(int)
    
    def predict_proba(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 概率矩阵，形状为 (n_samples, 2)
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        return self._model.predict_proba(X_array)
    
    def get_feature_importance(
        self, 
        importance_type: str = "FeatureImportance"
    ) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型
                - "FeatureImportance": 基于预测值变化
                - "ShapValues": SHAP 值
                - "PredictionValuesChange": 预测值变化
            
        Returns:
            pd.DataFrame: 特征重要性数据框
        """
        self._check_is_fitted()
        
        importance = self._model.get_feature_importance(type=importance_type)
        
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        })
        
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    
    def update_params(self, **kwargs) -> None:
        """
        更新模型参数（需要重新训练）
        
        Args:
            **kwargs: 要更新的参数
        """
        self.params.update(kwargs)
        self._is_fitted = False
        self._model = None
        logger.info(f"[{self.name}] 参数已更新，需要重新训练")
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取当前参数
        
        Returns:
            Dict[str, Any]: 参数字典
        """
        return self.params.copy()
    
    @property
    def catboost_model(self) -> CatBoostClassifier:
        """获取内部 CatBoost 模型实例"""
        self._check_is_fitted()
        return self._model


# ==================== 使用示例 ====================
if __name__ == "__main__":
    print("CatBoost 分类器模块加载成功")
    print("使用方法:")
    print("  from ml_core.models import CatBoostWrapper")
    print("  classifier = CatBoostWrapper(iterations=500, learning_rate=0.1)")
    print("  classifier.fit(X_train, y_train, eval_set=(X_val, y_val))")
    print("  proba = classifier.predict_proba(X_test)")