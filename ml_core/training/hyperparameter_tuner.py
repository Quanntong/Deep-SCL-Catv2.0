# -*- coding: utf-8 -*-
"""
超参数调优模块

使用 Optuna 进行 CatBoost 超参数优化。
"""

import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


def optimize_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    使用 Optuna 优化 CatBoost 超参数
    
    Args:
        X: 特征矩阵
        y: 目标变量
        n_trials: Optuna 试验次数
        n_splits: 交叉验证折数
        random_state: 随机种子
        
    Returns:
        Dict[str, Any]: 最优参数字典
    """
    logger.info(f"开始超参数优化，试验次数: {n_trials}")
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna 目标函数"""
        # 定义搜索空间
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.5, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": random_state,
            "verbose": False,
            "auto_class_weights": "Balanced"
        }
        
        # 使用分层 K 折交叉验证
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        model = CatBoostClassifier(**params)
        
        # 计算 F1 分数
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        return scores.mean()
    
    # 创建 Optuna 研究
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 获取最优参数
    best_params = study.best_params
    best_params["random_seed"] = random_state
    best_params["verbose"] = False
    best_params["auto_class_weights"] = "Balanced"
    
    logger.info(f"最优 F1 分数: {study.best_value:.4f}")
    logger.info(f"最优参数: {best_params}")
    
    return best_params


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("超参数调优模块")
    print("使用方法:")
    print("  from ml_core.training.hyperparameter_tuner import optimize_catboost")
    print("  best_params = optimize_catboost(X, y, n_trials=50)")
