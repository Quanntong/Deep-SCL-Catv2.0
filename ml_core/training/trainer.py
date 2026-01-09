# -*- coding: utf-8 -*-
"""
训练器模块

实现完整的模型训练流程。
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from ..data.data_loader import load_cleaned_data
from ..models.hybrid_pipeline import RiskPipeline
from ..config.feature_config import FEATURE_CONFIG
from ..config.model_config import MODEL_CONFIG
from .hyperparameter_tuner import optimize_catboost

logger = logging.getLogger(__name__)


def train_model(
    data_path: str = None,
    output_dir: str = None,
    use_optuna: bool = False,
    n_trials: int = 50
) -> RiskPipeline:
    """
    训练风险预测模型
    
    Args:
        data_path: 数据目录路径
        output_dir: 模型输出目录
        use_optuna: 是否使用 Optuna 调参
        n_trials: Optuna 试验次数
        
    Returns:
        RiskPipeline: 训练好的流水线
    """
    # 设置默认路径
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "raw"
    if output_dir is None:
        output_dir = MODEL_CONFIG.ARTIFACTS_DIR
    
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    
    logger.info("=" * 50)
    logger.info("开始训练风险预测模型")
    logger.info("=" * 50)
    
    # 1. 加载数据
    logger.info("Step 1: 加载数据...")
    X, y = load_cleaned_data(data_dir=data_path)
    logger.info(f"特征矩阵形状: {X.shape}")
    logger.info(f"目标变量形状: {y.shape}")
    
    # 2. 超参数优化（可选）
    catboost_params = {}
    if use_optuna:
        logger.info("Step 2: 使用 Optuna 进行超参数优化...")
        catboost_params = optimize_catboost(X, y, n_trials=n_trials)
    else:
        logger.info("Step 2: 使用默认参数（跳过 Optuna 调参）")
    
    # 3. 初始化并训练流水线
    logger.info("Step 3: 训练 RiskPipeline...")
    pipeline = RiskPipeline(
        n_clusters=3,
        target_recall=0.95,
        catboost_params=catboost_params
    )
    pipeline.fit(X, y)
    
    # 4. 输出训练结果
    logger.info("Step 4: 评估模型性能...")
    y_pred = pipeline.predict(X)
    
    print("\n" + "=" * 50)
    print("分类报告 (训练集)")
    print("=" * 50)
    print(classification_report(y, y_pred, target_names=["低风险", "高风险"]))
    
    # 5. 保存模型
    logger.info("Step 5: 保存模型...")
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save_all(output_dir)
    
    logger.info("=" * 50)
    logger.info("训练完成!")
    logger.info(f"模型已保存至: {output_dir}")
    logger.info(f"最优阈值: {pipeline.best_threshold:.4f}")
    logger.info("=" * 50)
    
    return pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
