# -*- coding: utf-8 -*-
"""
模型配置模块

定义模型文件保存路径、Optuna 超参数搜索空间等配置。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple


@dataclass
class ModelConfig:
    """
    模型配置类
    
    定义模型训练和保存相关的配置参数。
    """
    
    # ==================== 路径配置 ====================
    # 项目根目录（相对于此文件的位置计算）
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    # 模型产物保存目录
    ARTIFACTS_DIR: Path = field(init=False)
    
    # 数据目录
    DATA_DIR: Path = field(init=False)
    
    def __post_init__(self):
        """初始化后处理：设置派生路径"""
        self.ARTIFACTS_DIR = self.PROJECT_ROOT / "artifacts"
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        
        # 确保目录存在
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==================== 模型文件名 ====================
    # 各组件保存的文件名
    SCALER_FILENAME: str = "standard_scaler.joblib"
    KMEANS_FILENAME: str = "kmeans_clusterer.joblib"
    CATBOOST_FILENAME: str = "catboost_classifier.joblib"
    THRESHOLD_FILENAME: str = "optimal_threshold.joblib"
    PIPELINE_FILENAME: str = "full_pipeline.joblib"
    SHAP_EXPLAINER_FILENAME: str = "shap_explainer.joblib"
    
    # ==================== 模型文件完整路径 ====================
    def get_scaler_path(self) -> Path:
        """获取标准化器保存路径"""
        return self.ARTIFACTS_DIR / self.SCALER_FILENAME
    
    def get_kmeans_path(self) -> Path:
        """获取 K-Means 模型保存路径"""
        return self.ARTIFACTS_DIR / self.KMEANS_FILENAME
    
    def get_catboost_path(self) -> Path:
        """获取 CatBoost 模型保存路径"""
        return self.ARTIFACTS_DIR / self.CATBOOST_FILENAME
    
    def get_threshold_path(self) -> Path:
        """获取最优阈值保存路径"""
        return self.ARTIFACTS_DIR / self.THRESHOLD_FILENAME
    
    def get_pipeline_path(self) -> Path:
        """获取完整流水线保存路径"""
        return self.ARTIFACTS_DIR / self.PIPELINE_FILENAME
    
    def get_shap_explainer_path(self) -> Path:
        """获取 SHAP 解释器保存路径"""
        return self.ARTIFACTS_DIR / self.SHAP_EXPLAINER_FILENAME
    
    # ==================== 训练配置 ====================
    # 随机种子（保证可复现性）
    RANDOM_STATE: int = 42
    
    # K-Means 聚类数
    N_CLUSTERS: int = 3
    
    # 交叉验证折数
    N_FOLDS: int = 5
    
    # 测试集比例
    TEST_SIZE: float = 0.2
    
    # 目标召回率阈值（宁可误报，不可漏报）
    TARGET_RECALL: float = 0.95


@dataclass
class OptunaSearchSpace:
    """
    Optuna 超参数搜索空间配置
    
    定义 CatBoost 模型的超参数搜索范围。
    """
    
    # ==================== Optuna 配置 ====================
    # 优化试验次数
    N_TRIALS: int = 50
    
    # 优化方向（最大化 F1-score 或 AUC）
    DIRECTION: str = "maximize"
    
    # 优化超时时间（秒），None 表示不限制
    TIMEOUT: int = None
    
    # ==================== CatBoost 超参数搜索空间 ====================
    # 学习率范围
    LEARNING_RATE_MIN: float = 0.01
    LEARNING_RATE_MAX: float = 0.3
    
    # 树深度范围
    DEPTH_MIN: int = 4
    DEPTH_MAX: int = 10
    
    # 迭代次数范围
    ITERATIONS_MIN: int = 100
    ITERATIONS_MAX: int = 1000
    
    # L2 正则化系数范围
    L2_LEAF_REG_MIN: float = 1.0
    L2_LEAF_REG_MAX: float = 10.0
    
    # 随机强度范围（用于防止过拟合）
    RANDOM_STRENGTH_MIN: float = 0.0
    RANDOM_STRENGTH_MAX: float = 10.0
    
    # Bagging 温度范围
    BAGGING_TEMPERATURE_MIN: float = 0.0
    BAGGING_TEMPERATURE_MAX: float = 1.0
    
    # 边界数量范围（用于数值特征分箱）
    BORDER_COUNT_OPTIONS: Tuple[int, ...] = (32, 64, 128, 254)
    
    def get_search_space_dict(self) -> Dict[str, Any]:
        """
        获取搜索空间配置字典（用于 Optuna trial.suggest_* 方法）
        
        Returns:
            Dict[str, Any]: 包含各超参数搜索范围的字典
        """
        return {
            "learning_rate": {
                "type": "float",
                "low": self.LEARNING_RATE_MIN,
                "high": self.LEARNING_RATE_MAX,
                "log": True,  # 对数均匀采样
            },
            "depth": {
                "type": "int",
                "low": self.DEPTH_MIN,
                "high": self.DEPTH_MAX,
            },
            "iterations": {
                "type": "int",
                "low": self.ITERATIONS_MIN,
                "high": self.ITERATIONS_MAX,
                "step": 50,
            },
            "l2_leaf_reg": {
                "type": "float",
                "low": self.L2_LEAF_REG_MIN,
                "high": self.L2_LEAF_REG_MAX,
            },
            "random_strength": {
                "type": "float",
                "low": self.RANDOM_STRENGTH_MIN,
                "high": self.RANDOM_STRENGTH_MAX,
            },
            "bagging_temperature": {
                "type": "float",
                "low": self.BAGGING_TEMPERATURE_MIN,
                "high": self.BAGGING_TEMPERATURE_MAX,
            },
            "border_count": {
                "type": "categorical",
                "choices": list(self.BORDER_COUNT_OPTIONS),
            },
        }


# 全局配置实例
MODEL_CONFIG = ModelConfig()
OPTUNA_SEARCH_SPACE = OptunaSearchSpace()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    config = ModelConfig()
    print(f"项目根目录: {config.PROJECT_ROOT}")
    print(f"模型保存目录: {config.ARTIFACTS_DIR}")
    print(f"数据目录: {config.DATA_DIR}")
    print(f"\n模型文件路径:")
    print(f"  Scaler: {config.get_scaler_path()}")
    print(f"  KMeans: {config.get_kmeans_path()}")
    print(f"  CatBoost: {config.get_catboost_path()}")
    print(f"  Threshold: {config.get_threshold_path()}")
    
    optuna_config = OptunaSearchSpace()
    print(f"\nOptuna 配置:")
    print(f"  试验次数: {optuna_config.N_TRIALS}")
    print(f"  学习率范围: [{optuna_config.LEARNING_RATE_MIN}, {optuna_config.LEARNING_RATE_MAX}]")
    print(f"  树深度范围: [{optuna_config.DEPTH_MIN}, {optuna_config.DEPTH_MAX}]")
