# -*- coding: utf-8 -*-
"""
混合流水线模块 - 项目核心

实现完整的风险预测流水线：
1. 标准化数据
2. K-Means 聚类生成心理画像标签
3. 特征拼接
4. CatBoost 分类预测
5. 阈值移动策略（确保高召回率）
"""

import logging
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from .base_model import SupervisedModel
from .kmeans_clusterer import KMeansClusterer
from .catboost_classifier import CatBoostWrapper
from ..config.feature_config import FeatureConfig, FEATURE_CONFIG
from ..config.model_config import ModelConfig, MODEL_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskPipeline(SupervisedModel):
    """
    风险预测混合流水线
    
    核心业务逻辑：
    1. 标准化 -> K-Means (k=3) -> 特征拼接 (Cluster_Label) -> CatBoost
    2. 使用阈值移动策略，确保召回率 >= 95%（宁可误报，不可漏报）
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        target_recall: float = 0.95,
        random_state: int = 42,
        feature_config: FeatureConfig = None,
        model_config: ModelConfig = None,
        catboost_params: Dict[str, Any] = None,
        name: str = "RiskPipeline"
    ):
        """
        初始化风险预测流水线
        
        Args:
            n_clusters: K-Means 聚类数
            target_recall: 目标召回率（默认 0.95）
            random_state: 随机种子
            feature_config: 特征配置
            model_config: 模型配置
            catboost_params: CatBoost 自定义参数
            name: 流水线名称
        """
        super().__init__(name=name)
        
        self.n_clusters = n_clusters
        self.target_recall = target_recall
        self.random_state = random_state
        self.feature_config = feature_config or FEATURE_CONFIG
        self.model_config = model_config or MODEL_CONFIG
        self.catboost_params = catboost_params or {}
        
        # 流水线组件
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeansClusterer] = None
        self.classifier: Optional[CatBoostWrapper] = None
        
        # 最优阈值（核心：阈值移动策略）
        self.best_threshold: float = 0.5  # 默认值，训练后会更新
        
        # 特征名
        self.feature_names_in_: Optional[list] = None
        self.feature_names_augmented_: Optional[list] = None
        
        # 训练指标
        self.training_metrics_: Optional[Dict[str, Any]] = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        validation_split: float = 0.2
    ) -> "RiskPipeline":
        """
        拟合整个流水线
        
        流程：
        1. 划分训练集和验证集
        2. 标准化数据
        3. K-Means 聚类生成 Cluster_Label
        4. 特征拼接
        5. 训练 CatBoost（可通过 Optuna 调参后传入最优参数）
        6. 在验证集上计算最优阈值（确保召回率 >= target_recall）
        
        Args:
            X: 特征矩阵
            y: 目标变量
            validation_split: 验证集比例（用于计算最优阈值）
            
        Returns:
            self
        """
        logger.info(f"[{self.name}] ========== 开始训练流水线 ==========")
        
        # 保存特征名
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = np.asarray(X)
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.asarray(y)
        
        logger.info(f"[{self.name}] 输入数据形状: X={X_array.shape}, y={y_array.shape}")
        logger.info(f"[{self.name}] 目标分布: 正类={np.sum(y_array==1)}, 负类={np.sum(y_array==0)}")
        
        # 1. 划分训练集和验证集（验证集用于计算最优阈值）
        X_train, X_val, y_train, y_val = train_test_split(
            X_array, y_array,
            test_size=validation_split,
            random_state=self.random_state,
            stratify=y_array  # 分层抽样
        )
        logger.info(f"[{self.name}] 训练集: {X_train.shape[0]} 样本, 验证集: {X_val.shape[0]} 样本")
        
        # 2. 标准化
        logger.info(f"[{self.name}] Step 1: 标准化数据...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 3. K-Means 聚类
        logger.info(f"[{self.name}] Step 2: K-Means 聚类 (k={self.n_clusters})...")
        self.kmeans = KMeansClusterer(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        cluster_labels_train = self.kmeans.fit_predict(X_train_scaled)
        cluster_labels_val = self.kmeans.predict(X_val_scaled)
        
        # 4. 特征拼接：将聚类标签作为新特征
        logger.info(f"[{self.name}] Step 3: 特征拼接（添加 Cluster_Label）...")
        X_train_augmented = np.column_stack([X_train_scaled, cluster_labels_train])
        X_val_augmented = np.column_stack([X_val_scaled, cluster_labels_val])
        
        # 更新增强后的特征名
        self.feature_names_augmented_ = (
            [f"{name}_scaled" for name in self.feature_names_in_] + 
            [self.feature_config.CLUSTER_LABEL]
        )
        
        logger.info(f"[{self.name}] 增强后特征数: {X_train_augmented.shape[1]}")
        
        # 5. 训练 CatBoost 分类器
        logger.info(f"[{self.name}] Step 4: 训练 CatBoost 分类器...")
        
        # 合并默认参数和自定义参数
        classifier_params = {
            "random_state": self.random_state,
            "verbose": False,
            **self.catboost_params
        }
        
        self.classifier = CatBoostWrapper(**classifier_params)
        self.classifier.fit(
            X_train_augmented, 
            y_train,
            eval_set=(X_val_augmented, y_val)
        )
        
        # 6. 计算最优阈值（核心：阈值移动策略）
        logger.info(f"[{self.name}] Step 5: 计算最优阈值（目标召回率 >= {self.target_recall}）...")
        self.best_threshold = self._find_optimal_threshold(
            X_val_augmented, 
            y_val
        )
        
        # 7. 计算训练指标
        self._compute_training_metrics(X_val_augmented, y_val)
        
        self._is_fitted = True
        logger.info(f"[{self.name}] ========== 流水线训练完成 ==========")
        
        return self
    
    def _find_optimal_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        寻找最优阈值（阈值移动策略）
        
        核心逻辑：
        - 在 P-R 曲线上找到使 Recall >= target_recall 的最大阈值
        - 这确保了"宁可误报，不可漏报"的业务需求
        
        Args:
            X_val: 验证集特征
            y_val: 验证集标签
            
        Returns:
            float: 最优阈值
        """
        # 获取正类概率
        y_proba = self.classifier.predict_proba(X_val)[:, 1]
        
        # 计算 P-R 曲线
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
        
        # 找到满足召回率要求的阈值
        # 注意：precision_recall_curve 返回的 recalls 是降序的
        valid_indices = np.where(recalls[:-1] >= self.target_recall)[0]
        
        if len(valid_indices) == 0:
            # 如果没有满足条件的阈值，使用最小阈值（最大召回）
            logger.warning(
                f"[{self.name}] 警告: 无法达到目标召回率 {self.target_recall}，"
                f"使用最大召回率对应的阈值"
            )
            optimal_threshold = thresholds[0] if len(thresholds) > 0 else 0.5
        else:
            # 在满足召回率的阈值中，选择最大的（精确率最高的）
            optimal_idx = valid_indices[-1]
            optimal_threshold = thresholds[optimal_idx]
        
        # 计算最优阈值下的性能
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_val, y_pred_optimal)
        
        tn, fp, fn, tp = cm.ravel()
        recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        logger.info(f"[{self.name}] 最优阈值: {optimal_threshold:.4f}")
        logger.info(f"[{self.name}] 该阈值下的召回率: {recall_at_threshold:.4f}")
        logger.info(f"[{self.name}] 该阈值下的精确率: {precision_at_threshold:.4f}")
        logger.info(f"[{self.name}] 混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        return optimal_threshold
    
    def _compute_training_metrics(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """
        计算并保存训练指标
        
        Args:
            X_val: 验证集特征
            y_val: 验证集标签
        """
        y_proba = self.classifier.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= self.best_threshold).astype(int)
        
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.training_metrics_ = {
            "best_threshold": self.best_threshold,
            "auc_roc": roc_auc_score(y_val, y_proba),
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "cluster_distribution": self.kmeans.cluster_sizes_
        }
        
        logger.info(f"[{self.name}] AUC-ROC: {self.training_metrics_['auc_roc']:.4f}")
    
    def _transform_features(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        特征转换（标准化 + 聚类标签拼接）
        
        Args:
            X: 原始特征矩阵
            
        Returns:
            np.ndarray: 增强后的特征矩阵
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        # 标准化
        X_scaled = self.scaler.transform(X_array)
        
        # 聚类
        cluster_labels = self.kmeans.predict(X_scaled)
        
        # 拼接
        X_augmented = np.column_stack([X_scaled, cluster_labels])
        
        return X_augmented
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        预测类别标签（使用最优阈值）
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测标签 (0=低风险, 1=高风险)
        """
        self._check_is_fitted()
        
        X_augmented = self._transform_features(X)
        y_proba = self.classifier.predict_proba(X_augmented)[:, 1]
        
        # 使用最优阈值进行分类
        return (y_proba >= self.best_threshold).astype(int)
    
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
        
        X_augmented = self._transform_features(X)
        return self.classifier.predict_proba(X_augmented)
    
    def predict_risk_level(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        预测风险等级（更详细的输出）
        
        Args:
            X: 特征矩阵
            
        Returns:
            pd.DataFrame: 包含风险概率、等级、聚类标签的数据框
        """
        self._check_is_fitted()
        
        X_augmented = self._transform_features(X)
        proba = self.classifier.predict_proba(X_augmented)[:, 1]
        labels = (proba >= self.best_threshold).astype(int)
        
        # 获取聚类标签
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        X_scaled = self.scaler.transform(X_array)
        cluster_labels = self.kmeans.predict(X_scaled)
        
        # 风险等级分类
        def get_risk_level(p: float) -> str:
            if p < 0.3:
                return "低风险"
            elif p < self.best_threshold:
                return "中风险"
            elif p < 0.8:
                return "高风险"
            else:
                return "极高风险"
        
        result = pd.DataFrame({
            "风险概率": proba,
            "预测标签": labels,
            "风险等级": [get_risk_level(p) for p in proba],
            "心理画像簇": cluster_labels,
            "是否需要关注": labels == 1
        })
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            pd.DataFrame: 特征重要性排名
        """
        self._check_is_fitted()
        return self.classifier.get_feature_importance()
    
    def save_all(self, output_dir: Union[str, Path] = None) -> None:
        """
        保存所有模型组件
        
        Args:
            output_dir: 输出目录，默认使用配置中的 artifacts 目录
        """
        self._check_is_fitted()
        
        if output_dir is None:
            output_dir = self.model_config.ARTIFACTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存各组件
        joblib.dump(self.scaler, output_dir / self.model_config.SCALER_FILENAME)
        self.kmeans.save(output_dir / self.model_config.KMEANS_FILENAME)
        self.classifier.save(output_dir / self.model_config.CATBOOST_FILENAME)
        
        # 保存阈值和元信息
        threshold_info = {
            "best_threshold": self.best_threshold,
            "target_recall": self.target_recall,
            "n_clusters": self.n_clusters,
            "feature_names_in": self.feature_names_in_,
            "feature_names_augmented": self.feature_names_augmented_,
            "training_metrics": self.training_metrics_
        }
        joblib.dump(threshold_info, output_dir / self.model_config.THRESHOLD_FILENAME)
        
        # 保存完整流水线
        self.save(output_dir / self.model_config.PIPELINE_FILENAME)
        
        logger.info(f"[{self.name}] 所有模型组件已保存至: {output_dir}")
    
    @classmethod
    def load_all(
        cls, 
        input_dir: Union[str, Path] = None,
        model_config: ModelConfig = None
    ) -> "RiskPipeline":
        """
        加载所有模型组件
        
        Args:
            input_dir: 输入目录
            model_config: 模型配置
            
        Returns:
            RiskPipeline: 加载的流水线实例
        """
        config = model_config or MODEL_CONFIG
        if input_dir is None:
            input_dir = config.ARTIFACTS_DIR
        input_dir = Path(input_dir)
        
        # 加载完整流水线
        pipeline = cls.load(input_dir / config.PIPELINE_FILENAME)
        
        logger.info(f"[RiskPipeline] 流水线已从 {input_dir} 加载")
        logger.info(f"[RiskPipeline] 最优阈值: {pipeline.best_threshold}")
        
        return pipeline


# ==================== 使用示例 ====================
if __name__ == "__main__":
    print("混合流水线模块加载成功")
    print("使用方法:")
    print("  from ml_core.models import RiskPipeline")
    print("  pipeline = RiskPipeline(n_clusters=3, target_recall=0.95)")
    print("  pipeline.fit(X, y)")
    print("  predictions = pipeline.predict(X_new)")
    print("  pipeline.save_all()")