# -*- coding: utf-8 -*-
"""
K-Means 聚类器封装

封装 scikit-learn 的 KMeans，提供统一接口。
用于生成学生心理画像聚类标签。
"""

import logging
from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base_model import UnsupervisedModel

# 配置日志
logger = logging.getLogger(__name__)


class KMeansClusterer(UnsupervisedModel):
    """
    K-Means 聚类器
    
    用于对标准化后的心理特征进行聚类，生成心理画像标签。
    注意：输入数据必须先经过标准化处理。
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        name: str = "KMeansClusterer"
    ):
        """
        初始化 K-Means 聚类器
        
        Args:
            n_clusters: 聚类数量，默认 3（对应三种心理画像）
            random_state: 随机种子
            n_init: K-Means 运行次数，选择最佳结果
            max_iter: 单次运行最大迭代次数
            name: 模型名称
        """
        super().__init__(name=name)
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        
        # 内部 KMeans 实例
        self._kmeans: Optional[KMeans] = None
        
        # 聚类统计信息
        self.cluster_sizes_: Optional[dict] = None
        self.inertia_: Optional[float] = None
    
    def _fit_impl(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> "KMeansClusterer":
        """
        拟合聚类器
        
        Args:
            X: 标准化后的特征矩阵
            
        Returns:
            self
        """
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        logger.info(f"[{self.name}] 开始拟合，输入形状: {X_array.shape}")
        
        # 初始化并拟合 KMeans
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter
        )
        self._kmeans.fit(X_array)
        
        # 记录统计信息
        labels = self._kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        self.cluster_sizes_ = dict(zip(unique.tolist(), counts.tolist()))
        self.inertia_ = self._kmeans.inertia_
        
        logger.info(f"[{self.name}] 聚类完成")
        logger.info(f"[{self.name}] 各簇样本数: {self.cluster_sizes_}")
        logger.info(f"[{self.name}] 惯性（Inertia）: {self.inertia_:.4f}")
        
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        预测聚类标签（用于新数据）
        
        Args:
            X: 标准化后的特征矩阵
            
        Returns:
            np.ndarray: 聚类标签数组
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        return self._kmeans.predict(X_array)
    
    def fit_predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        拟合并预测（用于训练集）
        
        Args:
            X: 标准化后的特征矩阵
            
        Returns:
            np.ndarray: 聚类标签数组
        """
        self._fit_impl(X)
        return self._kmeans.labels_
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        获取聚类中心
        
        Returns:
            np.ndarray: 聚类中心矩阵，形状为 (n_clusters, n_features)
        """
        self._check_is_fitted()
        return self._kmeans.cluster_centers_
    
    def get_distances_to_centers(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        计算样本到各聚类中心的距离
        
        Args:
            X: 标准化后的特征矩阵
            
        Returns:
            np.ndarray: 距离矩阵，形状为 (n_samples, n_clusters)
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        return self._kmeans.transform(X_array)
    
    def describe_clusters(
        self, 
        X_original: Union[pd.DataFrame, np.ndarray],
        feature_names: list = None
    ) -> pd.DataFrame:
        """
        描述各聚类的特征统计信息
        
        Args:
            X_original: 原始特征数据（未标准化），用于可解释性
            feature_names: 特征名列表
            
        Returns:
            pd.DataFrame: 各聚类的特征均值
        """
        self._check_is_fitted()
        
        # 需要先用标准化数据获取标签，再用原始数据计算统计
        # 这里假设传入的是原始数据，需要用户自行确保一致性
        
        if isinstance(X_original, np.ndarray):
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X_original.shape[1])]
            df = pd.DataFrame(X_original, columns=feature_names)
        else:
            df = X_original.copy()
        
        # 注意：这里的标签需要从外部传入或重新计算
        # 暂时返回聚类中心信息
        centers = self.get_cluster_centers()
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(centers.shape[1])]
        
        return pd.DataFrame(
            centers,
            columns=feature_names,
            index=[f"Cluster_{i}" for i in range(self.n_clusters)]
        )
    
    @property
    def labels_(self) -> np.ndarray:
        """获取训练集的聚类标签"""
        self._check_is_fitted()
        return self._kmeans.labels_


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例代码
    print("K-Means 聚类器模块加载成功")
    print("使用方法:")
    print("  from ml_core.models import KMeansClusterer")
    print("  clusterer = KMeansClusterer(n_clusters=3)")
    print("  labels = clusterer.fit_predict(X_scaled)")