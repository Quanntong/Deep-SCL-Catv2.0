# -*- coding: utf-8 -*-
"""
基础模型抽象类

定义所有模型组件必须实现的接口，确保一致性和可替换性。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union, Optional

import numpy as np
import pandas as pd
import joblib


class BaseModel(ABC):
    """
    模型抽象基类
    
    所有模型组件（聚类器、分类器、流水线等）都应继承此类，
    并实现定义的抽象方法。
    """
    
    def __init__(self, name: str = "BaseModel"):
        """
        初始化基类
        
        Args:
            name: 模型名称，用于日志和保存
        """
        self.name = name
        self._is_fitted = False
    
    @abstractmethod
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "BaseModel":
        """
        拟合模型
        
        Args:
            X: 特征矩阵
            y: 目标变量（监督学习需要，无监督学习可选）
            
        Returns:
            self: 返回自身，支持链式调用
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测结果
        """
        pass
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        print(f"[{self.name}] 模型已保存至: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseModel":
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            BaseModel: 加载的模型实例
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        model = joblib.load(filepath)
        print(f"[{model.name}] 模型已从 {filepath} 加载")
        
        return model
    
    @property
    def is_fitted(self) -> bool:
        """检查模型是否已拟合"""
        return self._is_fitted
    
    def _check_is_fitted(self) -> None:
        """
        检查模型是否已拟合，未拟合则抛出异常
        
        Raises:
            RuntimeError: 模型未拟合
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"[{self.name}] 模型尚未拟合，请先调用 fit() 方法"
            )
    
    def __repr__(self) -> str:
        """模型的字符串表示"""
        status = "已拟合" if self._is_fitted else "未拟合"
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"


class SupervisedModel(BaseModel):
    """
    监督学习模型抽象类
    
    扩展基类，添加概率预测等监督学习特有方法。
    """
    
    @abstractmethod
    def predict_proba(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测概率，形状为 (n_samples, n_classes)
        """
        pass
    
    def get_positive_proba(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        获取正类（高危）的预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 正类概率，形状为 (n_samples,)
        """
        proba = self.predict_proba(X)
        
        # 二分类情况：返回第二列（正类概率）
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        
        return proba


class UnsupervisedModel(BaseModel):
    """
    无监督学习模型抽象类
    
    用于聚类等无监督学习任务。
    """
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "UnsupervisedModel":
        """
        拟合模型（无监督学习忽略 y 参数）
        
        Args:
            X: 特征矩阵
            y: 忽略
            
        Returns:
            self
        """
        return self._fit_impl(X)
    
    @abstractmethod
    def _fit_impl(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> "UnsupervisedModel":
        """
        实际的拟合实现（子类重写）
        
        Args:
            X: 特征矩阵
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def fit_predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        拟合并预测（一步完成）
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测结果（如聚类标签）
        """
        pass