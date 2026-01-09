# -*- coding: utf-8 -*-
"""
特征配置模块

定义数据集中的特征列名、元数据列、目标列等配置。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureConfig:
    """特征配置类"""
    
    # SCL-90 心理因子 (10个维度)
    SCL90_FACTORS: List[str] = field(default_factory=lambda: [
        "躯体化", "强迫症状", "人际关系敏感", "抑郁", "焦虑",
        "敌对", "恐怖", "偏执", "精神病性", "其他"
    ])
    
    # EPQ 人格因子 (4个维度)
    EPQ_FACTORS: List[str] = field(default_factory=lambda: [
        "内外向E", "神经质N", "精神质P", "掩饰性L"
    ])
    
    # 元数据列 (不参与训练)
    META_COLUMNS: List[str] = field(default_factory=lambda: [
        "学号", "姓名", "班级"
    ])
    
    # 原始目标列 (回归目标)
    TARGET_REG: str = "挂科数目"
    
    # 衍生目标列 (二分类标签)
    TARGET_LABEL: str = "is_risk"
    
    # 聚类标签列名
    CLUSTER_LABEL: str = "Cluster_Label"
    
    @property
    def all_features(self) -> List[str]:
        """获取所有特征列 (SCL-90 + EPQ)"""
        return self.SCL90_FACTORS + self.EPQ_FACTORS
    
    @property
    def n_scl90_features(self) -> int:
        """SCL-90 特征数量"""
        return len(self.SCL90_FACTORS)
    
    @property
    def n_epq_features(self) -> int:
        """EPQ 特征数量"""
        return len(self.EPQ_FACTORS)
    
    @property
    def n_total_features(self) -> int:
        """总特征数量"""
        return len(self.all_features)


# 全局配置实例
FEATURE_CONFIG = FeatureConfig()
