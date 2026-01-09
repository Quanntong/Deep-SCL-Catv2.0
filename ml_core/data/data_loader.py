# -*- coding: utf-8 -*-
"""
数据加载模块

负责从 Excel 文件加载数据，进行清洗和特征工程。
"""

import logging
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from ..config.feature_config import FEATURE_CONFIG

logger = logging.getLogger(__name__)


def load_single_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    加载单个 Excel 文件
    
    Args:
        filepath: Excel 文件路径
        
    Returns:
        pd.DataFrame: 原始数据
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    logger.info(f"正在加载: {filepath}")
    df = pd.read_excel(filepath)
    logger.info(f"加载完成，共 {len(df)} 条记录")
    
    return df


def load_all_data(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    加载目录下所有年级的数据文件并合并
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        pd.DataFrame: 合并后的数据
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 查找所有 xlsx 文件（排除临时文件）
    xlsx_files = [f for f in data_dir.glob("*.xlsx") if not f.name.startswith("~$")]
    
    if not xlsx_files:
        raise FileNotFoundError(f"目录中没有找到 xlsx 文件: {data_dir}")
    
    logger.info(f"找到 {len(xlsx_files)} 个数据文件")
    
    # 加载并合并所有文件
    dfs = []
    for f in xlsx_files:
        try:
            df = load_single_file(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"加载文件失败 {f}: {e}")
    
    if not dfs:
        raise ValueError("没有成功加载任何数据文件")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"合并完成，共 {len(combined)} 条记录")
    
    return combined


def load_cleaned_data(
    filepath: Union[str, Path] = None,
    data_dir: Union[str, Path] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    加载并清洗数据，生成训练所需的特征矩阵和目标变量
    
    Args:
        filepath: 单个文件路径（优先使用）
        data_dir: 数据目录路径（加载所有文件）
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (特征矩阵 X, 目标变量 y)
    """
    # 加载数据
    if filepath is not None:
        df = load_single_file(filepath)
    elif data_dir is not None:
        df = load_all_data(data_dir)
    else:
        raise ValueError("必须指定 filepath 或 data_dir")
    
    # 打印列名用于调试
    logger.info(f"数据列名: {df.columns.tolist()}")
    
    # 检查必需的列是否存在
    required_cols = FEATURE_CONFIG.all_features + [FEATURE_CONFIG.TARGET_REG]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必需的列: {missing_cols}")
    
    # 提取特征列
    feature_cols = FEATURE_CONFIG.all_features
    X = df[feature_cols].copy()
    
    # 处理缺失值：使用中位数填充
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            logger.info(f"列 '{col}' 缺失值已用中位数 {median_val:.2f} 填充")
    
    # 生成二分类目标变量：挂科数目 > 0 则为高风险
    y = (df[FEATURE_CONFIG.TARGET_REG] > 0).astype(int)
    y.name = FEATURE_CONFIG.TARGET_LABEL
    
    # 统计目标分布
    n_risk = y.sum()
    n_total = len(y)
    logger.info(f"目标变量分布: 高风险={n_risk} ({n_risk/n_total*100:.1f}%), "
                f"低风险={n_total-n_risk} ({(n_total-n_risk)/n_total*100:.1f}%)")
    
    return X, y


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    print("数据加载模块测试")
    print("使用方法:")
    print("  from ml_core.data.data_loader import load_cleaned_data")
    print("  X, y = load_cleaned_data(data_dir='data/raw')")
