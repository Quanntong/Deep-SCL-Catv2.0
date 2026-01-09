# -*- coding: utf-8 -*-
"""
训练脚本 - 使用真实数据训练 CatBoost + KMeans 模型
运行: python backend/train_models.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

# 中文列名 -> 英文键名映射
COLUMN_MAP = {
    "躯体化": "somatization",
    "强迫症状": "obsessive_compulsive", 
    "人际关系敏感": "interpersonal_sensitivity",
    "抑郁": "depression",
    "焦虑": "anxiety",
    "敌对": "hostility",
    "恐怖": "phobic_anxiety",
    "偏执": "paranoid_ideation",
    "精神病性": "psychoticism",
    "其他": "other",
    "内外向E": "E",
    "神经质N": "N",
    "精神质P": "P",
    "掩饰性L": "L",
    "挂科数目": "failed_count"
}

# 特征列
SCL90_FACTORS = ["somatization", "obsessive_compulsive", "interpersonal_sensitivity",
                 "depression", "anxiety", "hostility", "phobic_anxiety",
                 "paranoid_ideation", "psychoticism", "other"]
EPQ_FACTORS = ["E", "P", "N", "L"]
ALL_FEATURES = SCL90_FACTORS + EPQ_FACTORS


def load_all_data():
    """加载所有年级数据"""
    dfs = []
    for file in DATA_DIR.glob("*.xlsx"):
        if file.name.startswith("~$"):
            continue
        print(f"加载: {file.name}")
        df = pd.read_excel(file)
        dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"未找到数据文件，请将 Excel 文件放入 {DATA_DIR}")
    
    return pd.concat(dfs, ignore_index=True)


def preprocess(df):
    """预处理数据"""
    # 重命名列
    df = df.rename(columns=COLUMN_MAP)
    
    # 检查必要列
    missing = [c for c in ALL_FEATURES + ["failed_count"] if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    
    # 提取特征和目标
    X = df[ALL_FEATURES].copy()
    y_class = (df["failed_count"] > 0).astype(int)  # 二分类: 是否挂科
    y_reg = df["failed_count"].copy()  # 回归: 挂科数目
    
    # 填充缺失值
    X = X.fillna(X.median())
    y_reg = y_reg.fillna(0)
    
    print(f"样本数: {len(X)}, 风险样本: {y_class.sum()} ({y_class.mean()*100:.1f}%)")
    return X, y_class, y_reg


def train_models(X, y_class, y_reg):
    """训练所有模型"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. StandardScaler
    print("\n[1/4] 训练 StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[SCL90_FACTORS])
    joblib.dump(scaler, MODELS_DIR / "standard_scaler.joblib")
    print(f"  ✓ 保存: standard_scaler.joblib")
    
    # 2. KMeans 聚类
    print("\n[2/4] 训练 KMeans (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    joblib.dump(kmeans, MODELS_DIR / "kmeans_clusterer.joblib")
    print(f"  ✓ 保存: kmeans_clusterer.joblib")
    print(f"  聚类分布: {np.bincount(cluster_labels)}")
    
    # 3. 添加聚类特征
    X_full = X.copy()
    X_full["cluster"] = cluster_labels
    
    # 划分训练/测试集
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        X_full, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # 4. CatBoost 分类器
    print("\n[3/4] 训练 CatBoost Classifier...")
    clf = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        cat_features=["cluster"],
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100
    )
    clf.fit(X_train, y_train_cls, eval_set=(X_test, y_test_cls), early_stopping_rounds=50)
    
    # 保存分类器
    clf.save_model(str(MODELS_DIR / "catboost_classifier.cbm"))
    joblib.dump(clf, MODELS_DIR / "catboost_classifier.joblib")
    print(f"  ✓ 保存: catboost_classifier.cbm / .joblib")
    
    # 评估
    train_acc = clf.score(X_train, y_train_cls)
    test_acc = clf.score(X_test, y_test_cls)
    print(f"  训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
    
    # 特征重要性
    importance = clf.get_feature_importance()
    feature_names = list(X_full.columns)
    print("\n  特征重要性 Top 5:")
    for idx in np.argsort(importance)[-5:][::-1]:
        print(f"    {feature_names[idx]}: {importance[idx]:.2f}")
    
    # 5. CatBoost 回归器 (Poisson 损失函数，适合计数数据)
    print("\n[4/4] 训练 CatBoost Regressor (Poisson)...")
    reg = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Poisson',
        cat_features=["cluster"],
        random_seed=42,
        verbose=100
    )
    reg.fit(X_train, y_train_reg, eval_set=(X_test, y_test_reg), early_stopping_rounds=50)
    
    reg.save_model(str(MODELS_DIR / "catboost_regressor.cbm"))
    joblib.dump(reg, MODELS_DIR / "catboost_regressor.joblib")
    print(f"  ✓ 保存: catboost_regressor.cbm / .joblib")
    
    # 计算最优阈值 (最大化 F1-Score)
    from sklearn.metrics import f1_score
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    best_f1, best_threshold = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test_cls, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, thresh
    
    print(f"\n最优阈值 (F1={best_f1:.4f}): {best_threshold:.4f}")
    
    # 保存阈值到 JSON
    import json
    config = {"threshold": float(best_threshold)}
    with open(MODELS_DIR / "model_config.json", "w") as f:
        json.dump(config, f)
    print(f"  ✓ 保存: model_config.json")
    
    joblib.dump(best_threshold, MODELS_DIR / "optimal_threshold.joblib")
    
    print("\n" + "="*50)
    print("✓ 所有模型训练完成！")
    print(f"模型保存位置: {MODELS_DIR}")


def main():
    print("="*50)
    print("心理风险预测模型训练")
    print("="*50)
    
    # 加载数据
    df = load_all_data()
    
    # 预处理
    X, y_class, y_reg = preprocess(df)
    
    # 训练
    train_models(X, y_class, y_reg)


if __name__ == "__main__":
    main()
