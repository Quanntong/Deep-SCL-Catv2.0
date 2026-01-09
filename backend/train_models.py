# -*- coding: utf-8 -*-
"""
训练脚本 - 使用 Logistic Regression + Ridge 线性模型（适合小样本）
运行: python backend/train_models.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

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

SCL90_FACTORS = ["somatization", "obsessive_compulsive", "interpersonal_sensitivity",
                 "depression", "anxiety", "hostility", "phobic_anxiety",
                 "paranoid_ideation", "psychoticism", "other"]
EPQ_FACTORS = ["E", "P", "N", "L"]
ALL_FEATURES = SCL90_FACTORS + EPQ_FACTORS


def load_all_data():
    dfs = []
    for file in DATA_DIR.glob("*.xlsx"):
        if file.name.startswith("~$"):
            continue
        print(f"加载: {file.name}")
        df = pd.read_excel(file)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"未找到数据文件，请将 Excel 放入 {DATA_DIR}")
    return pd.concat(dfs, ignore_index=True)


def preprocess(df):
    df = df.rename(columns=COLUMN_MAP)
    missing = [c for c in ALL_FEATURES + ["failed_count"] if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    
    X = df[ALL_FEATURES].copy()
    y_class = (df["failed_count"] > 0).astype(int)
    y_reg = df["failed_count"].fillna(0).copy()
    X = X.fillna(X.median())
    
    print(f"样本数: {len(X)}, 风险样本: {y_class.sum()} ({y_class.mean()*100:.1f}%)")
    return X, y_class, y_reg


def train_models(X, y_class, y_reg):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. StandardScaler
    print("\n[1/4] 训练 StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[SCL90_FACTORS])
    joblib.dump(scaler, MODELS_DIR / "standard_scaler.joblib")
    
    # 2. KMeans
    print("[2/4] 训练 KMeans (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    joblib.dump(kmeans, MODELS_DIR / "kmeans_clusterer.joblib")
    print(f"  聚类分布: {np.bincount(cluster_labels)}")
    
    # 3. 组合特征
    X_full = X.copy()
    X_full["cluster"] = cluster_labels
    
    # 全部标准化（线性模型需要）
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)
    joblib.dump(scaler_full, MODELS_DIR / "full_scaler.joblib")
    
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        X_full_scaled, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # 4. Logistic Regression 分类器
    print("[3/4] 训练 Logistic Regression...")
    clf = LogisticRegression(class_weight='balanced', C=1.0, solver='liblinear', random_state=42)
    clf.fit(X_train, y_train_cls)
    joblib.dump(clf, MODELS_DIR / "logistic_classifier.pkl")
    
    train_acc = clf.score(X_train, y_train_cls)
    test_acc = clf.score(X_test, y_test_cls)
    print(f"  训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
    
    # 特征重要性（系数绝对值）
    feature_names = list(X.columns) + ["cluster"]
    importance = np.abs(clf.coef_[0])
    print("\n  特征重要性 Top 5:")
    for idx in np.argsort(importance)[-5:][::-1]:
        print(f"    {feature_names[idx]}: {importance[idx]:.3f}")
    
    # 5. Ridge 回归器
    print("\n[4/4] 训练 Ridge Regression...")
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train_reg)
    joblib.dump(reg, MODELS_DIR / "ridge_regressor.pkl")
    
    # 最优阈值
    y_proba = clf.predict_proba(X_test)[:, 1]
    best_f1, best_threshold = 0, 0.5
    for thresh in np.arange(0.2, 0.8, 0.05):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test_cls, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, thresh
    
    print(f"\n最优阈值 (F1={best_f1:.4f}): {best_threshold:.2f}")
    
    config = {"threshold": float(best_threshold), "model_type": "linear"}
    with open(MODELS_DIR / "model_config.json", "w") as f:
        json.dump(config, f)
    
    print("\n" + "="*50)
    print("✓ 线性模型训练完成！")


def main():
    print("="*50)
    print("心理风险预测 - 线性模型训练")
    print("="*50)
    df = load_all_data()
    X, y_class, y_reg = preprocess(df)
    train_models(X, y_class, y_reg)


if __name__ == "__main__":
    main()
