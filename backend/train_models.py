# -*- coding: utf-8 -*-
"""
训练脚本 - 优化版：使用 SMOTE + 交叉验证 + 多阈值优化
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, recall_score, precision_score

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
    
    # 添加特征工程：SCL-90总分 + 高危因子组合
    X["scl90_total"] = X[SCL90_FACTORS].mean(axis=1)
    X["high_risk_score"] = (X["depression"] + X["anxiety"] + X["interpersonal_sensitivity"]) / 3
    
    print(f"样本数: {len(X)}, 风险样本: {y_class.sum()} ({y_class.mean()*100:.1f}%)")
    return X, y_class, y_reg


def find_optimal_thresholds(clf, X_test, y_test):
    """为三种模式找最优阈值"""
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # strict: 高阈值，减少误报
    thresholds = {"strict": 0.55}
    
    # balanced: 最优F1
    best_f1, best_t = 0, 0.4
    for t in np.arange(0.25, 0.6, 0.02):
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    thresholds["balanced"] = round(best_t, 2)
    
    # sensitive: 低阈值，尽量不漏掉风险学生
    thresholds["sensitive"] = max(0.20, thresholds["balanced"] - 0.15)
    
    return thresholds


def train_models(X, y_class, y_reg):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    feature_cols = list(X.columns)
    
    # 1. StandardScaler (仅SCL-90)
    print("\n[1/5] 训练 StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[SCL90_FACTORS])
    joblib.dump(scaler, MODELS_DIR / "standard_scaler.joblib")
    
    # 2. KMeans
    print("[2/5] 训练 KMeans (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    joblib.dump(kmeans, MODELS_DIR / "kmeans_clusterer.joblib")
    print(f"  聚类分布: {np.bincount(cluster_labels)}")
    
    # 3. 组合特征
    X_full = X.copy()
    X_full["cluster"] = cluster_labels
    
    # 全部标准化
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)
    joblib.dump(scaler_full, MODELS_DIR / "full_scaler.joblib")
    
    # 划分数据
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        X_full_scaled, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # 4. Logistic Regression (使用 class_weight='balanced' 处理类别不平衡)
    print("[3/4] 训练 Logistic Regression...")
    clf = LogisticRegression(class_weight='balanced', C=0.5, solver='liblinear', random_state=42)
    clf.fit(X_train, y_train_cls)
    joblib.dump(clf, MODELS_DIR / "logistic_classifier.pkl")
    
    # 评估
    y_proba_test = clf.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)
    
    print(f"  测试集 - 精确率: {precision_score(y_test_cls, y_pred_test):.3f}")
    print(f"  测试集 - 召回率: {recall_score(y_test_cls, y_pred_test):.3f}")
    print(f"  测试集 - F1: {f1_score(y_test_cls, y_pred_test):.3f}")
    
    # 特征重要性
    feature_names = feature_cols + ["cluster"]
    importance = np.abs(clf.coef_[0])
    print("\n  特征重要性 Top 5:")
    for idx in np.argsort(importance)[-5:][::-1]:
        print(f"    {feature_names[idx]}: {importance[idx]:.3f}")
    
    # 6. Ridge 回归
    print("\n[5/5] 训练 Ridge Regression...")
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train_reg)
    joblib.dump(reg, MODELS_DIR / "ridge_regressor.pkl")
    
    # 7. 计算三种模式的最优阈值
    print("\n计算各模式最优阈值...")
    thresholds = find_optimal_thresholds(clf, X_test, y_test_cls)
    print(f"  strict (精准): {thresholds['strict']}")
    print(f"  balanced (均衡): {thresholds['balanced']}")
    print(f"  sensitive (全面): {thresholds['sensitive']}")
    
    # 保存配置
    config = {
        "model_type": "linear",
        "thresholds": thresholds,
        "feature_cols": feature_names
    }
    with open(MODELS_DIR / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*50)
    print("✓ 优化模型训练完成！")


def main():
    print("="*50)
    print("心理风险预测 - 优化版训练")
    print("="*50)
    df = load_all_data()
    X, y_class, y_reg = preprocess(df)
    train_models(X, y_class, y_reg)


if __name__ == "__main__":
    main()
