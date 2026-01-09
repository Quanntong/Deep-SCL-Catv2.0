import pandas as pd
from io import BytesIO
from typing import Dict, Any

def predict_single_student(model, request) -> Dict[str, Any]:
    """单人预测"""
    # 构建特征字典
    features = {}
    features.update(request.scl90.model_dump())
    features.update(request.epq.model_dump())
    
    df = pd.DataFrame([features])
    
    # 调用模型预测
    result = model.predict(df)
    
    return {
        "is_risk": bool(result["is_risk"].iloc[0]),
        "risk_probability": float(result["risk_probability"].iloc[0]),
        "cluster_id": int(result["cluster_id"].iloc[0]) if "cluster_id" in result.columns else None,
        "shap_values": result.get("shap_values", [{}])[0] if "shap_values" in result.columns else None
    }

def process_batch_file(model, file_content: bytes, filename: str) -> Dict[str, Any]:
    """批量文件预测"""
    # 读取文件
    if filename.endswith('.csv'):
        df = pd.read_csv(BytesIO(file_content))
    else:
        df = pd.read_excel(BytesIO(file_content))
    
    # 提取特征列
    from ml_core.config.feature_config import SCL90_FACTORS, EPQ_FACTORS
    feature_cols = SCL90_FACTORS + EPQ_FACTORS
    
    # 检查必要列
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    
    # 预测
    features_df = df[feature_cols]
    result = model.predict(features_df)
    
    # 合并结果
    output = df.copy()
    output["is_risk"] = result["is_risk"]
    output["risk_probability"] = result["risk_probability"]
    if "cluster_id" in result.columns:
        output["cluster_id"] = result["cluster_id"]
    
    return {
        "results": output.to_dict(orient="records"),
        "total": len(output),
        "risk_count": int(result["is_risk"].sum())
    }
