from pydantic import BaseModel
from typing import Optional, Dict, List

class SCL90Factors(BaseModel):
    躯体化: float
    强迫症状: float
    人际关系敏感: float
    抑郁: float
    焦虑: float
    敌对: float
    恐怖: float
    偏执: float
    精神病性: float
    其他: float

class EPQFactors(BaseModel):
    内外向E: float
    神经质N: float
    精神质P: float
    掩饰性L: float

class StudentInfo(BaseModel):
    学号: Optional[str] = None
    姓名: Optional[str] = None
    班级: Optional[str] = None

class PredictionRequest(BaseModel):
    info: Optional[StudentInfo] = None
    scl90: SCL90Factors
    epq: EPQFactors

class PredictionResponse(BaseModel):
    is_risk: bool
    risk_probability: float
    fail_count_pred: Optional[float] = None
    shap_values: Optional[Dict[str, float]] = None
    cluster_id: Optional[int] = None

class BatchPredictionResponse(BaseModel):
    results: List[Dict]
    total: int
    risk_count: int
