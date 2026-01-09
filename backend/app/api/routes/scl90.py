from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter(prefix="/scl90", tags=["scl90"])

# SCL-90 标准题号映射 (题号从1开始)
SCL90_ITEM_MAPPING = {
    "躯体化": [1, 4, 12, 27, 40, 42, 48, 49, 52, 53, 56, 58],
    "强迫症状": [3, 9, 10, 28, 38, 45, 46, 51, 55, 65],
    "人际关系敏感": [6, 21, 34, 36, 37, 41, 61, 69, 73],
    "抑郁": [5, 14, 15, 20, 22, 26, 29, 30, 31, 32, 54, 71, 79],
    "焦虑": [2, 17, 23, 33, 39, 57, 72, 78, 80, 86],
    "敌对": [11, 24, 63, 67, 74, 81],
    "恐怖": [13, 25, 47, 50, 70, 75, 82],
    "偏执": [8, 18, 43, 68, 76, 83],
    "精神病性": [7, 16, 35, 62, 77, 84, 85, 87, 88, 90],
    "其他": [19, 44, 59, 60, 64, 66, 89],
}

class SCL90RawScores(BaseModel):
    scores: List[float]  # 90个题目的原始评分 (1-5分)

class SCL90FactorsResponse(BaseModel):
    factors: Dict[str, float]
    total_score: float
    positive_items: int  # 阳性项目数 (>=2分的题目数)
    positive_avg: float  # 阳性症状均分

@router.post("/calculate", response_model=SCL90FactorsResponse)
async def calculate_factors(data: SCL90RawScores):
    """根据90道题原始评分计算10个因子分"""
    scores = data.scores
    
    if len(scores) != 90:
        from fastapi import HTTPException
        raise HTTPException(422, f"需要90个评分，收到{len(scores)}个")
    
    # 计算各因子分 (因子分 = 该因子所有题目分数之和 / 题目数)
    factors = {}
    for factor_name, items in SCL90_ITEM_MAPPING.items():
        item_scores = [scores[i - 1] for i in items]  # 题号从1开始，索引从0开始
        factors[factor_name] = round(sum(item_scores) / len(items), 2)
    
    # 总分
    total_score = sum(scores)
    
    # 阳性项目数 (评分>=2的题目)
    positive_items = sum(1 for s in scores if s >= 2)
    
    # 阳性症状均分
    positive_scores = [s for s in scores if s >= 2]
    positive_avg = round(sum(positive_scores) / len(positive_scores), 2) if positive_scores else 0
    
    return {
        "factors": factors,
        "total_score": total_score,
        "positive_items": positive_items,
        "positive_avg": positive_avg
    }
