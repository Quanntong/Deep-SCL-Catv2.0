from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from backend.app.schemas.prediction import PredictionRequest, PredictionResponse, BatchPredictionResponse
from backend.app.services.prediction_service import predict_single_student, process_batch_file

router = APIRouter(prefix="/predict", tags=["prediction"])

@router.post("/single", response_model=PredictionResponse)
async def predict_single(request: Request, data: PredictionRequest):
    """单人风险预测"""
    model = request.app.state.model
    result = predict_single_student(model, data)
    return result

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: Request, file: UploadFile = File(...)):
    """批量文件预测"""
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(400, "仅支持 CSV 或 Excel 文件")
    
    model = request.app.state.model
    content = await file.read()
    
    try:
        result = process_batch_file(model, content, file.filename)
        return result
    except ValueError as e:
        raise HTTPException(422, str(e))
