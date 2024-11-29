from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from app.api.services.users import predict
class QueryData(BaseModel):
    question: str
    desc: str

router = APIRouter(prefix="/users", tags=["user"])


@router.post("/ask")
async def ask(q: QueryData):
    desc = q.desc.split('}')[-1].strip()
    s = q.question + '\n' + desc
    
    predictions, probabilities = predict(s)
    
    predictions_list = predictions.tolist()
    probabilities_list = probabilities.tolist()
    
    return {"output_data": predictions_list, "probabilities": probabilities_list}