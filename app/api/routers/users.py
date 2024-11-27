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
    s = pd.Series({"Question": q.question, "desc": desc})
    return {"output_data": predict(s)}