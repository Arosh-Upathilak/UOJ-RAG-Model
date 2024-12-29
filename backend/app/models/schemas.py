from pydantic import BaseModel
from typing import List, Optional

class Query(BaseModel):
    text: str
    chat_history: Optional[List[dict]] = []

class Response(BaseModel):
    answer: str
    sources: List[str]
