from fastapi import APIRouter, Depends, HTTPException
from ..models.schemas import Query, Response
from ..services.vector_store import VectorStore
from ..services.embeddings import EmbeddingService
from ..services.llm import LLMService
from ..dependencies import get_vector_store

router = APIRouter()

@router.post("/", response_model=Response)
async def query(
    query: Query,
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_service: EmbeddingService = Depends(),
    llm_service: LLMService = Depends(),
):
    try:
        query_embedding = embedding_service.get_query_embedding(query.text)
        relevant_docs = vector_store.search(query_embedding)
        answer = llm_service.get_response(query.text, relevant_docs)
        return Response(answer=answer, sources=relevant_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
