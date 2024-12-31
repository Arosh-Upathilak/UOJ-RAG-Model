from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import Query, Response
from app.services.vector_store import VectorStore
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.dependencies import get_vector_store

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
