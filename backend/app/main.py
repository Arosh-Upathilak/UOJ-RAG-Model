from backend.app.api import health_check
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import query_routes

app = FastAPI(title="RAG Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(query_routes.router, prefix="/api/v1/query")
app.include_router(health_check.router, prefix="/api/v1/health")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
