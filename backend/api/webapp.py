"""
Custom FastAPI app for LangGraph Server.

This app adds admin endpoints to the LangGraph Server for:
- Health checks
- Model management  
- Collection management
- Data ingestion

These routes are exposed alongside the default LangGraph routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.admin.routes import router as admin_router

# Create FastAPI app
app = FastAPI(
    title="Alim AI Admin API",
    description="Admin endpoints for Islamic Chatbot RAG system",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Vite dev server
        "http://localhost:3000",      # Common dev port
        "https://smith.langchain.com" # LangSmith Studio
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include admin router
app.include_router(admin_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Alim AI Admin API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# Health check at root level (for quick checks)
@app.get("/health")
async def health():
    """Quick health check."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8123)

