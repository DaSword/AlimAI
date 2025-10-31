"""
LangGraph Server entry point and configuration.

LangGraph Server provides native API endpoints for the RAG workflow with:
- Streaming support via Server-Sent Events
- Thread-based conversation management
- Automatic state persistence
- Built-in error handling

The server is configured via langgraph.json and typically run with:
    langgraph dev --port 8123

For production deployment:
    langgraph up --port 8123

API Endpoints (automatically provided by LangGraph):
- POST   /runs/stream              - Stream workflow execution
- POST   /threads                  - Create new conversation thread
- GET    /threads/{thread_id}      - Get thread details
- GET    /threads/{thread_id}/history - Get conversation history
- POST   /threads/{thread_id}/runs - Run workflow in existing thread
- GET    /runs/{run_id}            - Get run status
- POST   /assistants/{assistant_id}/invoke - Invoke workflow synchronously
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path

# Import the RAG graph
from backend.rag.rag_graph import graph, create_rag_graph
from backend.core.config import Config


# Initialize configuration
config = Config()


def get_graph():
    """
    Get the RAG workflow graph instance.
    
    This function is called by LangGraph Server to load the graph.
    The graph is registered in langgraph.json as:
        "graphs": {"rag_assistant": "./backend/api/server.py:get_graph"}
    
    Returns:
        Compiled StateGraph instance
    """
    return graph


def create_app_config() -> Dict[str, Any]:
    """
    Create application configuration for LangGraph Server.
    
    Returns:
        Configuration dictionary
    """
    return {
        "title": "Islamic Chatbot RAG API",
        "description": "RAG-powered Islamic knowledge assistant with source citations",
        "version": "1.0.0",
        "graphs": {
            "rag_assistant": graph,
        },
        "config": {
            "qdrant_url": config.QDRANT_URL,
            "qdrant_collection": config.QDRANT_COLLECTION,
            "embedding_backend": config.EMBEDDING_BACKEND,
            "llm_backend": config.LLM_BACKEND,
        }
    }


# ============================================================================
# Custom Endpoints (Optional - for admin operations)
# ============================================================================
# 
# LangGraph Server can be extended with custom FastAPI routes if needed.
# For now, admin operations are provided as standalone functions in backend/api/admin/
# 
# To add custom endpoints, you would create a FastAPI app and mount it:
# 
# from fastapi import FastAPI, HTTPException
# from backend.api.admin.ingestion_api import ingest_file_handler
# from backend.api.admin.collection_api import list_collections
# from backend.api.admin.models_api import get_health_status
#
# admin_app = FastAPI(title="Admin API")
#
# @admin_app.post("/admin/ingest")
# async def ingest_endpoint(
#     file_path: str,
#     source_type: str,
#     batch_size: int = 100,
# ):
#     result = await ingest_file_handler(file_path, source_type, batch_size=batch_size)
#     if not result["success"]:
#         raise HTTPException(status_code=400, detail=result["error"])
#     return result
#
# @admin_app.get("/admin/collections")
# async def collections_endpoint():
#     result = await list_collections()
#     if not result["success"]:
#         raise HTTPException(status_code=500, detail=result["error"])
#     return result
#
# @admin_app.get("/admin/health")
# async def health_endpoint():
#     result = await get_health_status()
#     return result


# ============================================================================
# Development Server
# ============================================================================

def run_dev_server(port: int = 8123, host: str = "0.0.0.0"):
    """
    Run LangGraph development server.
    
    This is a wrapper around the LangGraph CLI for convenience.
    Normally you would run: langgraph dev --port 8123
    
    Args:
        port: Port to run server on
        host: Host to bind to
    """
    import subprocess
    
    print(f"🚀 Starting LangGraph Server on {host}:{port}")
    print(f"📊 Graph: rag_assistant")
    print(f"📁 Config: langgraph.json")
    print(f"🔧 Backend: {config.EMBEDDING_BACKEND} (embeddings), {config.LLM_BACKEND} (LLM)")
    print()
    
    try:
        subprocess.run(
            ["langgraph", "dev", "--port", str(port), "--host", host],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        print("\nMake sure langgraph-cli is installed:")
        print("  uv pip install langgraph-cli")
    except FileNotFoundError:
        print("❌ langgraph command not found")
        print("\nInstall langgraph-cli:")
        print("  uv pip install langgraph-cli")


# ============================================================================
# CLI Usage
# ============================================================================

if __name__ == "__main__":
    """
    Run the development server from command line.
    
    Usage:
        python backend/api/server.py [--port PORT] [--host HOST]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Islamic Chatbot RAG API Server")
    parser.add_argument("--port", type=int, default=8123, help="Port to run on (default: 8123)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Islamic Chatbot RAG API Server")
    print("="*60)
    print()
    
    # Show configuration
    print("Configuration:")
    print(f"  Qdrant URL: {config.QDRANT_URL}")
    print(f"  Collection: {config.QDRANT_COLLECTION}")
    print(f"  Embedding Backend: {config.EMBEDDING_BACKEND}")
    print(f"  LLM Backend: {config.LLM_BACKEND}")
    print()
    
    print("API Endpoints:")
    print(f"  POST   http://{args.host}:{args.port}/runs/stream")
    print(f"  POST   http://{args.host}:{args.port}/threads")
    print(f"  GET    http://{args.host}:{args.port}/threads/{{thread_id}}")
    print()
    
    # Run server
    run_dev_server(port=args.port, host=args.host)

