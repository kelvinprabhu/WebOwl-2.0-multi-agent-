# app.py -- for fast api creation

# app.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager
import uuid
from datetime import datetime

# Import your modules
from neo4j import GraphDatabase
from KnowledgeRetriever import KnowledgeRetriever, SearchMode
from OfflineKnowledgeRetriever import OfflineKnowledgeRetriever
from WebOwlMultiAgentRAG import WebOwlMultiAgentRAG
app = FastAPI()
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for dependency injection
retriever = None
web_owl_rag = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    search_mode: str = "HYBRID"
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    confidence_score: float
    sources: List[str]
    navigation_path: List[str]
    timestamp: datetime
    search_mode: str

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    created_at: datetime
    last_updated: datetime

class HealthResponse(BaseModel):
    status: str
    system_stats: Dict[str, Any]
    retriever_type: str

# In-memory conversation storage (use Redis or database in production)
conversations: Dict[str, List[Dict]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    global retriever, web_owl_rag
    
    try:
        # Try Neo4j connection first
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER") 
        neo4j_pass = os.getenv("NEO4J_PASS")
        
        if neo4j_uri and neo4j_user and neo4j_pass:
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
            retriever = KnowledgeRetriever(driver)
            retriever.build_vector_index()
            logger.info("Connected to Neo4j successfully")
        else:
            raise Exception("Neo4j credentials not found")
            
    except Exception as e:
        logger.warning(f"Neo4j connection failed: {e}. Falling back to offline mode.")
        
        # Fallback to offline retriever
        try:
            retriever = OfflineKnowledgeRetriever()
            retriever.load_offline("retriever_offline")
            logger.info("Loaded offline retriever successfully")
        except Exception as offline_error:
            logger.error(f"Failed to load offline retriever: {offline_error}")
            retriever = None
    
    # Initialize Web Owl RAG system
    if retriever:
        groq_api_key = os.getenv("NEW_GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if groq_api_key:
            web_owl_rag = WebOwlMultiAgentRAG(retriever, groq_api_key)
            logger.info("Web Owl RAG system initialized")
        else:
            logger.error("GROQ API key not found")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Web Owl RAG system")

# Initialize FastAPI app
app = FastAPI(
    title="Web Owl RAG API",
    description="Intelligent web navigation and information retrieval system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for getting Web Owl RAG system
async def get_web_owl_rag():
    if web_owl_rag is None:
        raise HTTPException(status_code=503, detail="Web Owl RAG system not available")
    return web_owl_rag

# Dependency for session management
def get_or_create_session(session_id: Optional[str] = None) -> str:
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    return session_id

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "Web Owl RAG API",
        "version": "1.0.0",
        "description": "Intelligent web navigation assistant",
        "endpoints": "/docs for API documentation"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        if web_owl_rag:
            stats = web_owl_rag.get_system_stats()
            retriever_type = "Neo4j" if hasattr(retriever, 'driver') else "Offline"
            
            return HealthResponse(
                status="healthy",
                system_stats=stats,
                retriever_type=retriever_type
            )
        else:
            return HealthResponse(
                status="degraded",
                system_stats={},
                retriever_type="none"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(
    request: QueryRequest,
    owl_rag: WebOwlMultiAgentRAG = Depends(get_web_owl_rag)
):
    """Main query endpoint for Web Owl RAG system"""
    
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        
        # Add conversation context if available
        conversation_context = ""
        if session_id in conversations and conversations[session_id]:
            # Include last few messages for context
            recent_messages = conversations[session_id][-3:]
            conversation_context = "\n".join([
                f"Previous Q: {msg['query']}\nPrevious A: {msg['answer'][:200]}..."
                for msg in recent_messages
            ])
        
        # Enhanced query with context
        enhanced_query = request.query
        if conversation_context:
            enhanced_query = f"Conversation context:\n{conversation_context}\n\nCurrent question: {request.query}"
        
        # Process query through Web Owl RAG
        response = owl_rag.answer_query(enhanced_query, request.search_mode)
        
        # Store in conversation history
        conversation_entry = {
            "query": request.query,
            "answer": response.get("structured_response", "No response generated"),
            "timestamp": datetime.now(),
            "confidence_score": response.get("confidence_indicators", {}).get("information_completeness", 0.0),
            "sources": response.get("actionable_next_steps", [])
        }
        conversations[session_id].append(conversation_entry)
        
        # Format response
        return QueryResponse(
            session_id=session_id,
            query=request.query,
            answer=response.get("structured_response", "No response generated"),
            confidence_score=response.get("confidence_indicators", {}).get("information_completeness", 0.0),
            sources=response.get("actionable_next_steps", []),
            navigation_path=[],
            timestamp=datetime.now(),
            search_mode=request.search_mode
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = conversations[session_id]
    
    return ConversationHistory(
        session_id=session_id,
        messages=messages,
        created_at=messages[0]["timestamp"] if messages else datetime.now(),
        last_updated=messages[-1]["timestamp"] if messages else datetime.now()
    )

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    
    if session_id in conversations:
        del conversations[session_id]
        return {"message": f"Conversation {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/search-modes")
async def get_search_modes():
    """Get available search modes"""
    return {
        "search_modes": [mode.value for mode in SearchMode],
        "descriptions": {
            "SEMANTIC": "Pure semantic search using embeddings",
            "GRAPH_WALK": "Graph-based search following relationships", 
            "HYBRID": "Combines semantic and graph search (recommended)",
            "MULTIMODAL": "Search across different content types with context"
        }
    }

@app.get("/sessions")
async def list_active_sessions():
    """List all active conversation sessions"""
    return {
        "active_sessions": list(conversations.keys()),
        "total_sessions": len(conversations),
        "session_details": {
            session_id: {
                "message_count": len(messages),
                "last_activity": messages[-1]["timestamp"] if messages else None
            }
            for session_id, messages in conversations.items()
        }
    }

# Simple search endpoint for basic usage
@app.get("/search")
async def simple_search(
    q: str,
    mode: str = "HYBRID",
    owl_rag: WebOwlMultiAgentRAG = Depends(get_web_owl_rag)
):
    """Simple search endpoint using query parameters"""
    
    request = QueryRequest(query=q, search_mode=mode)
    return await query_knowledge(request, owl_rag)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)