"""
FastAPI main app for MathRAG system. 
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI instance
app = FastAPI(
    title="MathRAG API",
    description="A RAG system for mathematical problem solving and education",
    version="0.1.0",
    docs_url="/docs",#Swagger UI
    redoc_url="/redoc"#Alternative API docs
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"], #Steamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Root endpoint for simple health check.
    """
    return {"message": "MathRAG API is running."}
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy",
            "service": "MathRAG API",
            "version": "0.1.0"
    }
if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    