import os
import shutil

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

# Initialize FastAPI app
app = FastAPI(
    title="Adaptive RAG Agent API",
    description="API for RAG with Multi-query, Fusion, and Self-Correction",
    version="1.0.0"
)

# Configure CORS (Critical for frontend communication)
# Allowing all origins ("*") for development.
# In production, replace with your frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router defined in app/api/routes.py
app.include_router(router, prefix="/api")

@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "running", "service": "Adaptive RAG Agent", "version": "1.0.0", "docs": "http://127.0.0.1:8000/docs"}

if __name__ == "__main__":
    # Reset all temp files
    if os.path.exists("temp_uploads"):
        shutil.rmtree("temp_uploads")
    os.makedirs("temp_uploads")

    import uvicorn
    # Run the server using Uvicorn
    # reload=True enables auto-restart when code changes (dev mode)
    print("Starting server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)