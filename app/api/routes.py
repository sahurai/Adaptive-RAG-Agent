import os
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from langchain_core.messages import HumanMessage

from app.services.db import ingest_file
from app.services.graph import app_graph

router = APIRouter()


@router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        session_id: str = Form(...)
):
    """
    Endpoint to upload a PDF file.
    1. Saves the file temporarily.
    2. Ingests it into ChromaDB tagged with the session_id.
    3. Cleans up the temporary file.
    """
    temp_dir = "temp_uploads"
    # FIX: Initialize file_path to None to avoid UnboundLocalError in the except block
    file_path: Optional[str] = None

    try:
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

        # Define temporary file path
        file_path = os.path.join(temp_dir, file.filename)

        # Write the uploaded file to disk
        # We use 'wb' (write binary) which is standard for saving uploads
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run the ingestion logic (The Pink Node in the architecture)
        # This splits the PDF and saves it to ChromaDB
        ingest_file(file_path, session_id)

        # Clean up: remove the file after successful indexing
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        return {
            "status": "success",
            "message": "File processed and indexed successfully.",
            "session_id": session_id
        }

    except Exception as e:
        # Cleanup in case of error
        # FIX: Check if file_path was assigned and exists before trying to remove it
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        # Log error to console (optional) and return 500 response
        print(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.post("/chat")
async def chat_endpoint(
        question: str = Form(...),
        session_id: str = Form(...)
):
    """
    Main chat endpoint.
    Runs the LangGraph agent workflow: Router -> Retrieval/Search -> Grading -> Generation.
    """
    try:
        # 1. Create the configuration for memory
        # LangGraph uses 'thread_id' to persist and retrieve state from the Checkpointer
        config = {"configurable": {"thread_id": session_id}}

        # 2. Initialize state
        # We explicitly add the user's question to the 'messages' list.
        # LangGraph will append this to the existing history found by thread_id.
        initial_state = {
            "question": question,
            "session_id": session_id,
            "messages": [HumanMessage(content=question)]
        }

        # 3. Invoke the graph with the config
        # Without 'config', MemorySaver creates a new thread every time.
        result = app_graph.invoke(initial_state, config=config)

        return {
            "answer": result.get("generation"),
            "source": result.get("route"),
            "hallucination_grade": result.get("hallucination_grade")
        }

    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")