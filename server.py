import json
import os
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from UIApp import UIController
from datetime import datetime
from cust_logger import logger, set_files_message_color
import shutil
from config import Config
from controllers.vector_db_manager import VectorDBManager

app = FastAPI()

# Set log message color for all logs from this file to 'purple' for easier identification in logs
set_files_message_color('purple')

# Mount static files directory from React frontend build.
# Enables serving CSS, JS, images etc. at /static path.
app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

# Initialize UIController instance to handle application logic and communication
uicontroller = UIController()

@app.get("/")
async def serve_root():
    """
    Serve the root route "/".

    Returns:
    --------
    FileResponse
        Sends the React app's main index.html file to bootstrap the SPA frontend.
    """
    return FileResponse(os.path.join("frontend", "build", "index.html"))

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """
    Serve all other GET requests to enable React Router support for deep links.

    Parameters:
    -----------
    full_path : str
        The requested URI path after the root.

    Returns:
    --------
    FileResponse
        Returns the requested static file if it exists,
        otherwise falls back to sending index.html to let React Router handle routing.
    """
    file_path = os.path.join("frontend", "build", full_path)
    # Serve static asset if exists, else fallback to SPA entrypoint
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join("frontend", "build", "index.html"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional communication with the frontend.

    Maintains the lifecycle of the WebSocket connection and handles incoming messages using UIController.

    Logs all received messages, errors, and connection events with timestamp and conversation UUID.

    Parameters:
    -----------
    websocket : WebSocket
        The active WebSocket connection instance.

    Operations:
    -----------
    - Accepts connection
    - Listens continuously for incoming JSON messages with at least "uuid" and "message" keys
    - On first message (init flag), logs initialization
    - For subsequent messages, forwards content and uuid to UIController to process and respond
    - Handles JSON decode errors and general exceptions with detailed logs
    - Ensures graceful connection closure and logs connection termination
    """
    await websocket.accept()  # Accept incoming WebSocket connection
    user_uuid = None  # Tracks the unique conversation identifier for logging context
    try:
        while True:
            data = await websocket.receive_text()  # Wait for next message from frontend client

            # Log the raw received message with timestamp and conversation UUID (can be None initially)
            logger.info(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "uuid": user_uuid,
                "received": json.loads(data)
            }))

            try:
                payload = json.loads(data)  # Parse JSON payload from received text
                user_uuid = payload.get("uuid")  # Extract conversation UUID
                message = payload.get("message")  # Extract user message content
                init = payload.get("init", False)  # Flag indicating first/init message of conversation

                if init:
                    # Log initialization event on first message of a conversation
                    logger.info(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "uuid": user_uuid,
                        "op": "Initializing ws with client."
                    }))
                else:
                    # For non-init messages with content, invoke async processing logic in UIController
                    if message:
                        await uicontroller.invoke_our_graph(websocket, message, user_uuid)
            except json.JSONDecodeError as e:
                # Log JSON parsing errors with context for easier debugging
                logger.error(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "uuid": user_uuid,
                    "op": f"JSON encoding error - {e}"
                }))
    except Exception as e:
        # Log any unexpected exceptions for operational monitoring and incident response
        logger.error(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "uuid": user_uuid,
            "op": f"Error: {e}"
        }))
    finally:
        # On exit/close, log the connection termination event if UUID is known
        if user_uuid:
            logger.info(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "uuid": user_uuid,
                "op": "Closing connection."
            }))
        try:
            # Attempt to close the WebSocket connection gracefully
            await websocket.close()
        except RuntimeError as e:
            # Catch specific error when connection was already closed by client
            logger.error(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "uuid": user_uuid,
                "op": f"WebSocket close error: {e}"
            }))

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = os.path.join(Config.DATA_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print("Vectorstore not found or empty. Creating it first.")
    vector_db = VectorDBManager()
    document_tuples = vector_db.load_documents_from_directory()
    print(f"Loaded {len(document_tuples)} files from data directory.")
    vector_db.create_vectorstore_incrementally(document_tuples)
    return {"message": f"File '{file.filename}' uploaded successfully!"}

# Entry point to run the FastAPI app when executing this file directly
# Uses uvicorn ASGI server with host 0.0.0.0 and port 8000, minimizing uvicorn default verbosity
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
