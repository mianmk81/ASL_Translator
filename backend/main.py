from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
import json
import cv2
import base64
import numpy as np
from hybrid_sign_detector import HybridSignDetector
import time
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="ASL Translator API")

# Initialize sign detector
sign_detector = HybridSignDetector()

# Define paths
BASE_DIR = Path(__file__).resolve().parent  # backend directory
IMAGES_DIR = BASE_DIR.parent / "images"
SIGNS_DIR = IMAGES_DIR / "asl_signs"

logger.info(f"Starting server with paths:")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"IMAGES_DIR: {IMAGES_DIR}")
logger.info(f"SIGNS_DIR: {SIGNS_DIR}")

# Mount static files directory
try:
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
    logger.info("Successfully mounted static files directory")
except Exception as e:
    logger.error(f"Failed to mount static files: {e}")
    raise

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/webcam")
async def webcam_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time sign language detection"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        frame_count = 0
        start_time = time.time()
        is_connected = True
        
        while is_connected:
            try:
                # Add a timeout to the receive operation
                data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                
                if not data.get('frame'):
                    continue
                    
                # Decode base64 frame
                img_bytes = base64.b64decode(data['frame'])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None or frame.size == 0:
                    logger.warning("Received invalid frame")
                    continue
                
                # Process frame with sign detector
                frame_with_landmarks, detected_signs, confidences = sign_detector.process_frame(frame, convert_to_rgb=True)
                
                # Encode frame directly
                _, buffer = cv2.imencode('.jpg', frame_with_landmarks, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Check if still connected before sending
                try:
                    await websocket.send_json({
                        "frame": frame_base64,
                        "detected_signs": detected_signs if detected_signs else []
                    })
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected while sending")
                    is_connected = False
                    break
                
                # Log FPS every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    logger.info(f"Processing FPS: {fps:.2f}")
                    frame_count = 0
                    start_time = time.time()
                
            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout")
                try:
                    pong = await websocket.ping()
                    await asyncio.wait_for(pong, timeout=1.0)
                except:
                    logger.info("WebSocket ping failed, closing connection")
                    is_connected = False
                    break
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected while receiving")
                is_connected = False
                break
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")
        try:
            await websocket.close()
        except:
            pass

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "The requested resource was not found"}
    )

@app.on_event("startup")
def startup():
    """Log when the application starts"""
    logger.info("Application starting up...")
    logger.info(f"Current directory: {Path.cwd()}")
    logger.info(f"Images directory: {IMAGES_DIR}")
    logger.info(f"Signs directory: {SIGNS_DIR}")

@app.get("/")
def root():
    """Root endpoint"""
    logger.debug("Root endpoint called")
    return {
        "status": "API is running",
        "endpoints": {
            "/": "This help message",
            "/signs/{word}": "Get sign language representation for a word",
            "/ws/webcam": "WebSocket endpoint for real-time sign language detection"
        }
    }

@app.get("/signs/{word}")
def get_sign(word: str):
    """Get sign language representation for a word"""
    logger.debug(f"Received request for word: {word}")
    result = []
    
    for letter in word.lower():
        if letter.isalpha():
            image_path = SIGNS_DIR / f"{letter.upper()}_test.jpg"
            logger.debug(f"Looking for image at: {image_path}")
            
            if not image_path.exists():
                logger.error(f"Image not found for letter: {letter}")
                raise HTTPException(
                    status_code=404, 
                    detail=f"Sign image for letter '{letter}' not found"
                )
            
            result.append({
                "letter": letter.upper(),
                "image_url": f"/images/asl_signs/{letter.upper()}_test.jpg",
                "description": f"ASL sign for letter '{letter.upper()}'"
            })
    
    logger.debug(f"Returning result: {result}")
    return result

if __name__ == "__main__":
    import uvicorn
    # Try a different port
    uvicorn.run(app, host="127.0.0.1", port=8080)
