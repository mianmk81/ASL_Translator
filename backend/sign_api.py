from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to ASL sign images
SIGNS_DIR = Path(__file__).parent.parent / "images" / "asl_signs"
logger.debug(f"SIGNS_DIR absolute path: {SIGNS_DIR.absolute()}")
logger.debug(f"SIGNS_DIR exists: {SIGNS_DIR.exists()}")
logger.debug(f"SIGNS_DIR is directory: {SIGNS_DIR.is_dir()}")

# Create directory if it doesn't exist
SIGNS_DIR.mkdir(parents=True, exist_ok=True)

# Mount the images directory
IMAGES_DIR = Path(__file__).parent.parent / "images"
logger.debug(f"IMAGES_DIR absolute path: {IMAGES_DIR.absolute()}")
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

@app.get("/")
async def root():
    """Root endpoint"""
    logger.debug("Root endpoint called")
    return {
        "status": "API is running",
        "endpoints": {
            "/": "This help message",
            "/signs/{word}": "Get sign language representation for a word"
        }
    }

@app.get("/signs/{word}")
async def get_sign(word: str):
    """Get sign language representation for a word"""
    logger.debug(f"Received request for word: {word}")
    result = []
    
    # List all files in the directory
    logger.debug("Files in SIGNS_DIR:")
    for f in SIGNS_DIR.glob("*"):
        logger.debug(f"  - {f.name}")
    
    for letter in word.lower():
        if letter.isalpha():
            image_path = SIGNS_DIR / f"{letter.upper()}_test.jpg"
            logger.debug(f"Looking for image at: {image_path.absolute()}")
            logger.debug(f"File exists: {image_path.exists()}")
            logger.debug(f"Is file: {image_path.is_file() if image_path.exists() else 'N/A'}")
            
            if not image_path.exists():
                logger.error(f"Image not found for letter: {letter}")
                raise HTTPException(status_code=404, detail=f"Sign for letter '{letter}' not found")
            
            url = f"/images/asl_signs/{letter.upper()}_test.jpg"
            logger.debug(f"Generated URL: {url}")
            result.append({
                "word": letter,
                "image_url": url,
                "description": f"ASL sign for letter '{letter.upper()}'"
            })
    
    logger.debug(f"Returning result: {result}")
    return result

# Run with: uvicorn sign_api:app --reload --port 8000

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")  # Debug print
    uvicorn.run(app, host="127.0.0.1", port=8000)
