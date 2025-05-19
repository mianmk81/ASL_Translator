from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="ASL Translator API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Go up to ASL root
IMAGES_DIR = BASE_DIR / "images"
SIGNS_DIR = IMAGES_DIR / "asl_signs"

# Mount static files directory
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "status": "API is running",
        "endpoints": {
            "/": "This help message",
            "/signs/{word}": "Get sign language representation for a word"
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
