from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more info
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR = Path(__file__).parent.parent  # Go up one level to main ASL directory
SIGNS_DIR = BASE_DIR / "images" / "asl_signs"

# Mount static files
app.mount("/images", StaticFiles(directory=str(BASE_DIR / "images")), name="images")

@app.get("/")
async def root():
    logger.debug("Root endpoint hit!")
    return {"message": "Sign Language Translator API"}

@app.get("/signs/{sign}")
async def get_sign_image(sign: str):
    """Get the image representation of a sign"""
    logger.debug(f"Received request for sign: {sign}")
    logger.debug(f"SIGNS_DIR: {SIGNS_DIR}")
    
    try:
        # Create signs directory if it doesn't exist
        SIGNS_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created/verified signs directory at {SIGNS_DIR}")
        
        # Check if sign image exists
        sign_path = SIGNS_DIR / f"{sign.lower()}.png"
        logger.debug(f"Looking for sign image at: {sign_path}")
        
        if not sign_path.exists():
            logger.warning(f"Sign image not found at: {sign_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Sign image for '{sign}' not found"
            )
        
        logger.debug(f"Found sign image at: {sign_path}")
        return {
            "word": sign,
            "image_url": f"/images/asl_signs/{sign.lower()}.png",
            "description": f"ASL sign for '{sign.upper()}'"
        }
    except Exception as e:
        logger.error(f"Error getting sign image: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
