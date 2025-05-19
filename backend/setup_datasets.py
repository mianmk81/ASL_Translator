import os
import json
import shutil
import requests
from pathlib import Path
import logging
import zipfile
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_kaggle_credentials():
    """Set up Kaggle credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Look for kaggle.json in Downloads folder
    downloads_dir = Path.home() / 'Downloads'
    kaggle_json = downloads_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        logger.error("""
        Could not find kaggle.json in Downloads folder!
        Please:
        1. Go to kaggle.com and sign in
        2. Click on your profile picture → Account
        3. Scroll down to API section
        4. Click 'Create New API Token'
        5. Move the downloaded kaggle.json to your Downloads folder
        6. Run this script again
        """)
        sys.exit(1)
    
    # Copy to .kaggle directory
    target_path = kaggle_dir / 'kaggle.json'
    shutil.copy2(kaggle_json, target_path)
    
    # Set correct permissions
    os.chmod(target_path, 0o600)
    logger.info("Successfully set up Kaggle credentials!")

def download_asl_alphabet():
    """Download ASL Alphabet dataset from Kaggle"""
    try:
        import kaggle
        
        dataset_path = Path('data/asl_alphabet')
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        if not (dataset_path / 'downloaded').exists():
            logger.info("Downloading ASL Alphabet dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                'grassknoted/asl-alphabet',
                path=dataset_path,
                unzip=True
            )
            (dataset_path / 'downloaded').touch()
            logger.info("ASL Alphabet dataset downloaded successfully!")
        else:
            logger.info("ASL Alphabet dataset already exists!")
            
    except Exception as e:
        logger.error(f"Error downloading ASL Alphabet dataset: {e}")
        raise

def download_wlasl():
    """Download WLASL dataset"""
    try:
        dataset_path = Path('data/wlasl')
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        if not (dataset_path / 'downloaded').exists():
            logger.info("Downloading WLASL dataset...")
            
            # Download dataset JSON
            json_url = "https://github.com/dxli94/WLASL/raw/master/data/WLASL_v0.3.json"
            response = requests.get(json_url)
            response.raise_for_status()
            
            with open(dataset_path / 'wlasl_v0.3.json', 'wb') as f:
                f.write(response.content)
            
            # Parse JSON to get video URLs
            data = json.loads(response.content)
            
            # Create videos directory
            videos_dir = dataset_path / 'videos'
            videos_dir.mkdir(exist_ok=True)
            
            # Download first 100 videos as sample
            for entry in data[:100]:  # Limit to 100 videos initially
                word = entry['gloss']
                instances = entry['instances']
                
                word_dir = videos_dir / word
                word_dir.mkdir(exist_ok=True)
                
                for instance in instances:
                    try:
                        video_url = instance['url']
                        video_id = instance['video_id']
                        
                        # Download video
                        response = requests.get(video_url, stream=True)
                        if response.status_code == 200:
                            video_path = word_dir / f"{video_id}.mp4"
                            with open(video_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                    except Exception as e:
                        logger.warning(f"Failed to download video {video_id}: {e}")
                        continue
            
            (dataset_path / 'downloaded').touch()
            logger.info("WLASL dataset downloaded successfully!")
        else:
            logger.info("WLASL dataset already exists!")
            
    except Exception as e:
        logger.error(f"Error downloading WLASL dataset: {e}")
        raise

def main():
    """Main setup function"""
    try:
        logger.info("Setting up datasets...")
        
        # Setup Kaggle credentials
        setup_kaggle_credentials()
        
        # Download datasets
        download_asl_alphabet()
        download_wlasl()
        
        logger.info("All datasets downloaded successfully!")
        
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
