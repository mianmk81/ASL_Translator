import os
import shutil
import requests
import zipfile
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.datasets = {
            "asl_alphabet": {
                "local_path": self.base_dir / "asl_alphabet",
                "classes": None  # Will be populated after download
            },
            "wlasl": {
                "url": "https://github.com/dxli94/WLASL/raw/master/data/WLASL_v0.3.json",
                "local_path": self.base_dir / "wlasl",
                "classes": None
            },
            "custom": {
                "local_path": self.base_dir / "custom",
                "classes": set()
            }
        }
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for dataset in self.datasets.values():
            dataset["local_path"].mkdir(parents=True, exist_ok=True)

    async def setup_datasets(self):
        """Initialize and download all datasets"""
        try:
            # Skip Kaggle dataset for now
            # await self.download_asl_alphabet()
            await self.download_wlasl()
            logger.info("All datasets downloaded successfully")
        except Exception as e:
            logger.error(f"Error setting up datasets: {e}")
            raise

    async def download_wlasl(self):
        """Download WLASL dataset"""
        dataset_path = self.datasets["wlasl"]["local_path"]
        
        if not (dataset_path / "downloaded").exists():
            try:
                logger.info("Downloading WLASL dataset...")
                response = requests.get(self.datasets["wlasl"]["url"])
                response.raise_for_status()
                
                with open(dataset_path / "wlasl_v0.3.json", "wb") as f:
                    f.write(response.content)
                
                # Mark as downloaded
                (dataset_path / "downloaded").touch()
                logger.info("WLASL dataset downloaded successfully")
            except Exception as e:
                logger.error(f"Error downloading WLASL dataset: {e}")
                raise

    def add_custom_data(self, image_path: str, label: str) -> bool:
        """Add a custom image to the dataset"""
        try:
            # Create directory for this class if it doesn't exist
            class_dir = self.datasets["custom"]["local_path"] / label
            class_dir.mkdir(exist_ok=True)
            
            # Copy image to class directory
            image_name = f"{len(list(class_dir.glob('*.*'))) + 1}.jpg"
            shutil.copy2(image_path, class_dir / image_name)
            
            # Update classes
            self.datasets["custom"]["classes"].add(label)
            return True
        except Exception as e:
            logger.error(f"Error adding custom data: {e}")
            return False

    def prepare_dataset(
        self,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (224, 224),
        validation_split: float = 0.2,
        use_datasets: List[str] = ["custom"]  # Default to only custom dataset
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
        """Prepare dataset for training"""
        
        images = []
        labels = []
        label_mapping = {}
        current_label = 0

        # Collect all data
        for dataset_name in use_datasets:
            if dataset_name not in self.datasets:
                logger.warning(f"Dataset {dataset_name} not found, skipping...")
                continue
                
            dataset = self.datasets[dataset_name]
            dataset_path = dataset["local_path"]

            for class_dir in dataset_path.glob("*"):
                if not class_dir.is_dir():
                    continue

                # Add to label mapping
                if class_dir.name not in label_mapping:
                    label_mapping[class_dir.name] = current_label
                    current_label += 1

                # Collect images
                for img_path in class_dir.glob("*.*"):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(image_size)
                        img_array = np.array(img) / 255.0
                        images.append(img_array)
                        labels.append(label_mapping[class_dir.name])
                    except Exception as e:
                        logger.warning(f"Error processing image {img_path}: {e}")

        if not images:
            raise ValueError("No images found in the specified datasets")

        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Shuffle data
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = images[indices]
        labels = labels[indices]

        # Split into train and validation
        split_idx = int(len(images) * (1 - validation_split))
        train_images, val_images = images[:split_idx], images[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]

        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

        # Configure datasets
        train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, label_mapping

    def get_dataset_stats(self) -> Dict:
        """Get statistics about the datasets"""
        stats = {}
        
        for dataset_name, dataset in self.datasets.items():
            dataset_path = dataset["local_path"]
            stats[dataset_name] = {
                "total_images": 0,
                "classes": 0,
                "images_per_class": {}
            }
            
            if dataset_path.exists():
                for class_dir in dataset_path.glob("*"):
                    if class_dir.is_dir():
                        image_count = len(list(class_dir.glob("*.*")))
                        stats[dataset_name]["images_per_class"][class_dir.name] = image_count
                        stats[dataset_name]["total_images"] += image_count
                
                stats[dataset_name]["classes"] = len(stats[dataset_name]["images_per_class"])
        
        return stats

    def apply_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """Apply data augmentation to an image"""
        # Random rotation
        image = tf.image.random_rotation(image, 0.2)
        
        # Random brightness
        image = tf.image.random_brightness(image, 0.2)
        
        # Random contrast
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random flip
        image = tf.image.random_flip_left_right(image)
        
        # Ensure values are in [0, 1]
        image = tf.clip_by_value(image, 0, 1)
        
        return image
