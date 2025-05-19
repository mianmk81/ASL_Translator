import tensorflow as tf
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLDataLoader:
    def __init__(self, data_dir: str, img_height: int = 224, img_width: int = 224):
        """
        Initialize the ASL data loader
        Args:
            data_dir: Path to the data directory containing train and test folders
            img_height: Height to resize images to
            img_width: Width to resize images to
        """
        self.data_dir = Path(data_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = 3
        
        # Validate directory structure
        self.train_dir = self.data_dir / 'train'
        self.test_dir = self.data_dir / 'test'
        
        if not self.train_dir.exists() or not self.test_dir.exists():
            raise ValueError(f"Data directory structure invalid. Expected train and test directories in {data_dir}")
        
        # Get class names from training directory
        self.class_names = sorted([item.name for item in self.train_dir.iterdir() if item.is_dir()])
        self.num_classes = len(self.class_names)
        logger.info(f"Found {self.num_classes} classes: {self.class_names}")

    def _create_train_dataset(self, batch_size: int = 32):
        """Create training dataset with augmentation"""
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size
        )
        
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ])
        
        # Configure dataset for performance
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    def _create_validation_dataset(self, batch_size: int = 32):
        """Create validation dataset"""
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size
        )
        return val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    def _create_test_dataset(self, batch_size: int = 32):
        """Create test dataset"""
        # For test data, we need to manually create the labels from filenames
        test_images = []
        test_labels = []
        
        for image_path in self.test_dir.glob("*.jpg"):
            # Read and preprocess image
            img = tf.keras.preprocessing.image.load_img(
                image_path, 
                target_size=(self.img_height, self.img_width)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            test_images.append(img_array)
            
            # Extract label from filename (e.g., 'A_test.jpg' -> 'A')
            label = image_path.stem.split('_')[0].lower()
            if label == 'nothing' or label == 'space' or label == 'del':
                test_labels.append(self.class_names.index(label))
            else:
                test_labels.append(self.class_names.index(label.upper()))
        
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_ds = test_ds.batch(batch_size)
        return test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    def get_datasets(self, batch_size: int = 32):
        """
        Get train, validation, and test datasets
        Args:
            batch_size: Batch size for training
        Returns:
            train_ds, val_ds, test_ds: TensorFlow datasets
        """
        logger.info("Creating training dataset...")
        train_ds = self._create_train_dataset(batch_size)
        
        logger.info("Creating validation dataset...")
        val_ds = self._create_validation_dataset(batch_size)
        
        logger.info("Creating test dataset...")
        test_ds = self._create_test_dataset(batch_size)
        
        # Normalize the data
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
        
        return train_ds, val_ds, test_ds

    def get_class_info(self):
        """Get information about the classes"""
        return {
            'class_names': self.class_names,
            'num_classes': self.num_classes
        }

if __name__ == "__main__":
    # Test the data loader
    data_dir = Path("../data/asl_alphabet")
    loader = ASLDataLoader(data_dir)
    
    # Get datasets
    train_ds, val_ds, test_ds = loader.get_datasets()
    
    # Print some information
    logger.info("Dataset information:")
    for images, labels in train_ds.take(1):
        logger.info(f"Image shape: {images.shape}")
        logger.info(f"Label shape: {labels.shape}")
        logger.info(f"Labels: {labels.numpy()}")
