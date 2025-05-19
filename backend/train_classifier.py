import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
import mediapipe as mp
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignClassifierTrainer:
    def __init__(self, data_dir: str):
        """
        Initialize the trainer
        Args:
            data_dir: Directory containing the training data
        """
        self.data_dir = Path(data_dir)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
        # Get class names from directory
        self.class_names = sorted([d.name for d in (self.data_dir / 'train').iterdir() if d.is_dir()])
        self.num_classes = len(self.class_names)
        logger.info(f"Found {self.num_classes} classes: {self.class_names}")

    def _extract_features(self, image_path: Path) -> np.ndarray:
        """Extract hand landmarks and derived features from an image"""
        # Read and process image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect hand landmarks
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract features
        landmarks = results.multi_hand_landmarks[0]
        features = []
        
        # Basic landmark coordinates
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # Add hand angle
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        angle = np.arctan2(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
        features.append(angle)
        
        # Add finger states
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [2, 6, 10, 14, 18]
        for tip, pip in zip(finger_tips, finger_pips):
            extended = float(landmarks.landmark[tip].y < landmarks.landmark[pip].y)
            features.append(extended)
        
        return np.array(features)

    def prepare_dataset(self):
        """Prepare the dataset by extracting features from all images"""
        features = []
        labels = []
        
        logger.info("Extracting features from training images...")
        total_processed = 0
        total_failed = 0
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / 'train' / class_name
            images = list(class_dir.glob('*.jpg'))
            logger.info(f"Processing class {class_name} ({len(images)} images)")
            
            # Process each image in the class directory
            class_processed = 0
            for image_path in tqdm(images, desc=f"Class {class_name}"):
                try:
                    image_features = self._extract_features(image_path)
                    if image_features is not None:
                        features.append(image_features)
                        labels.append(class_idx)
                        class_processed += 1
                        total_processed += 1
                    else:
                        total_failed += 1
                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    total_failed += 1
            
            logger.info(f"Successfully processed {class_processed}/{len(images)} images for class {class_name}")
        
        logger.info(f"\nDataset preparation complete:")
        logger.info(f"Total images processed successfully: {total_processed}")
        logger.info(f"Total images failed: {total_failed}")
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        logger.info(f"Final dataset split:")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Feature vector size: {X_train.shape[1]}")
        
        return X_train, X_val, y_train, y_val

    def create_model(self, input_shape):
        """Create the classifier model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, epochs=50, batch_size=32):
        """Train the classifier"""
        # Prepare dataset
        logger.info("\nPreparing dataset...")
        X_train, X_val, y_train, y_val = self.prepare_dataset()
        
        # Create and compile model
        logger.info("\nCreating model...")
        model = self.create_model(X_train.shape[1])
        model.summary()
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/sign_classifier.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            ),
            tf.keras.callbacks.ProgbarLogger(count_mode='steps')
        ]
        
        # Train the model
        logger.info("\nStarting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model
        model.save('models/sign_classifier_final.h5')
        logger.info("\nTraining completed. Model saved.")
        
        # Print final metrics
        logger.info("\nFinal metrics:")
        logger.info(f"Training accuracy: {history.history['accuracy'][-1]:.4f}")
        logger.info(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        logger.info(f"Training loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"Validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return history

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    # Initialize trainer with relative path
    trainer = SignClassifierTrainer('../data/asl_alphabet')
    
    # Train the model
    history = trainer.train()
    
    # Print final metrics
    logger.info("Final training metrics:")
    for metric, value in history.history.items():
        logger.info(f"{metric}: {value[-1]:.4f}")
