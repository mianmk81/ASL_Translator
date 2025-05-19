import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
import logging
from .dataset import SignLanguageDataset

logger = logging.getLogger(__name__)

class SignLanguageModel:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "sign_language_model.h5"
        self.model = None
        self.dataset = SignLanguageDataset()
    
    def create_model(self, num_classes):
        """
        Create a CNN model for sign language recognition
        """
        model = models.Sequential([
            # Input layer for hand landmarks (21 landmarks x 3 coordinates)
            layers.Input(shape=(63,)),
            
            # Dense layers for processing landmark coordinates
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

    def train(self, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the sign language recognition model
        """
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.dataset.prepare_data()
            
            # Create model if not exists
            if self.model is None:
                num_classes = self.dataset.get_num_classes()
                self.create_model(num_classes)
            
            # Train model
            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        str(self.model_path),
                        monitor='val_accuracy',
                        save_best_only=True
                    )
                ]
            )
            
            # Evaluate model
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
            logger.info(f"Test accuracy: {test_accuracy:.4f}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, landmarks):
        """
        Make predictions using the trained model
        """
        try:
            if self.model is None:
                self.load()
            
            # Preprocess landmarks
            X = np.array(landmarks).reshape(1, -1)
            
            # Make prediction
            predictions = self.model.predict(X)
            predicted_class = np.argmax(predictions[0])
            
            # Get class mapping
            class_mapping = self.dataset.get_class_mapping()
            predicted_label = class_mapping.get(predicted_class)
            
            return {
                "label": predicted_label,
                "confidence": float(predictions[0][predicted_class])
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def save(self):
        """
        Save the trained model
        """
        if self.model:
            self.model.save(str(self.model_path))
            logger.info(f"Model saved to {self.model_path}")

    def load(self):
        """
        Load a trained model
        """
        if self.model_path.exists():
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError("No trained model found")
