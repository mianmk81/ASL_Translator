import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
import json
from datetime import datetime

from data_manager import DataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingManager:
    def __init__(self, data_manager: DataManager, model_dir: str = "models"):
        self.data_manager = data_manager
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.current_model = None
        self.training_history = None
        
        # Available architectures
        self.architectures = {
            "mobilenet": self._create_mobilenet,
            "resnet": self._create_resnet,
            "efficientnet": self._create_efficientnet
        }

    def _create_mobilenet(self, num_classes: int) -> Model:
        """Create MobileNetV2 model"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        return model

    def _create_resnet(self, num_classes: int) -> Model:
        """Create ResNet50 model"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        return model

    def _create_efficientnet(self, num_classes: int) -> Model:
        """Create EfficientNetB0 model"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        return model

    async def train(
        self,
        architecture: str = "mobilenet",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 10,
        validation_split: float = 0.2,
        use_augmentation: bool = True,
        use_datasets: Optional[list] = None
    ) -> Dict:
        """Train the model"""
        try:
            # Prepare data
            train_ds, val_ds, label_mapping = self.data_manager.prepare_dataset(
                batch_size=batch_size,
                validation_split=validation_split,
                use_datasets=use_datasets or ["asl_alphabet", "custom"]
            )

            # Apply augmentation if requested
            if use_augmentation:
                train_ds = train_ds.map(
                    lambda x, y: (self.data_manager.apply_augmentation(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )

            # Create model
            num_classes = len(label_mapping)
            self.current_model = self.architectures[architecture](num_classes)

            # Compile model
            self.current_model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Setup callbacks
            callbacks = [
                ModelCheckpoint(
                    self.model_dir / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5",
                    monitor='val_accuracy',
                    save_best_only=True
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                )
            ]

            # Train model
            history = self.current_model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks
            )

            # Save training history
            self.training_history = history.history
            
            # Save label mapping
            with open(self.model_dir / "label_mapping.json", "w") as f:
                json.dump(label_mapping, f)

            return {
                "status": "success",
                "history": self.training_history,
                "label_mapping": label_mapping
            }

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Make a prediction for a single image"""
        if self.current_model is None:
            raise ValueError("No model has been trained yet")

        # Preprocess image
        image = tf.image.resize(image, (224, 224))
        image = tf.expand_dims(image, 0)
        
        # Make prediction
        prediction = self.current_model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]

        # Load label mapping
        with open(self.model_dir / "label_mapping.json", "r") as f:
            label_mapping = json.load(f)

        # Reverse label mapping to get class name
        label_mapping_rev = {v: k for k, v in label_mapping.items()}
        predicted_label = label_mapping_rev[predicted_class]

        return predicted_label, float(confidence)

    def get_training_stats(self) -> Dict:
        """Get statistics about the training history"""
        if self.training_history is None:
            return {"status": "No training history available"}

        return {
            "status": "success",
            "final_accuracy": self.training_history["accuracy"][-1],
            "final_val_accuracy": self.training_history["val_accuracy"][-1],
            "final_loss": self.training_history["loss"][-1],
            "final_val_loss": self.training_history["val_loss"][-1],
            "epochs_trained": len(self.training_history["accuracy"])
        }
