import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridSignDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the hybrid sign detector using both MediaPipe and custom classifier
        Args:
            model_path: Path to the trained classifier model
        """
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands with good defaults for sign language
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,          # Set to 2 to detect both hands
            min_detection_confidence=0.6,  # Slightly lower threshold for better multi-hand detection
            min_tracking_confidence=0.5
        )
        
        # Class names for ASL alphabet
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                           'del', 'nothing', 'space']
        
        # Load custom classifier if provided
        self.classifier = None
        if model_path:
            self.classifier = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded classifier model from {model_path}")
        
        # Initialize gesture detection parameters
        self.prev_landmarks = None
        self.gesture_buffer = []
        self.gesture_threshold = 0.8
        self.movement_threshold = 0.1
        
        # Performance tracking
        self.fps = 0
        self.prev_time = 0

    def _calculate_hand_angle(self, landmarks) -> float:
        """Calculate the angle of the hand relative to vertical"""
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        
        # Calculate angle between wrist-middle_tip line and vertical
        dx = middle_tip.x - wrist.x
        dy = middle_tip.y - wrist.y
        angle = np.arctan2(dx, dy)
        return np.degrees(angle)

    def _detect_dynamic_gesture(self, landmarks) -> Optional[str]:
        """Detect dynamic gestures using movement patterns"""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return None
        
        # Calculate movement between frames
        movement = 0
        for i, (curr, prev) in enumerate(zip(landmarks.landmark, self.prev_landmarks.landmark)):
            movement += np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
        
        movement = movement / len(landmarks.landmark)
        
        # Update gesture buffer
        if movement > self.movement_threshold:
            self.gesture_buffer.append(movement)
        
        # Keep buffer size manageable
        if len(self.gesture_buffer) > 10:
            self.gesture_buffer.pop(0)
        
        # Detect specific gestures
        if len(self.gesture_buffer) >= 5:
            # Example: Detect waving motion
            if self._is_waving_pattern(self.gesture_buffer):
                return "WAVE"
            
            # Example: Detect circular motion
            if self._is_circular_pattern(self.gesture_buffer):
                return "CIRCLE"
        
        self.prev_landmarks = landmarks
        return None

    def _is_waving_pattern(self, movements: List[float]) -> bool:
        """Detect if movement pattern represents waving"""
        if len(movements) < 5:
            return False
        
        # Look for alternating high-low movement pattern
        peaks = 0
        for i in range(1, len(movements) - 1):
            if movements[i] > movements[i-1] and movements[i] > movements[i+1]:
                peaks += 1
        
        return peaks >= 2

    def _is_circular_pattern(self, movements: List[float]) -> bool:
        """Detect if movement pattern represents circular motion"""
        if len(movements) < 8:
            return False
        
        # Look for smooth, consistent movement
        avg_movement = np.mean(movements)
        std_movement = np.std(movements)
        
        return std_movement < 0.3 * avg_movement

    def _extract_features(self, landmarks) -> np.ndarray:
        """Extract consistent features from landmarks"""
        features = []
        
        # Extract basic landmark coordinates (21 landmarks * 3 coordinates = 63 features)
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
            
        # Add hand angle as the 64th feature
        angle = self._calculate_hand_angle(landmarks)
        features.append(angle)
        
        return np.array(features).reshape(1, -1)  # Reshape for model input

    def process_frame(self, frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, List[str], List[float]]:
        """Process a frame and return the detected signs for both hands"""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(frame_rgb)
        
        # Lists to store results for each hand
        detected_signs = []
        confidences = []
        
        # Create semi-transparent overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Draw hand landmarks and get predictions
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get wrist position for hand label
                wrist = hand_landmarks.landmark[0]
                x = int(wrist.x * frame.shape[1])
                y = int(wrist.y * frame.shape[0])
                cv2.putText(frame, f"Hand {idx+1}", (x-20, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Extract features and predict static sign
                if self.classifier is not None:
                    try:
                        features = self._extract_features(hand_landmarks)
                        prediction = self.classifier.predict(features)[0]
                        sign = self.class_names[np.argmax(prediction)]
                        confidence = float(np.max(prediction))
                        
                        detected_signs.append(sign)
                        confidences.append(confidence)
                        
                        # Draw prediction and confidence for each hand
                        y_offset = 40 + idx * 50
                        cv2.putText(frame, f"Hand {idx+1}: {sign}", (10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, y_offset + 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                    except Exception as e:
                        logger.error(f"Error during prediction: {str(e)}")
                        detected_signs.append("ERROR")
                        confidences.append(0.0)
        
        # Draw FPS
        fps = self._calculate_fps()
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame, detected_signs, confidences

    def _calculate_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time) if self.prev_time else 0.0
        self.prev_time = current_time
        self.fps = fps
        return fps

    def start_webcam(self):
        """Start webcam capture and detection"""
        try:
            # Try different webcam indices
            for cam_index in [0, 1, -1]:
                cap = cv2.VideoCapture(cam_index)
                if cap.isOpened():
                    break
                cap.release()
            
            if not cap.isOpened():
                logger.error("Could not find any working webcam")
                return
                
            logger.info("Starting webcam capture. Press 'q' to quit.")
            logger.info("Try showing both hands to see dual detection!")
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    logger.error("Failed to read frame")
                    break
                
                # Process frame
                processed_frame, signs, confidences = self.process_frame(frame)
                
                if processed_frame is not None:
                    cv2.imshow('ASL Detection', processed_frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logger.error(f"Error during webcam capture: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def train_model(self, data_dir: str, epochs: int = 50, batch_size: int = 32):
        """Train the model on ASL dataset"""
        logger.info("Starting model training...")
        data_path = Path(data_dir)
        
        # Prepare datasets
        X_train = []
        y_train = []
        
        logger.info("Processing training images...")
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = data_path / 'train' / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob('*.jpg'):
                try:
                    # Load and process image
                    image = cv2.imread(str(img_path))
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Extract landmarks
                    results = self.hands.process(image_rgb)
                    if not results.multi_hand_landmarks:
                        continue
                    
                    # Extract features from the first hand
                    landmarks = results.multi_hand_landmarks[0]
                    features = self._extract_features(landmarks)
                    
                    X_train.append(features[0])
                    y_train.append(class_idx)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Split into train/validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Create model
        input_shape = X_train.shape[1]
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create models directory if it doesn't exist
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model path
        model_path = models_dir / 'hand_landmarks_model.h5'
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    str(model_path),  # Convert path to string
                    save_best_only=True
                )
            ]
        )
        
        # Save model using string path
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Plot training history
        self._plot_training_history(history.history)
        
        return history

    def _plot_training_history(self, history):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training')
        plt.plot(history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hybrid Sign Language Detector')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data_dir', type=str, default='../data/asl_alphabet',
                       help='Directory containing training data')
    parser.add_argument('--model_path', type=str, default='models/hand_landmarks_model.h5',
                       help='Path to save/load the model')
    args = parser.parse_args()
    
    detector = HybridSignDetector(model_path=args.model_path if not args.train else None)
    
    if args.train:
        detector.train_model(args.data_dir)
    else:
        detector.start_webcam()
