import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandSignDetector:
    def __init__(self, model_path=None):
        """
        Initialize the hand sign detector using MediaPipe
        Args:
            model_path: Optional path to a trained classifier model
        """
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize the hand detector with good defaults for sign language
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Set to False for video
            max_num_hands=2,          # Detect up to 2 hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Load the classifier model if provided
        self.classifier = None
        if model_path:
            self.classifier = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded classifier model from {model_path}")

    def process_frame(self, frame, draw=True):
        """
        Process a single frame and detect hand landmarks
        Args:
            frame: BGR image
            draw: Whether to draw the landmarks on the frame
        Returns:
            processed_frame: Frame with landmarks drawn (if draw=True)
            hand_landmarks: List of detected hand landmarks
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(frame_rgb)
        
        # Draw landmarks if requested
        if draw and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame, results.multi_hand_landmarks

    def extract_hand_features(self, landmarks):
        """
        Extract features from hand landmarks
        Args:
            landmarks: MediaPipe hand landmarks
        Returns:
            features: Normalized landmark coordinates
        """
        if not landmarks:
            return None
        
        # Extract x, y, z coordinates of all landmarks
        features = []
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(features)

    def predict_sign(self, frame):
        """
        Predict the sign from a frame using MediaPipe and the classifier
        Args:
            frame: BGR image
        Returns:
            prediction: Predicted sign class
            confidence: Confidence score
            processed_frame: Frame with landmarks drawn
        """
        # Process the frame
        processed_frame, hand_landmarks = self.process_frame(frame)
        
        if not hand_landmarks:
            return None, 0.0, processed_frame
        
        # Extract features from the first detected hand
        features = self.extract_hand_features(hand_landmarks[0])
        
        if features is None or self.classifier is None:
            return None, 0.0, processed_frame
        
        # Reshape features for the classifier
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = self.classifier.predict(features)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return predicted_class, confidence, processed_frame

    def start_webcam(self):
        """
        Start webcam capture and real-time hand sign detection
        """
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.error("Failed to read from webcam")
                break
            
            # Process frame
            processed_frame, hand_landmarks = self.process_frame(frame)
            
            # Make prediction if classifier is available
            if self.classifier and hand_landmarks:
                predicted_class, confidence, _ = self.predict_sign(frame)
                if predicted_class is not None:
                    # Draw prediction on frame
                    cv2.putText(
                        processed_frame,
                        f"Sign: {predicted_class} ({confidence:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
            
            # Show the frame
            cv2.imshow('ASL Sign Language Detection', processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test the hand detector
    detector = HandSignDetector()
    detector.start_webcam()
