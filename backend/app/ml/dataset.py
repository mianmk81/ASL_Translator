import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class SignLanguageDataset:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.landmarks_file = self.data_dir / "landmarks.json"
        self.label_encoder = LabelEncoder()
        
    def add_sample(self, landmarks, label):
        """
        Add a new sample to the dataset
        """
        try:
            # Load existing data
            data = self._load_data()
            
            # Add new sample
            data.append({
                "landmarks": landmarks,
                "label": label
            })
            
            # Save updated data
            self._save_data(data)
            logger.info(f"Added new sample with label: {label}")
            
        except Exception as e:
            logger.error(f"Error adding sample: {str(e)}")
            raise
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for training
        """
        try:
            # Load all data
            data = self._load_data()
            if not data:
                raise ValueError("No data available for training")
            
            # Split into features and labels
            X = np.array([sample["landmarks"] for sample in data])
            y = np.array([sample["label"] for sample in data])
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            y_onehot = pd.get_dummies(y_encoded).values
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_onehot,
                test_size=test_size,
                random_state=random_state,
                stratify=y_onehot
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def _load_data(self):
        """
        Load dataset from file
        """
        if self.landmarks_file.exists():
            with open(self.landmarks_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_data(self, data):
        """
        Save dataset to file
        """
        with open(self.landmarks_file, 'w') as f:
            json.dump(data, f)
    
    def get_num_classes(self):
        """
        Get number of unique classes in dataset
        """
        data = self._load_data()
        if not data:
            return 0
        labels = [sample["label"] for sample in data]
        return len(set(labels))

    def get_class_mapping(self):
        """
        Get mapping between class indices and labels
        """
        data = self._load_data()
        if not data:
            return {}
        labels = [sample["label"] for sample in data]
        self.label_encoder.fit(labels)
        return dict(enumerate(self.label_encoder.classes_))
