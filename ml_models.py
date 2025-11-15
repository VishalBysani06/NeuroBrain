import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import pandas as pd

class EEGClassifier:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
        # Intent mappings
        self.motor_imagery_intents = {
            0: "Left Hand - 0",
            1: "Right Hand - 1", 
            2: "Feet - 2 ",
            3: "Tongue - 3 "
        }
        
        self.speech_intents = {
            0: "Yes - 0",
            1: "No - 1",
            2: "Start - 2",
            3: "Stop - 3",
            4: "Help - 4"
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_synthetic_labels(self, features, intent_type='motor_imagery'):
        """Create synthetic labels for demonstration (clustering-based)"""
        from sklearn.cluster import KMeans
        
        if intent_type == 'motor_imagery':
            n_clusters = len(self.motor_imagery_intents)
        else:
            n_clusters = len(self.speech_intents)
        
        # Use KMeans to create initial clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        return labels
    
    def train(self, features, labels=None, intent_type='motor_imagery', test_size=0.2):
        """Train the classifier"""
        if labels is None:
            # Create synthetic labels for demonstration
            labels = self.prepare_synthetic_labels(features, intent_type)
        
        # Store feature names
        if hasattr(features, 'columns'):
            self.feature_names = features.columns.tolist()
            features = features.values
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(features)
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=5)
        
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self.get_feature_importance()
        }
        
        return results
    
    def predict(self, features):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(features, 'values'):
            features = features.values
        
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode predictions
        decoded_predictions = self.label_encoder.inverse_transform(predictions)
        
        return {
            'predictions': decoded_predictions,
            'probabilities': probabilities,
            'raw_predictions': predictions
        }
    
    def predict_with_confidence(self, features, threshold=0.7):
        """Make predictions with confidence filtering"""
        results = self.predict(features)
        
        confident_predictions = []
        for i, prob in enumerate(results['probabilities']):
            max_prob = np.max(prob)
            if max_prob >= threshold:
                confident_predictions.append({
                    'prediction': results['predictions'][i],
                    'confidence': max_prob,
                    'reliable': True
                })
            else:
                confident_predictions.append({
                    'prediction': results['predictions'][i],
                    'confidence': max_prob,
                    'reliable': False
                })
        
        return confident_predictions
    
    def get_feature_importance(self):
        """Get feature importance (for tree-based models)"""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if self.feature_names and len(self.feature_names) == len(importance):
                return dict(zip(self.feature_names, importance))
            else:
                # Return as dict with generic names
                return {f'Feature_{i}': imp for i, imp in enumerate(importance)}
        else:
            return None
    
    def optimize_hyperparameters(self, features, labels, cv=5):
        """Optimize model hyperparameters using GridSearch"""
        if hasattr(features, 'values'):
            features = features.values
        
        X_scaled = self.scaler.fit_transform(features)
        y_encoded = self.label_encoder.fit_transform(labels)
        
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'svm':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_scaled, y_encoded)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'motor_imagery_intents': self.motor_imagery_intents,
            'speech_intents': self.speech_intents
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.motor_imagery_intents = model_data.get('motor_imagery_intents', self.motor_imagery_intents)
        self.speech_intents = model_data.get('speech_intents', self.speech_intents)
        self.is_trained = True
