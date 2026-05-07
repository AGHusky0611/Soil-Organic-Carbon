import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class SoilClassifierXGB:
    def __init__(self, model_path="soil_xgb_model.json", learning_rate=0.1):
        self.model_path = model_path
        self.label_encoder = LabelEncoder()
        
        # We define the model parameters here
        self.params = {
            'n_estimators': 100,
            'learning_rate': learning_rate,
            'max_depth': 5,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss'
        }
        self.clf = xgb.XGBClassifier(**self.params)

    def train(self, X, y, update_existing=True):
        y_encoded = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Incremental learning: update weights if model exists
        if update_existing and os.path.exists(self.model_path):
            self.clf.fit(X_train, y_train, xgb_model=self.model_path)
        else:
            self.clf.fit(X_train, y_train)
            
        accuracy = self.clf.score(X_test, y_test)
        
        self.clf.save_model(self.model_path)
        np.save("soil_classes.npy", self.label_encoder.classes_)
        
        return accuracy

    def update_learning_rate(self, new_lr):
        """Allows the manager to decay the learning rate to improve fine-tuning."""
        self.params['learning_rate'] = new_lr
        self.clf = xgb.XGBClassifier(**self.params)