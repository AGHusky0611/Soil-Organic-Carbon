import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

class SoilClassifierXGB:
    def __init__(self, model_path="soil_xgb_model.json"):
        """Initializes the XGBoost classifier with fixed hyperparameters."""
        self.model_path = model_path
        self.encoder = LabelEncoder()
        
        # Configuration for multi-class probability classification
        self.params = {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 6,
            
            # --- FEATURE & ROW DROPOUT ---
            'subsample': 0.8,         # Drops 20% of images per tree
            'colsample_bytree': 0.8,  # Drops 20% of math features per tree
            
            # --- THE TRUE DROPOUT METHOD (TEACHER'S REQUEST) ---
            'booster': 'dart',        # Activates DART (Dropout Architecture)
            'rate_drop': 0.1,         # Randomly drops 10% of previous trees 
            'skip_drop': 0.5,         # 50% chance to skip dropout in an iteration
            
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
        self.clf = xgb.XGBClassifier(**self.params)

    def train(self, X, y):
        """
        Processes labels and trains the model. 
        Returns performance metrics based on a 20% test split.
        """
        y_enc = self.encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        
        report = classification_report(
            y_test, y_pred, target_names=self.encoder.classes_, output_dict=True, zero_division=0
        )
        conf_mat = confusion_matrix(y_test, y_pred)

        results = {
            "accuracy": self.clf.score(X_test, y_test),
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "confusion_matrix": conf_mat,
            "class_names": self.encoder.classes_,
            "learning_rate": self.params['learning_rate'],
            "max_depth": self.params['max_depth'],
            "n_estimators": self.params['n_estimators'],
            "subsample": self.params['subsample']
        }
        
        self.clf.save_model(self.model_path)
        np.save("soil_classes.npy", self.encoder.classes_)
        
        return results

    def update_params(self, new_lr=None, new_depth=None):
        if new_lr: self.params['learning_rate'] = new_lr
        if new_depth: self.params['max_depth'] = new_depth
        self.clf = xgb.XGBClassifier(**self.params)


def tune_xgb_classification(
    X,
    y,
    param_grid=None,
    test_size=0.2,
    random_state=42,
    model_path="soil_xgb_model.json",
    classes_path="soil_classes.npy",
    results_path="xgb_classification_tuning.csv",
    report_path="xgb_classification_best_report.json",
    confusion_path="xgb_classification_confusion_matrix.xlsx",
):
    if param_grid is None:
        param_grid = {
            "n_estimators": [150, 250, 350],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [4, 6, 8],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "booster": ["dart"],
            "rate_drop": [0.1, 0.2],
            "skip_drop": [0.5],
        }

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc,
    )

    base_params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": random_state,
    }

    best_score = -1.0
    best_model = None
    best_params = None
    best_report = None
    best_confusion = None
    results = []

    for params in ParameterGrid(param_grid):
        model = xgb.XGBClassifier(**base_params, **params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
            early_stopping_rounds=30,
        )

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(
            y_test, preds, average="weighted", zero_division=0
        )
        report = classification_report(
            y_test,
            preds,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        conf_mat = confusion_matrix(y_test, preds)

        results.append(
            {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                **params,
            }
        )

        if macro_f1 > best_score:
            best_score = macro_f1
            best_model = model
            best_params = params
            best_report = report
            best_confusion = conf_mat

    if best_model is None:
        return None

    best_model.save_model(model_path)
    np.save(classes_path, encoder.classes_)

    pd.DataFrame(results).to_csv(results_path, index=False)
    if best_report is not None:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(best_report, handle, indent=2)
    if best_confusion is not None:
        df_cm = pd.DataFrame(
            best_confusion, index=encoder.classes_, columns=encoder.classes_
        )
        df_cm.to_excel(confusion_path, index=True)

    return {
        "best_params": best_params,
        "best_macro_f1": best_score,
        "results_path": results_path,
        "model_path": model_path,
        "classes_path": classes_path,
        "report_path": report_path,
        "confusion_path": confusion_path,
    }