"""
Network Anomaly Detection - Complete Implementation
All-in-one Python script for your CSC networking final project
Author: Samuel Knowles and Matthew Uttecht
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime

# ML Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# PHASE 1: DATA LOADING AND EXPLORATION
# ============================================================================

class DataProcessor:
    def __init__(self, train_path, test_path):
        print("Loading data...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        
        # Clean column names
        self.train_df.columns = self.train_df.columns.str.strip()
        self.test_df.columns = self.test_df.columns.str.strip()
        
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Testing data shape: {self.test_df.shape}")
    
    def explore(self):
        print("\n===== DATASET EXPLORATION =====")
        print("\nDataset Info:")
        print(self.train_df.info())
        print("\nFirst 5 rows:")
        print(self.train_df.head())
        print("\nMissing values:")
        print(self.train_df.isnull().sum().sum())
        print("\nClass distribution:")
        print(self.train_df["attack"].value_counts())
    
    def preprocess(self, apply_smote=False):
        print("\n===== DATA PREPROCESSING =====")

        label_col = "attack"
        
        # Create binary classification
        self.train_df['binary_class'] = (self.train_df[label_col] != 'normal').astype(int)
        self.test_df['binary_class'] = (self.test_df[label_col] != 'normal').astype(int)
        
        # Identify categorical and numerical columns
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            self.train_df[col] = le.fit_transform(self.train_df[col])
            self.test_df[col] = le.transform(self.test_df[col])
            self.label_encoders[col] = le
        
        # Prepare features and targets
        self.X_train = self.train_df.drop([label_col, 'binary_class'], axis=1)
        self.y_train = self.train_df['binary_class']
        
        self.X_test = self.test_df.drop([label_col, 'binary_class'], axis=1)
        self.y_test = self.test_df['binary_class']
        
        # Store feature names
        self.feature_names = self.X_train.columns.tolist()
        
        # Handle missing values
        self.X_train = self.X_train.fillna(self.X_train.mean(numeric_only=True))
        self.X_test = self.X_test.fillna(self.X_test.mean(numeric_only=True))
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        
        # Apply SMOTE if requested
        if apply_smote:
            smote = SMOTE(random_state=42, k_neighbors=3)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_preprocessed_data(self, csv_prefix='', model_prefix=''):
        pd.DataFrame(self.X_train).to_csv(f'{csv_prefix}X_train_preprocessed.csv', index=False)
        pd.DataFrame(self.X_test).to_csv(f'{csv_prefix}X_test_preprocessed.csv', index=False)
        self.y_train.to_csv(f'{csv_prefix}y_train.csv', index=False, header=False)
        self.y_test.to_csv(f'{csv_prefix}y_test.csv', index=False, header=False)
        
        with open(f'{model_prefix}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{model_prefix}feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)


# ============================================================================
# PHASE 2: FEATURE ENGINEERING AND SELECTION
# ============================================================================

class FeatureEngineer:
    def __init__(self, X_train, y_train, X_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.feature_importance = None
    
    def analyze_importance_rf(self, n_features=20):
        print("\n===== FEATURE IMPORTANCE ANALYSIS (Random Forest) =====")
        
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_temp.fit(self.X_train, self.y_train)
        
        # Get importances
        importances = rf_temp.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Initialize categories for graph
        categories = []
        
        print(f"\nTop {n_features} Important Features:")
        for i in range(min(n_features, len(importances))):
            print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.6f}")
            categories.append(f"Feature {indices[i]}")

        
        # Visualize
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.bar(range(n_features), importances[indices[:n_features]])
        plt.xticks(range(n_features), categories, rotation=45)
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig('images/feature_importance.png', dpi=300)
        print("\nFeature importance plot saved: images/feature_importance.png")
        
        self.feature_importance = importances
        return importances


# ============================================================================
# PHASE 3: MODEL TRAINING
# ============================================================================

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.predictions = {}
        self.results = []
    
    def train_decision_tree(self):
        model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            criterion='entropy'
        )
        model.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = model
        
        return model
    
    def train_random_forest(self):
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        
        return model
    
    def train_xgboost(self):
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        
        return model
    
    def train_knn(self, n_neighbors=5):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        self.models['KNN'] = model
        
        return model
    
    def train_logistic_regression(self):
        model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = model
        
        return model
    
    def train_svm(self, sample=True):
        if sample:
            # Use subset for speed
            sample_size = min(10000, len(self.X_train))
            indices = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_sample = self.X_train[indices]
            y_sample = self.y_train.iloc[indices] if hasattr(self.y_train, 'iloc') else self.y_train[indices]
        else:
            X_sample = self.X_train
            y_sample = self.y_train
        
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_sample, y_sample)
        self.models['SVM'] = model
        
        return model
    
    def train_all(self):
        print("\n===== TRAINING ALL MODELS =====")
        
        self.train_decision_tree()
        print("Decision Tree trained.")
        self.train_random_forest()
        print("Random Forest trained.")
        self.train_xgboost()
        print("XGBoost trained.")
        self.train_knn()
        print("KNN trained.")
        self.train_logistic_regression()
        print("Logistic Regression trained.")
        self.train_svm()
        print("SVM trained.\n")

        print("All models trained.")
    
    def evaluate_model(self, model_name, model):
        y_pred = model.predict(self.X_test)
        
        # Get probabilities for ROC-AUC
        try:
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        except:
            y_pred_proba = None
            roc_auc = None
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        result = {
            'Model': model_name,
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(self.y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc if roc_auc is not None else 0,
            'False Positive Rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'False Negative Rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'True Positive Rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'True Negative Rate': tn / (tn + fp) if (tn + fp) > 0 else 0
        }
        
        self.predictions[model_name] = y_pred
        self.results.append(result)
        
        return result
    
    def evaluate_all(self):
        print("\n===== MODEL EVALUATION =====")
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            result = self.evaluate_model(model_name, model)
            
            print(f"  Accuracy: {result['Accuracy']:.4f}")
            print(f"  Precision: {result['Precision']:.4f}")
            print(f"  Recall: {result['Recall']:.4f}")
            print(f"  F1-Score: {result['F1-Score']:.4f}")
            print(f"  ROC-AUC: {result['ROC-AUC']:.4f}")
            print(f"  FPR: {result['False Positive Rate']:.4f}")
            print(f"  FNR: {result['False Negative Rate']:.4f}")
    
    def get_comparison_df(self):
        df = pd.DataFrame(self.results)
        return df.sort_values('F1-Score', ascending=False)
    
    def print_detailed_report(self):
        print("\n===== DETAILED CLASSIFICATION REPORTS =====")
        
        df = self.get_comparison_df()
        best_model_name = df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"F1-Score: {df.iloc[0]['F1-Score']:.4f}\n")
        
        y_pred = self.predictions[best_model_name]
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Normal', 'Attack']))
    
    def plot_comparison(self):
        df = self.get_comparison_df()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for metric in metrics:
            ax.plot(df['Model'], df[metric], 'o-', label=metric, linewidth=2, markersize=8)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('images/model_comparison.png', dpi=300)
        print("\nModel comparison plot saved: images/model_comparison.png")
        
        return df
    
    def plot_confusion_matrices(self):
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = self.predictions[model_name]
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Normal', 'Attack'],
                       yticklabels=['Normal', 'Attack'])
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('images/confusion_matrices.png', dpi=300)
        print("Confusion matrices saved: images/confusion_matrices.png")
    
    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            try:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)
            except:
                pass
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/roc_curves.png', dpi=300)
        print("ROC curves saved: images/roc_curves.png")
    
    def save_best_model(self):
        df = self.get_comparison_df()
        best_model_name = df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        filename = f"models/best_model_{best_model_name.lower().replace(' ', '_')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"\nBest model saved: {filename}")
        return filename
    
    def save_all_models(self):
        print("\n===== SAVING MODELS =====")
        
        for model_name, model in self.models.items():
            filename = f"models/model_{model_name.lower().replace(' ', '_')}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved: {filename}")


# ============================================================================
# PHASE 4: HYPERPARAMETER TUNING
# ============================================================================

class HyperparameterTuner:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def tune_decision_tree(self):
        print("\n===== TUNING DECISION TREE =====")
        
        param_grid = {
            'max_depth': [10, 15, 20, 25],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [5, 10, 15],
            'criterion': ['gini', 'entropy']
        }
        
        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {grid.best_params_}")
        print(f"Best CV score: {grid.best_score_:.4f}")
        
        return grid.best_estimator_
    
    def tune_random_forest(self):
        print("\n===== TUNING RANDOM FOREST =====")
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [15, 20, 25],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10]
        }
        
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {grid.best_params_}")
        print(f"Best CV score: {grid.best_score_:.4f}")
        
        return grid.best_estimator_


# ============================================================================
# PHASE 5: MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("NETWORK ANOMALY DETECTION - FINAL PROJECT")
    print("CSC Networking - Samuel Knowles & Matthew Uttecht")
    print("=" * 60)
    
    # Phase 1: Data Processing
    print("\n" + "=" * 60)
    print("DATA LOADING AND PREPROCESSING")
    print("=" * 60)
    processor = DataProcessor('raw_data/Training_data.csv', 'raw_data/Testing_data.csv')
    processor.explore()
    X_train, X_test, y_train, y_test = processor.preprocess(apply_smote=False)
    processor.save_preprocessed_data(csv_prefix='processed_data/', model_prefix='models/')
    
    # Phase 2: Feature Engineering
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    engineer = FeatureEngineer(X_train, y_train)
    engineer.analyze_importance_rf(n_features=20)
    
    # Phase 3: Model Training and Evaluation
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    trainer.train_all()
    trainer.evaluate_all()
    
    # Generate Visuals
    print("\n" + "=" * 60)
    print("GENERATING VISUALS")
    print("=" * 60)
    trainer.plot_comparison()
    trainer.plot_confusion_matrices()
    trainer.plot_roc_curves()
    
    # Detailed report
    trainer.print_detailed_report()
    
    # Save models
    trainer.save_all_models()
    trainer.save_best_model()
    
    # Phase 4: Hyperparameter Tuning
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    tuner = HyperparameterTuner(X_train, y_train)
    tuned_dt = tuner.tune_decision_tree()
    tuned_rf = tuner.tune_random_forest()


if __name__ == "__main__":
    main()
