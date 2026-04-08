"""
Model Trainer  --  Trains and evaluates ML models for A/B test outcome prediction.

Implements Logistic Regression (baseline), Random Forest, and XGBoost
with experiment-level train-test splitting to prevent data leakage.

Author: Sanman Kadam
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Trains and evaluates ML models for predicting A/B test outcomes.
    
    Key design decisions:
    - Split by experiment_id (not random rows) to prevent leakage
    - Train separate models per checkpoint day (1, 3, 5, 7)
    - Compare Logistic Regression, Random Forest, XGBoost
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.models = {}
        self.results = {}
        self.label_encoder = LabelEncoder()
    
    def split_by_experiment(
        self, df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        target_col: str = 'final_outcome'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split experiment IDs into train/val/test sets.
        
        CRITICAL: All 14 daily rows for a given experiment stay
        together in the same split to prevent data leakage.
        
        Uses stratified splitting to maintain class balance.
        
        :return: (train_ids, val_ids, test_ids)
        """
        # Get unique experiment IDs with their labels
        exp_labels = df.groupby('experiment_id')[target_col].first().reset_index()
        
        # Stratified shuffle
        rng = np.random.RandomState(self.seed)
        
        train_ids = []
        val_ids = []
        test_ids = []
        
        for label in exp_labels[target_col].unique():
            label_exps = exp_labels[exp_labels[target_col] == label]['experiment_id'].values.copy()
            rng.shuffle(label_exps)
            
            n = len(label_exps)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_ids.extend(label_exps[:n_train])
            val_ids.extend(label_exps[n_train:n_train + n_val])
            test_ids.extend(label_exps[n_train + n_val:])
        
        return np.array(train_ids), np.array(val_ids), np.array(test_ids)
    
    def prepare_checkpoint_data(
        self, df: pd.DataFrame,
        day: int,
        feature_cols: List[str],
        target_col: str,
        train_ids: np.ndarray,
        test_ids: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Filter data for a specific checkpoint day and split into train/test.
        
        :return: (X_train, y_train, X_test, y_test)
        """
        day_data = df[df['day_number'] == day].copy()
        
        train_data = day_data[day_data['experiment_id'].isin(train_ids)]
        test_data = day_data[day_data['experiment_id'].isin(test_ids)]
        
        # Ensure feature columns exist
        available_features = [c for c in feature_cols if c in day_data.columns]
        
        X_train = train_data[available_features].fillna(0)
        y_train = train_data[target_col]
        X_test = test_data[available_features].fillna(0)
        y_test = test_data[target_col]
        
        return X_train, y_train, X_test, y_test
    
    def get_models(self) -> Dict[str, Any]:
        """Return dictionary of model instances to train."""
        xgb_available = False
        try:
            from xgboost import XGBClassifier
            # Test that the native library actually loads
            XGBClassifier()
            xgb_available = True
        except (ImportError, OSError, Exception):
            xgb_available = False
        
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.seed,
                class_weight='balanced',
                C=1.0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=self.seed,
                class_weight='balanced',
                n_jobs=-1
            ),
        }
        
        if xgb_available:
            models['XGBoost'] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        else:
            print("  [NOTE] XGBoost not available (missing libomp). Using sklearn GradientBoosting as fallback.")
            models['XGBoost'] = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.seed,
            )
            self._using_sklearn_gb = True
        
        return models
    
    def evaluate_model(
        self, model, X_test: pd.DataFrame, y_test: pd.Series,
        model_name: str, day: int, class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model and return comprehensive metrics.
        """
        y_pred = model.predict(X_test)
        
        # Basic metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # ROC-AUC (one-vs-rest)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                y_test_bin = label_binarize(y_test, classes=class_names)
                if y_test_bin.shape[1] == 1:
                    auc = roc_auc_score(y_test_bin, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
            else:
                auc = None
        except Exception:
            auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        return {
            'model_name': model_name,
            'checkpoint_day': day,
            'accuracy': acc,
            'precision_macro': prec,
            'recall_macro': rec,
            'f1_macro': f1,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred,
            'y_test': y_test.values,
        }
    
    def train_and_evaluate_all(
        self, df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'final_outcome',
        checkpoints: List[int] = [1, 3, 5, 7],
    ) -> pd.DataFrame:
        """
        Train all models at all checkpoint days and collect results.
        
        :return: Summary DataFrame with metrics per model per day
        """
        # Encode labels for XGBoost
        class_names = sorted(df[target_col].unique())
        
        # Split by experiment
        train_ids, val_ids, test_ids = self.split_by_experiment(df, target_col=target_col)
        # Combine train + val for final training
        train_ids_full = np.concatenate([train_ids, val_ids])
        
        print(f"Train experiments: {len(train_ids_full)}, Test experiments: {len(test_ids)}")
        print(f"Class names: {class_names}")
        print()
        
        all_results = []
        models_dict = self.get_models()
        
        for day in checkpoints:
            print(f"--- Checkpoint Day {day} ---")
            
            X_train, y_train, X_test, y_test = self.prepare_checkpoint_data(
                df, day, feature_cols, target_col, train_ids_full, test_ids
            )
            
            print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
            
            for model_name, model in models_dict.items():
                # Clone model for each day
                from sklearn.base import clone
                model_clone = clone(model)
                
                # Handle XGBoost label encoding (real XGBoost needs integer labels)
                if model_name == 'XGBoost' and not getattr(self, '_using_sklearn_gb', False):
                    le = LabelEncoder()
                    le.fit(class_names)
                    y_train_enc = le.transform(y_train)
                    model_clone.fit(X_train, y_train_enc)
                    # Wrap for prediction
                    y_pred = le.inverse_transform(model_clone.predict(X_test))
                    y_proba = model_clone.predict_proba(X_test)
                    
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    
                    try:
                        y_test_bin = label_binarize(y_test, classes=class_names)
                        if y_test_bin.shape[1] == 1:
                            auc = roc_auc_score(y_test, y_proba[:, 1])
                        else:
                            auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
                    except Exception:
                        auc = None
                    
                    cm = confusion_matrix(y_test, y_pred, labels=class_names)
                    
                    result = {
                        'model_name': model_name,
                        'checkpoint_day': day,
                        'accuracy': acc,
                        'precision_macro': prec,
                        'recall_macro': rec,
                        'f1_macro': f1,
                        'roc_auc': auc,
                        'confusion_matrix': cm,
                        'y_pred': y_pred,
                        'y_test': y_test.values,
                    }
                    
                    # Store model
                    self.models[(model_name, day)] = model_clone
                else:
                    model_clone.fit(X_train, y_train)
                    result = self.evaluate_model(
                        model_clone, X_test, y_test, model_name, day, class_names
                    )
                    self.models[(model_name, day)] = model_clone
                
                auc_str = f"  AUC: {result['roc_auc']:.3f}" if result.get('roc_auc') else ""
                print(f"  {model_name:25s} | Accuracy: {result['accuracy']:.3f}  "
                      f"F1: {result['f1_macro']:.3f}{auc_str}")
                
                all_results.append(result)
            
            print()
        
        self.results = all_results
        
        # Create summary DataFrame
        summary = pd.DataFrame([{
            'Model': r['model_name'],
            'Day': r['checkpoint_day'],
            'Accuracy': round(r['accuracy'], 3),
            'Precision': round(r['precision_macro'], 3),
            'Recall': round(r['recall_macro'], 3),
            'F1 Score': round(r['f1_macro'], 3),
            'ROC-AUC': round(r['roc_auc'], 3) if r['roc_auc'] else None,
        } for r in all_results])
        
        return summary
    
    def plot_accuracy_comparison(
        self, summary: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot accuracy across checkpoint days for all models."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy over days
        ax = axes[0]
        for model in summary['Model'].unique():
            model_data = summary[summary['Model'] == model]
            ax.plot(model_data['Day'], model_data['Accuracy'], 
                    marker='o', linewidth=2, markersize=8, label=model)
        
        ax.set_xlabel('Checkpoint Day', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Accuracy by Checkpoint Day', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(summary['Day'].unique())
        
        # F1 over days
        ax = axes[1]
        for model in summary['Model'].unique():
            model_data = summary[summary['Model'] == model]
            ax.plot(model_data['Day'], model_data['F1 Score'],
                    marker='s', linewidth=2, markersize=8, label=model)
        
        ax.set_xlabel('Checkpoint Day', fontsize=12)
        ax.set_ylabel('F1 Score (Macro)', fontsize=12)
        ax.set_title('F1 Score by Checkpoint Day', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(summary['Day'].unique())
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(
        self, model_name: str = 'XGBoost',
        class_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrices for a specific model across all checkpoint days."""
        day_results = [r for r in self.results if r['model_name'] == model_name]
        
        if not day_results:
            print(f"No results found for {model_name}")
            return
        
        n_days = len(day_results)
        fig, axes = plt.subplots(1, n_days, figsize=(6 * n_days, 5))
        if n_days == 1:
            axes = [axes]
        
        for idx, result in enumerate(day_results):
            cm = result['confusion_matrix']
            if class_names is None:
                labels = sorted(set(result['y_test']))
            else:
                labels = class_names
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=axes[idx]
            )
            axes[idx].set_xlabel('Predicted', fontsize=11)
            axes[idx].set_ylabel('Actual', fontsize=11)
            axes[idx].set_title(
                f'{model_name}  --  Day {result["checkpoint_day"]}\n'
                f'Accuracy: {result["accuracy"]:.1%}',
                fontsize=12, fontweight='bold'
            )
        
        plt.suptitle(f'Confusion Matrices: {model_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def compute_time_savings(
        self, summary: pd.DataFrame,
        stat_benchmark: pd.DataFrame,
        total_experiment_days: int = 14
    ) -> pd.DataFrame:
        """
        Compare ML prediction speed vs traditional statistical testing.
        
        :param summary: ML model results summary
        :param stat_benchmark: Statistical benchmark from StatisticalTester
        :return: Comparison DataFrame
        """
        best_model = 'XGBoost'
        ml_data = summary[summary['Model'] == best_model].copy()
        
        comparison = []
        for _, row in ml_data.iterrows():
            day = row['Day']
            ml_acc = row['Accuracy']
            
            stat_row = stat_benchmark[stat_benchmark['checkpoint_day'] == day]
            stat_pct = stat_row['pct_resolved'].values[0] if len(stat_row) > 0 else 0
            
            days_saved = total_experiment_days - day
            
            comparison.append({
                'Checkpoint Day': day,
                'ML Accuracy (XGBoost)': f"{ml_acc:.1%}",
                'Stat Test Resolved (%)': f"{stat_pct:.1f}%",
                'Days Saved vs Day 14': days_saved,
                'Time Reduction (%)': f"{days_saved / total_experiment_days:.0%}",
            })
        
        return pd.DataFrame(comparison)
