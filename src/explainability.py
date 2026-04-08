"""
Model Explainability  --  SHAP values, feature importance, and interpretability.

Provides recruiter-friendly explanations of model predictions using
SHAP TreeExplainer for XGBoost and feature importance analysis.

Author: Sanman Kadam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    Explains ML model predictions using:
    1. Built-in feature importance (XGBoost/RF)
    2. SHAP values (TreeExplainer)
    3. Partial dependence analysis
    """
    
    def __init__(self):
        self.shap_values = None
        self.explainer = None
    
    def plot_feature_importance(
        self, model, feature_names: List[str],
        top_n: int = 15, title: str = 'Feature Importance',
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Plot built-in feature importance from tree-based models.
        
        :param model: Trained model with feature_importances_ attribute
        :param feature_names: List of feature names
        :param top_n: Number of top features to display
        :return: DataFrame of feature importances
        """
        importances = model.feature_importances_
        
        imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            imp_df['Feature'][::-1],
            imp_df['Importance'][::-1],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, top_n)),
            edgecolor='white', linewidth=0.5
        )
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, imp_df['Importance'][::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return imp_df
    
    def compute_shap_values(
        self, model, X: pd.DataFrame,
        feature_names: List[str] = None
    ) -> Any:
        """
        Compute SHAP values using TreeExplainer.
        
        :param model: Trained tree-based model (XGBoost, RF)
        :param X: Feature matrix
        :return: SHAP values object
        """
        try:
            import shap
        except ImportError:
            print("SHAP library not installed. Install with: pip install shap")
            return None
        
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X)
        
        return self.shap_values
    
    def plot_shap_summary(
        self, X: pd.DataFrame,
        class_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot SHAP summary (beeswarm) plot.
        
        Shows which features have the most impact on predictions
        and in which direction.
        """
        try:
            import shap
        except ImportError:
            print("SHAP library not installed.")
            return
        
        if self.shap_values is None:
            print("Run compute_shap_values() first.")
            return
        
        # For multi-class, shap_values is a list of arrays
        if isinstance(self.shap_values, list):
            # Plot for each class
            for i, class_name in enumerate(class_names or range(len(self.shap_values))):
                fig = plt.figure(figsize=(12, 7))
                shap.summary_plot(
                    self.shap_values[i], X,
                    show=False, max_display=15
                )
                plt.title(f'SHAP Summary  --  Class: {class_name}',
                         fontsize=14, fontweight='bold')
                plt.tight_layout()
                if save_path:
                    plt.savefig(
                        save_path.replace('.png', f'_{class_name}.png'),
                        dpi=150, bbox_inches='tight'
                    )
                plt.show()
        else:
            fig = plt.figure(figsize=(12, 7))
            shap.summary_plot(self.shap_values, X, show=False, max_display=15)
            plt.title('SHAP Feature Impact Summary', fontsize=14, fontweight='bold')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
    
    def plot_shap_dependence(
        self, X: pd.DataFrame,
        feature: str,
        interaction_feature: str = 'auto',
        class_idx: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot SHAP dependence plot for a specific feature.
        
        Shows how a single feature's value impacts the prediction,
        colored by interaction with another feature.
        """
        try:
            import shap
        except ImportError:
            print("SHAP library not installed.")
            return
        
        if self.shap_values is None:
            print("Run compute_shap_values() first.")
            return
        
        sv = self.shap_values[class_idx] if isinstance(self.shap_values, list) else self.shap_values
        
        fig = plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, sv, X,
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f'SHAP Dependence: {feature}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_plain_english_summary(
        self, importance_df: pd.DataFrame,
        model_name: str = 'XGBoost',
        checkpoint_day: int = 5
    ) -> str:
        """
        Generate a recruiter-friendly plain-English summary of model behavior.
        
        :param importance_df: Feature importance DataFrame
        :return: Formatted string explanation
        """
        top_features = importance_df.head(5)
        
        feature_explanations = {
            'z_statistic': 'the strength of statistical evidence (z-score) between groups',
            'bayesian_prob_treatment_wins': 'the Bayesian probability that the treatment outperforms the control',
            'lift_to_se_ratio': 'the signal-to-noise ratio of the observed improvement',
            'cumulative_lift': 'the raw difference in conversion rates between treatment and control',
            'p_value_current': 'the current statistical p-value of the experiment',
            'sample_size_progress': 'how much of the expected traffic has been collected',
            'credible_interval_width': 'how uncertain the lift estimate still is',
            'relative_uplift': 'the percentage improvement of treatment over control',
            'total_visitors': 'the total number of visitors observed so far',
            'conversion_gap_trend': 'whether the performance gap is growing or shrinking over time',
            'rolling_cr_control_3d': 'the recent 3-day conversion trend for the control group',
            'rolling_cr_treatment_3d': 'the recent 3-day conversion trend for the treatment group',
            'baseline_conversion_rate': 'the starting conversion rate before the experiment',
            'pooled_standard_error': 'the statistical uncertainty in the lift measurement',
            'bayesian_expected_lift': 'the Bayesian estimate of the true lift',
            'sample_size_ratio': 'the balance of traffic between treatment and control',
        }
        
        lines = [
            f"## Model Interpretation Summary ({model_name}, Day {checkpoint_day})",
            "",
            f"The {model_name} model uses the following signals (in order of importance) "
            f"to predict whether an experiment's treatment will win, lose, or be inconclusive "
            f"by day {checkpoint_day}:",
            "",
        ]
        
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            feat = row['Feature']
            imp = row['Importance']
            explanation = feature_explanations.get(feat, feat)
            lines.append(
                f"**{i}. {feat}** (importance: {imp:.3f}): "
                f"Measures {explanation}."
            )
        
        lines.extend([
            "",
            "### What This Means in Practice",
            "",
            "The model primarily relies on **statistical evidence strength** (z-statistic, "
            "Bayesian probability) rather than raw conversion numbers. This makes sense because:",
            "",
            "1. A large z-statistic at day 5 is a strong signal that the final result will "
            "be significant  --  the evidence is already accumulating.",
            "2. The Bayesian probability provides a direct estimate of the chance that "
            "treatment outperforms control, accounting for uncertainty.",
            "3. The signal-to-noise ratio captures whether the observed lift is large "
            "relative to the remaining uncertainty.",
            "",
            "Features like sample size progress and credible interval width help the model "
            "understand *how confident* it should be, not just *what direction* the result "
            "points in.",
        ])
        
        return "\n".join(lines)
