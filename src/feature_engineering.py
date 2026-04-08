"""
Feature Engineering  --  Builds ML features from partial experiment data.

Creates conversion features, statistical uncertainty measures,
Bayesian posterior features, and trend signals for each experiment
at each day checkpoint.

Author: Sanman Kadam
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Optional


class FeatureEngineer:
    """
    Transforms raw daily experiment snapshots into ML-ready features.
    
    Feature categories:
    1. Conversion features (lift, uplift, gap trends)
    2. Statistical uncertainty (SE, z-stat, p-value)
    3. Sample size features (progress, balance)
    4. Bayesian features (posterior probability, credible intervals)
    5. Experiment metadata (one-hot encoded)
    """
    
    CATEGORICAL_COLS = ['device_type', 'region', 'traffic_source', 
                        'segment', 'experiment_category']
    
    def __init__(self):
        self.feature_names = []
    
    def compute_conversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute conversion-rate-based features.
        
        Features:
        - cumulative_lift: treatment CR - control CR
        - relative_uplift: lift / control CR (relative improvement)
        - conversion_gap_trend: slope of observed_lift over recent days
        - rolling_cr_control_3d: 3-day rolling average control CR
        - rolling_cr_treatment_3d: 3-day rolling average treatment CR
        """
        df = df.copy()
        
        # Basic lift features (already present, ensuring they exist)
        df['cumulative_lift'] = (
            df['conversion_rate_treatment'] - df['conversion_rate_control']
        )
        
        df['relative_uplift'] = np.where(
            df['conversion_rate_control'] > 0,
            df['cumulative_lift'] / df['conversion_rate_control'],
            0
        )
        
        # Conversion gap trend: slope of lift over last 3 days
        # Computed per experiment using rolling window
        def compute_lift_slope(group):
            """Compute rolling slope of observed lift."""
            slopes = []
            for i in range(len(group)):
                if i < 2:  # Not enough history
                    slopes.append(0.0)
                else:
                    window = group['observed_lift'].iloc[max(0, i-2):i+1].values
                    x = np.arange(len(window))
                    if len(window) >= 2:
                        slope, _, _, _, _ = stats.linregress(x, window)
                        slopes.append(slope)
                    else:
                        slopes.append(0.0)
            return pd.Series(slopes, index=group.index)
        
        df['conversion_gap_trend'] = df.groupby('experiment_id', group_keys=False).apply(
            compute_lift_slope
        )
        
        # Rolling conversion rates (3-day window)
        # We need to compute from daily data, not cumulative
        def compute_rolling_cr(group, col_conv, col_vis):
            """Compute 3-day rolling conversion rate from daily increments."""
            daily_conv = group[col_conv].diff().fillna(group[col_conv])
            daily_vis = group[col_vis].diff().fillna(group[col_vis])
            
            rolling_conv = daily_conv.rolling(window=3, min_periods=1).sum()
            rolling_vis = daily_vis.rolling(window=3, min_periods=1).sum()
            
            return np.where(rolling_vis > 0, rolling_conv / rolling_vis, 0)
        
        df['rolling_cr_control_3d'] = df.groupby('experiment_id', group_keys=False).apply(
            lambda g: pd.Series(
                compute_rolling_cr(g, 'conversions_control', 'visitors_control'),
                index=g.index
            )
        )
        
        df['rolling_cr_treatment_3d'] = df.groupby('experiment_id', group_keys=False).apply(
            lambda g: pd.Series(
                compute_rolling_cr(g, 'conversions_treatment', 'visitors_treatment'),
                index=g.index
            )
        )
        
        return df
    
    def compute_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistical uncertainty features.
        
        Features:
        - pooled_standard_error: SE of the lift estimate
        - z_statistic: current z-stat (evidence strength)
        - p_value_current: current two-sided p-value
        - lift_to_se_ratio: signal-to-noise ratio (lift / SE)
        """
        df = df.copy()
        
        n_c = df['visitors_control'].values.astype(float)
        n_t = df['visitors_treatment'].values.astype(float)
        x_c = df['conversions_control'].values.astype(float)
        x_t = df['conversions_treatment'].values.astype(float)
        
        p_c = np.where(n_c > 0, x_c / n_c, 0)
        p_t = np.where(n_t > 0, x_t / n_t, 0)
        
        # Pooled proportion
        p_pool = np.where(
            (n_c + n_t) > 0,
            (x_c + x_t) / (n_c + n_t),
            0
        )
        
        # Pooled standard error
        se = np.sqrt(
            np.where(
                p_pool * (1 - p_pool) * (1/np.maximum(n_c, 1) + 1/np.maximum(n_t, 1)) > 0,
                p_pool * (1 - p_pool) * (1/np.maximum(n_c, 1) + 1/np.maximum(n_t, 1)),
                1e-10
            )
        )
        
        df['pooled_standard_error'] = se
        
        # Z-statistic
        lift = p_t - p_c
        df['z_statistic'] = np.where(se > 1e-10, lift / se, 0)
        
        # P-value (two-sided)
        df['p_value_current'] = 2 * (1 - stats.norm.cdf(np.abs(df['z_statistic'])))
        
        # Signal-to-noise ratio
        df['lift_to_se_ratio'] = np.where(se > 1e-10, lift / se, 0)
        
        return df
    
    def compute_bayesian_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Bayesian features using closed-form Beta-Binomial posterior.
        
        Uses Beta(1, 1) prior (uniform) and computes:
        - bayesian_prob_treatment_wins: P(p_t > p_c | data)
        - bayesian_expected_lift: E[p_t - p_c | data]
        - credible_interval_width: Width of 95% credible interval for lift
        
        The probability P(p_t > p_c) is estimated via Monte Carlo sampling
        from the posterior Beta distributions.
        """
        df = df.copy()
        
        rng = np.random.RandomState(42)
        n_samples = 5000  # MC samples for posterior probability
        
        probs = []
        expected_lifts = []
        ci_widths = []
        
        for _, row in df.iterrows():
            # Beta posterior parameters (with Beta(1,1) prior)
            alpha_c = row['conversions_control'] + 1
            beta_c = row['visitors_control'] - row['conversions_control'] + 1
            alpha_t = row['conversions_treatment'] + 1
            beta_t = row['visitors_treatment'] - row['conversions_treatment'] + 1
            
            # Monte Carlo samples from posteriors
            samples_c = rng.beta(alpha_c, beta_c, n_samples)
            samples_t = rng.beta(alpha_t, beta_t, n_samples)
            
            lift_samples = samples_t - samples_c
            
            # P(treatment > control)
            prob_t_wins = np.mean(samples_t > samples_c)
            probs.append(prob_t_wins)
            
            # Expected lift
            expected_lifts.append(np.mean(lift_samples))
            
            # 95% credible interval width
            ci_lo, ci_hi = np.percentile(lift_samples, [2.5, 97.5])
            ci_widths.append(ci_hi - ci_lo)
        
        df['bayesian_prob_treatment_wins'] = probs
        df['bayesian_expected_lift'] = expected_lifts
        df['credible_interval_width'] = ci_widths
        
        return df
    
    def compute_sample_size_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sample size related features.
        
        Features:
        - total_visitors: cumulative visitors across both arms
        - sample_size_ratio: treatment / control visitors (balance)
        - sample_size_progress: already present from simulation
        """
        df = df.copy()
        
        df['total_visitors'] = df['visitors_control'] + df['visitors_treatment']
        
        df['sample_size_ratio'] = np.where(
            df['visitors_control'] > 0,
            df['visitors_treatment'] / df['visitors_control'],
            1.0
        )
        
        return df
    
    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical metadata columns."""
        df = df.copy()
        
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Return the list of feature column names for ML modeling.
        Excludes IDs, labels, raw counts, and metadata strings.
        """
        exclude_cols = {
            'experiment_id', 'day_number',
            'visitors_control', 'visitors_treatment',
            'conversions_control', 'conversions_treatment',
            'conversion_rate_control', 'conversion_rate_treatment',
            'observed_lift',
            'daily_visitors_this_day', 'daily_conversions_control', 'daily_conversions_treatment',
            'true_treatment_effect', 'effect_type',
            'final_outcome', 'final_p_value', 'final_z_statistic', 'final_lift',
            'winner_binary',
        }
        exclude_cols.update(self.CATEGORICAL_COLS)
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.feature_names = feature_cols
        return feature_cols
    
    def transform(self, df: pd.DataFrame, include_bayesian: bool = True) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        :param df: Raw daily snapshots dataframe
        :param include_bayesian: Whether to compute Bayesian features (slower)
        :return: DataFrame with all engineered features
        """
        print("Engineering features...")
        
        df = self.compute_conversion_features(df)
        print("  [OK] Conversion features")
        
        df = self.compute_statistical_features(df)
        print("  [OK] Statistical uncertainty features")
        
        df = self.compute_sample_size_features(df)
        print("  [OK] Sample size features")
        
        if include_bayesian:
            print("  Computing Bayesian posteriors (this may take a minute)...")
            df = self.compute_bayesian_features(df)
            print("  [OK] Bayesian features")
        
        df = self.encode_categoricals(df)
        print("  [OK] Categorical encoding")
        
        feature_cols = self.get_feature_columns(df)
        print(f"  [OK] Total features: {len(feature_cols)}")
        
        return df
