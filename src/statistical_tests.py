"""
Statistical Tests  --  Traditional A/B testing methods for benchmarking.

Implements two-proportion z-tests, p-value tracking across days,
and time-to-significance analysis.

Author: Sanman Kadam
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List


class StatisticalTester:
    """
    Provides traditional statistical testing for A/B experiments,
    used as a benchmark against the ML approach.
    """
    
    @staticmethod
    def two_proportion_ztest(
        n_control: int, x_control: int,
        n_treatment: int, x_treatment: int,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Two-proportion z-test for comparing conversion rates.
        
        H0: p_treatment = p_control
        H1: p_treatment != p_control (two-sided)
        
        :return: Dictionary with z-stat, p-value, CI, and decision
        """
        p_c = x_control / n_control if n_control > 0 else 0
        p_t = x_treatment / n_treatment if n_treatment > 0 else 0
        
        # Pooled proportion
        p_pool = (x_control + x_treatment) / (n_control + n_treatment)
        se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
        
        if se_pool > 0:
            z_stat = (p_t - p_c) / se_pool
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0.0
            p_value = 1.0
        
        # Confidence interval for the difference
        se_diff = np.sqrt(
            p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment
        )
        z_crit = stats.norm.ppf(1 - alpha / 2)
        diff = p_t - p_c
        ci_lower = diff - z_crit * se_diff
        ci_upper = diff + z_crit * se_diff
        
        is_significant = p_value < alpha
        
        if is_significant and diff > 0:
            decision = 'treatment_wins'
        elif is_significant and diff < 0:
            decision = 'control_wins'
        else:
            decision = 'inconclusive'
        
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'control_cr': p_c,
            'treatment_cr': p_t,
            'absolute_lift': diff,
            'relative_lift': diff / p_c if p_c > 0 else 0,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'se_pooled': se_pool,
            'is_significant': is_significant,
            'decision': decision,
        }
    
    @staticmethod
    def minimum_sample_size(
        baseline_cr: float, mde_relative: float,
        alpha: float = 0.05, power: float = 0.80
    ) -> int:
        """
        Calculate minimum sample size per arm for a two-proportion test.
        
        Uses the normal approximation formula.
        
        :param baseline_cr: Baseline conversion rate (e.g., 0.10)
        :param mde_relative: Minimum detectable effect as relative lift (e.g., 0.10 for 10%)
        :param alpha: Significance level
        :param power: Statistical power (1 - beta)
        :return: Required sample size per arm
        """
        p1 = baseline_cr
        p2 = baseline_cr * (1 + mde_relative)
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        p_bar = (p1 + p2) / 2
        
        n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2
        
        return int(np.ceil(n))
    
    def time_to_significance(
        self, df: pd.DataFrame,
        checkpoints: List[int] = None,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        For each experiment, determine when (if ever) the z-test reaches significance.
        
        :param df: Daily snapshots dataframe
        :param checkpoints: Days to evaluate (default: all days)
        :param alpha: Significance threshold
        :return: DataFrame with first_significant_day per experiment
        """
        if checkpoints is None:
            checkpoints = sorted(df['day_number'].unique())
        
        results = []
        
        for exp_id in df['experiment_id'].unique():
            exp_data = df[df['experiment_id'] == exp_id]
            first_sig_day = None
            first_decision = None
            
            for day in checkpoints:
                day_row = exp_data[exp_data['day_number'] == day]
                if day_row.empty:
                    continue
                row = day_row.iloc[0]
                
                test_result = self.two_proportion_ztest(
                    n_control=int(row['visitors_control']),
                    x_control=int(row['conversions_control']),
                    n_treatment=int(row['visitors_treatment']),
                    x_treatment=int(row['conversions_treatment']),
                    alpha=alpha
                )
                
                if test_result['is_significant'] and first_sig_day is None:
                    first_sig_day = day
                    first_decision = test_result['decision']
            
            results.append({
                'experiment_id': exp_id,
                'first_significant_day': first_sig_day,
                'first_decision': first_decision,
                'resolved_by_day14': first_sig_day is not None,
            })
        
        return pd.DataFrame(results)
    
    def benchmark_by_checkpoint(
        self, df: pd.DataFrame,
        checkpoints: List[int] = [1, 3, 5, 7, 10, 14],
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Calculate the percentage of experiments resolved by each checkpoint day.
        
        :return: DataFrame with checkpoint stats
        """
        tts = self.time_to_significance(df, alpha=alpha)
        total = len(tts)
        
        records = []
        for day in checkpoints:
            resolved = tts[tts['first_significant_day'].le(day)].shape[0]
            pct = resolved / total * 100
            records.append({
                'checkpoint_day': day,
                'experiments_resolved': resolved,
                'pct_resolved': round(pct, 1),
                'pct_unresolved': round(100 - pct, 1),
            })
        
        return pd.DataFrame(records)
