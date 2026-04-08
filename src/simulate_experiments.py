"""
Experiment Simulator  --  Generates realistic A/B test datasets.

Simulates 1,200 experiments, each running 14 days, with controlled
distributions of true treatment effects (positive, negative, null).

Author: Sanman Kadam
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class ExperimentSimulator:
    """
    Simulates realistic A/B testing experiments with daily snapshots.
    
    Each experiment has:
    - A baseline conversion rate drawn from U(0.03, 0.20)
    - A true treatment effect (40% positive, 30% negative, 30% null)
    - 14 daily snapshots of cumulative visitors and conversions
    - Metadata: device_type, region, traffic_source, segment, experiment_category
    """
    
    # Experiment metadata distributions
    DEVICE_TYPES = ['Mobile', 'Desktop', 'Tablet']
    DEVICE_PROBS = [0.45, 0.40, 0.15]
    
    REGIONS = ['US', 'EU', 'APAC', 'LATAM']
    REGION_PROBS = [0.40, 0.30, 0.20, 0.10]
    
    TRAFFIC_SOURCES = ['Organic', 'Paid', 'Email', 'Social']
    TRAFFIC_PROBS = [0.35, 0.30, 0.20, 0.15]
    
    SEGMENTS = ['New Users', 'Returning']
    SEGMENT_PROBS = [0.55, 0.45]
    
    CATEGORIES = ['Checkout', 'Landing Page', 'Pricing', 'CTA', 'Onboarding']
    CATEGORY_PROBS = [0.25, 0.25, 0.20, 0.15, 0.15]
    
    def __init__(self, n_experiments: int = 1200, n_days: int = 14, seed: int = 42):
        """
        :param n_experiments: Number of experiments to simulate
        :param n_days: Duration of each experiment in days
        :param seed: Random seed for reproducibility
        """
        self.n_experiments = n_experiments
        self.n_days = n_days
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _generate_experiment_metadata(self) -> pd.DataFrame:
        """Generate static metadata for each experiment."""
        metadata = pd.DataFrame({
            'experiment_id': range(1, self.n_experiments + 1),
            'baseline_conversion_rate': self.rng.uniform(0.03, 0.20, self.n_experiments),
            'daily_visitors_per_arm': self.rng.randint(200, 2001, self.n_experiments),
            'device_type': self.rng.choice(self.DEVICE_TYPES, self.n_experiments, p=self.DEVICE_PROBS),
            'region': self.rng.choice(self.REGIONS, self.n_experiments, p=self.REGION_PROBS),
            'traffic_source': self.rng.choice(self.TRAFFIC_SOURCES, self.n_experiments, p=self.TRAFFIC_PROBS),
            'segment': self.rng.choice(self.SEGMENTS, self.n_experiments, p=self.SEGMENT_PROBS),
            'experiment_category': self.rng.choice(self.CATEGORIES, self.n_experiments, p=self.CATEGORY_PROBS),
        })
        
        # Generate true treatment effects
        # 40% positive (0.5% to 5% absolute lift)
        # 30% negative (-0.5% to -5% absolute lift)
        # 30% null (0 effect)
        effect_type = self.rng.choice(
            ['positive', 'negative', 'null'],
            self.n_experiments,
            p=[0.40, 0.30, 0.30]
        )
        
        true_effect = np.zeros(self.n_experiments)
        for i in range(self.n_experiments):
            if effect_type[i] == 'positive':
                true_effect[i] = self.rng.uniform(0.005, 0.05)
            elif effect_type[i] == 'negative':
                true_effect[i] = self.rng.uniform(-0.05, -0.005)
            else:  # null
                true_effect[i] = 0.0
        
        metadata['true_treatment_effect'] = true_effect
        metadata['effect_type'] = effect_type
        
        return metadata
    
    def _simulate_daily_data(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Generate daily cumulative snapshots for each experiment.
        
        For each experiment and each day, draws conversions from a Binomial
        distribution and accumulates visitors and conversions.
        """
        records = []
        
        for _, exp in metadata.iterrows():
            exp_id = exp['experiment_id']
            base_cr = exp['baseline_conversion_rate']
            treatment_cr = np.clip(base_cr + exp['true_treatment_effect'], 0.001, 0.999)
            daily_visitors = exp['daily_visitors_per_arm']
            
            # Add small daily traffic variation (+/-15%)
            daily_variation = self.rng.uniform(0.85, 1.15, self.n_days)
            
            cum_visitors_c = 0
            cum_visitors_t = 0
            cum_conversions_c = 0
            cum_conversions_t = 0
            
            for day in range(1, self.n_days + 1):
                # Daily visitors with variation
                day_visitors = int(daily_visitors * daily_variation[day - 1])
                
                # Draw daily conversions from Binomial
                daily_conv_c = self.rng.binomial(day_visitors, base_cr)
                daily_conv_t = self.rng.binomial(day_visitors, treatment_cr)
                
                # Accumulate
                cum_visitors_c += day_visitors
                cum_visitors_t += day_visitors
                cum_conversions_c += daily_conv_c
                cum_conversions_t += daily_conv_t
                
                # Calculate rates
                cr_c = cum_conversions_c / cum_visitors_c if cum_visitors_c > 0 else 0
                cr_t = cum_conversions_t / cum_visitors_t if cum_visitors_t > 0 else 0
                observed_lift = cr_t - cr_c
                
                # Expected final visitors
                expected_final = daily_visitors * self.n_days
                sample_progress = (cum_visitors_c + cum_visitors_t) / (2 * expected_final)
                
                records.append({
                    'experiment_id': exp_id,
                    'day_number': day,
                    'visitors_control': cum_visitors_c,
                    'visitors_treatment': cum_visitors_t,
                    'conversions_control': cum_conversions_c,
                    'conversions_treatment': cum_conversions_t,
                    'conversion_rate_control': round(cr_c, 6),
                    'conversion_rate_treatment': round(cr_t, 6),
                    'observed_lift': round(observed_lift, 6),
                    'daily_visitors_this_day': day_visitors,
                    'daily_conversions_control': daily_conv_c,
                    'daily_conversions_treatment': daily_conv_t,
                    'sample_size_progress': round(sample_progress, 4),
                    # Metadata (repeated per row for convenience)
                    'device_type': exp['device_type'],
                    'region': exp['region'],
                    'traffic_source': exp['traffic_source'],
                    'segment': exp['segment'],
                    'experiment_category': exp['experiment_category'],
                    'baseline_conversion_rate': exp['baseline_conversion_rate'],
                    'true_treatment_effect': exp['true_treatment_effect'],
                    'effect_type': exp['effect_type'],
                })
        
        return pd.DataFrame(records)
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate final outcome labels from day-14 data using z-test.
        
        Labels:
        - treatment_wins: p < 0.05 and lift > 0
        - control_wins: p < 0.05 and lift < 0
        - inconclusive: p >= 0.05
        """
        from scipy import stats
        
        # Get day-14 data for each experiment
        day14 = df[df['day_number'] == self.n_days].copy()
        
        labels = {}
        for _, row in day14.iterrows():
            n_c = row['visitors_control']
            n_t = row['visitors_treatment']
            x_c = row['conversions_control']
            x_t = row['conversions_treatment']
            
            p_c = x_c / n_c
            p_t = x_t / n_t
            
            # Pooled proportion z-test
            p_pool = (x_c + x_t) / (n_c + n_t)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
            
            if se > 0:
                z = (p_t - p_c) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # two-sided
            else:
                z = 0
                p_value = 1.0
            
            lift = p_t - p_c
            
            if p_value < 0.05 and lift > 0:
                label = 'treatment_wins'
            elif p_value < 0.05 and lift < 0:
                label = 'control_wins'
            else:
                label = 'inconclusive'
            
            labels[row['experiment_id']] = {
                'final_outcome': label,
                'final_p_value': p_value,
                'final_z_statistic': z,
                'final_lift': lift,
                'winner_binary': 1 if label == 'treatment_wins' else 0,
            }
        
        labels_df = pd.DataFrame.from_dict(labels, orient='index')
        labels_df.index.name = 'experiment_id'
        labels_df = labels_df.reset_index()
        
        return labels_df
    
    def simulate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the full simulation pipeline.
        
        Returns:
            df: Daily snapshots (n_experiments * n_days rows)
            labels: Final outcome labels (n_experiments rows)
        """
        print(f"Simulating {self.n_experiments} experiments over {self.n_days} days...")
        
        # Step 1: Generate metadata
        metadata = self._generate_experiment_metadata()
        print(f"  [OK] Generated experiment metadata")
        print(f"    Effect distribution: {dict(metadata['effect_type'].value_counts())}")
        
        # Step 2: Generate daily data
        df = self._simulate_daily_data(metadata)
        print(f"  [OK] Generated daily snapshots: {len(df)} rows")
        
        # Step 3: Generate labels
        labels = self._generate_labels(df)
        print(f"  [OK] Generated final outcome labels:")
        print(f"    {dict(labels['final_outcome'].value_counts())}")
        
        # Step 4: Merge labels into daily data
        df = df.merge(labels, on='experiment_id', how='left')
        
        return df, labels
    
    def save(self, df: pd.DataFrame, labels: pd.DataFrame, 
             data_dir: str = 'data') -> None:
        """Save datasets to CSV."""
        import os
        os.makedirs(data_dir, exist_ok=True)
        
        df.to_csv(f'{data_dir}/simulated_experiments.csv', index=False)
        labels.to_csv(f'{data_dir}/experiment_labels.csv', index=False)
        print(f"  [OK] Saved to {data_dir}/")


if __name__ == '__main__':
    sim = ExperimentSimulator(n_experiments=1200, seed=42)
    df, labels = sim.simulate()
    sim.save(df, labels)
    print(f"\nDataset shape: {df.shape}")
    print(f"Labels shape: {labels.shape}")
