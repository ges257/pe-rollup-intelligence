"""
Peer Count Scorer

A heuristic that scores site×vendor pairs based on peer adoption patterns.

Key Insight: Sites tend to adopt vendors that similar sites (peers) have adopted.

Peer Definition:
- Sites with the same EHR system AND same region

Formula:
    score = peer_adoptions / (total_peers + 1)

Where:
- peer_adoptions: Number of peer sites that adopted this vendor in the last 12 months
- total_peers: Total number of peer sites
- +1 in denominator prevents division by zero

Expected Performance: PR-AUC ~0.40-0.50 (peer effects are strong signal)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class PeerCountScorer:
    """
    Heuristic scorer based on peer adoption patterns
    """

    def __init__(self, data_dir, historical_contracts_df, lookback_months=12):
        """
        Initialize scorer

        Parameters:
        -----------
        data_dir : Path or str
            Path to Section 2 data directory containing sites.csv
        historical_contracts_df : pd.DataFrame
            Historical contracts used to compute peer adoptions
            Should include: site_id, vendor_id, contract_start_date
        lookback_months : int
            How many months back to count peer adoptions (default: 12)
        """
        self.data_dir = Path(data_dir)
        self.historical_contracts = historical_contracts_df
        self.lookback_months = lookback_months
        self.sites = None

        self._load_data()
        self._precompute_peer_groups()

    def _load_data(self):
        """Load sites data"""
        print("Loading data for Peer Count Scorer...")

        self.sites = pd.read_csv(self.data_dir / "sites.csv")
        print(f"  ✓ Loaded {len(self.sites)} sites")

        # Convert contract dates
        if 'contract_start_date' in self.historical_contracts.columns:
            self.historical_contracts['contract_start_date'] = pd.to_datetime(
                self.historical_contracts['contract_start_date']
            )

    def _precompute_peer_groups(self):
        """
        Precompute peer groups for each site

        Peers are defined as sites with:
        - Same EHR system
        - Same region
        """
        print("  Computing peer groups...")

        self.peer_groups = {}

        for _, site in self.sites.iterrows():
            site_id = site['site_id']

            # Find peers: same EHR + same region
            peers = self.sites[
                (self.sites['ehr_system'] == site['ehr_system']) &
                (self.sites['region'] == site['region']) &
                (self.sites['site_id'] != site_id)  # Exclude self
            ]

            self.peer_groups[site_id] = set(peers['site_id'].values)

        avg_peer_count = np.mean([len(peers) for peers in self.peer_groups.values()])
        print(f"  ✓ Average peers per site: {avg_peer_count:.1f}")

    def _count_peer_adoptions(self, site_id, vendor_id, reference_date=None):
        """
        Count how many peers adopted this vendor in the last N months

        Parameters:
        -----------
        site_id : str
        vendor_id : str
        reference_date : datetime or None
            Date to count backwards from (if None, use latest contract date)

        Returns:
        --------
        peer_adoption_count : int
            Number of peer sites that adopted vendor in lookback window
        """
        # Get peers
        peers = self.peer_groups.get(site_id, set())

        if len(peers) == 0:
            return 0

        # Set reference date
        if reference_date is None:
            reference_date = self.historical_contracts['contract_start_date'].max()

        # Define lookback window
        cutoff_date = reference_date - timedelta(days=self.lookback_months * 30)

        # Count peer adoptions in window
        peer_adoptions = self.historical_contracts[
            (self.historical_contracts['site_id'].isin(peers)) &
            (self.historical_contracts['vendor_id'] == vendor_id) &
            (self.historical_contracts['contract_start_date'] >= cutoff_date) &
            (self.historical_contracts['contract_start_date'] <= reference_date)
        ]

        return len(peer_adoptions)

    def score_pair(self, site_id, vendor_id, reference_date=None):
        """
        Score a single site×vendor pair

        Parameters:
        -----------
        site_id : str
        vendor_id : str
        reference_date : datetime or None
            Date to evaluate at (default: latest)

        Returns:
        --------
        score : float
            Peer adoption rate in [0, 1]
        """
        # Get peer group size
        peers = self.peer_groups.get(site_id, set())
        total_peers = len(peers)

        if total_peers == 0:
            # No peers - return neutral score
            return 0.0

        # Count peer adoptions
        peer_adoptions = self._count_peer_adoptions(site_id, vendor_id, reference_date)

        # Compute score
        score = peer_adoptions / (total_peers + 1)  # +1 to prevent extreme scores

        return score

    def score_dataframe(self, pairs_df):
        """
        Score a DataFrame of site×vendor pairs

        Parameters:
        -----------
        pairs_df : pd.DataFrame
            DataFrame with columns: site_id, vendor_id
            Optional: year (to set reference date)

        Returns:
        --------
        scores : np.array
            Array of scores, same length as pairs_df
        """
        scores = []

        for _, row in pairs_df.iterrows():
            # Set reference date based on year if available
            reference_date = None
            if 'year' in row:
                reference_date = datetime(int(row['year']), 12, 31)

            score = self.score_pair(row['site_id'], row['vendor_id'], reference_date)
            scores.append(score)

        return np.array(scores)

    def predict_proba(self, pairs_df):
        """
        Predict probabilities for pairs (alias for score_dataframe)

        Parameters:
        -----------
        pairs_df : pd.DataFrame
            DataFrame with columns: site_id, vendor_id

        Returns:
        --------
        scores : np.array
            Array of predicted probabilities in [0, 1]
        """
        return self.score_dataframe(pairs_df)


def main():
    """Test the Peer Count Scorer"""
    print("="*60)
    print("TESTING PEER COUNT SCORER")
    print("="*60)

    # Paths
    data_dir = Path("/home/g12/pa_final_project_fall25/v0.section2.data_design/2_synthetic_practice_data")
    splits_dir = Path("/home/g12/pa_final_project_fall25/v0.section3.model_architecture/data_splits")

    # Load historical contracts (from original contracts file - has dates!)
    print("\nLoading historical contracts (for peer adoption history)...")
    contracts_df = pd.read_csv(data_dir / "contracts_2019_2024.csv")
    contracts_df['contract_start_date'] = pd.to_datetime(contracts_df['contract_start_date'])

    # NOTE: For heuristics, we use ALL contracts (2019-2024) to count peer patterns
    # This is NOT data leakage because heuristics don't "train" - they just count
    # historical patterns. We're looking BACKWARDS from each prediction point.
    print(f"  ✓ Loaded {len(contracts_df)} historical contracts (2019-2024)")

    # Initialize scorer
    scorer = PeerCountScorer(
        data_dir=data_dir,
        historical_contracts_df=contracts_df,
        lookback_months=12
    )

    # Load dev set
    print("\nLoading dev set...")
    dev_df = pd.read_csv(splits_dir / "dev_2023_2024.csv")
    print(f"  ✓ Loaded {len(dev_df)} pairs")

    # Score dev set
    print("\nScoring dev set...")
    scores = scorer.predict_proba(dev_df)
    print(f"  ✓ Generated {len(scores)} scores")

    # Statistics
    print("\n" + "="*60)
    print("SCORE STATISTICS")
    print("="*60)
    print(f"  Mean score: {scores.mean():.3f}")
    print(f"  Std score: {scores.std():.3f}")
    print(f"  Min score: {scores.min():.3f}")
    print(f"  Max score: {scores.max():.3f}")

    # Score distribution by label
    if 'label' in dev_df.columns:
        pos_scores = scores[dev_df['label'] == 1]
        neg_scores = scores[dev_df['label'] == 0]

        print(f"\n  Positive examples (label=1):")
        print(f"    Mean: {pos_scores.mean():.3f}")
        print(f"    Std: {pos_scores.std():.3f}")

        print(f"\n  Negative examples (label=0):")
        print(f"    Mean: {neg_scores.mean():.3f}")
        print(f"    Std: {neg_scores.std():.3f}")

        print(f"\n  Separation (pos - neg): {pos_scores.mean() - neg_scores.mean():.3f}")

    # Non-zero scores
    nonzero_scores = scores[scores > 0]
    print(f"\n  Non-zero scores: {len(nonzero_scores)} / {len(scores)} ({len(nonzero_scores)/len(scores)*100:.1f}%)")

    print("\n✓ Peer Count Scorer test complete!")


if __name__ == "__main__":
    main()
