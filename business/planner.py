"""
Greedy Planner: Convert Model Predictions to Quarterly Action Plan

PURPOSE:
    Convert R-GCN link predictions + Risk Head outputs into a ranked,
    staged quarterly plan for vendor consolidation.

ALGORITHM:
    1. Score each recommendation: benefit = p × price_delta + kpi_bonus
    2. Adjust by risk: score = benefit - risk_penalty
    3. Rank recommendations by adjusted score
    4. Stage by risk: low-risk → Q1, high-risk → Q3/Q4
    5. Enforce constraints (EHR+Lab rule, vendor capacity)
    6. Output quarterly assignments

CONSTRAINTS:
    1. EHR + Lab rule: Can't switch both EHR and Lab in same quarter
    2. Vendor capacity: Max N sites per vendor per quarter
    3. Site capacity: Max M switches per site per year
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "v0.section3.model_architecture" / "tier4_bipartite_gnn" / "graph_construction"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "v0.section3.model_architecture" / "tier4_bipartite_gnn" / "risk_head"))


class Recommendation:
    """Single recommendation for (site, vendor) pair"""

    def __init__(self, site_id, vendor_id, category, p_adoption, days_ar_delta,
                 current_vendor_price, new_vendor_price, integration_quality):
        self.site_id = site_id
        self.vendor_id = vendor_id
        self.category = category
        self.p_adoption = p_adoption
        self.days_ar_delta = days_ar_delta
        self.current_vendor_price = current_vendor_price
        self.new_vendor_price = new_vendor_price
        self.integration_quality = integration_quality

        # Computed fields
        self.price_delta = current_vendor_price - new_vendor_price  # Positive = savings
        self.benefit = self._compute_benefit()
        self.risk_label = self._get_risk_label()
        self.risk_score = self._get_risk_score()
        self.adjusted_score = self._compute_adjusted_score()
        self.quarter = None  # Assigned during planning

    def _compute_benefit(self):
        """
        Compute expected benefit

        benefit = p_adoption × (price_savings + kpi_bonus + tier_bonus)
        """
        # Price savings (normalized to $0-10k range)
        price_benefit = self.price_delta / 1000.0  # Convert to thousands

        # KPI bonus based on expected Days-A/R improvement
        # Negative delta = improvement = bonus
        kpi_bonus = max(0, -self.days_ar_delta) * 0.5  # $500 per day improved

        # Tier bonus for high-integration vendors
        tier_bonus = self.integration_quality * 1.0  # Up to $2000 for full API

        # Total benefit weighted by adoption probability
        total_benefit = self.p_adoption * (price_benefit + kpi_bonus + tier_bonus)

        return total_benefit

    def _get_risk_label(self):
        """Convert Days-A/R delta to risk label"""
        if self.days_ar_delta < -2:
            return 'Green'  # Significant improvement
        elif self.days_ar_delta > 2:
            return 'Red'  # Degradation
        else:
            return 'Amber'  # Neutral

    def _get_risk_score(self):
        """Convert Days-A/R delta to risk score (0-100)"""
        # Clip to ±10 days
        delta = max(-10, min(10, self.days_ar_delta))
        # Map [-10, 10] to [100, 0]
        return 50 - (delta / 10) * 50

    def _compute_adjusted_score(self):
        """Adjust benefit score by risk"""
        # Risk penalty: Red recommendations get penalty
        if self.risk_label == 'Red':
            risk_penalty = 0.5 * abs(self.days_ar_delta)
        elif self.risk_label == 'Amber':
            risk_penalty = 0.1 * abs(self.days_ar_delta)
        else:
            risk_penalty = 0  # Green = no penalty

        return self.benefit - risk_penalty

    def to_dict(self):
        """Convert to dictionary for DataFrame"""
        return {
            'site_id': self.site_id,
            'vendor_id': self.vendor_id,
            'category': self.category,
            'p_adoption': self.p_adoption,
            'days_ar_delta': self.days_ar_delta,
            'price_delta': self.price_delta,
            'integration_quality': self.integration_quality,
            'benefit': self.benefit,
            'risk_label': self.risk_label,
            'risk_score': self.risk_score,
            'adjusted_score': self.adjusted_score,
            'quarter': self.quarter
        }


class GreedyPlanner:
    """
    Greedy planner for vendor consolidation

    Assigns recommendations to quarters based on:
    1. Adjusted score (benefit - risk)
    2. Risk-based staging (low-risk → Q1, high-risk → Q3/Q4)
    3. Constraint satisfaction
    """

    def __init__(self, max_switches_per_site=3, max_sites_per_vendor_per_quarter=20):
        self.max_switches_per_site = max_switches_per_site
        self.max_sites_per_vendor_per_quarter = max_sites_per_vendor_per_quarter

    def plan(self, recommendations):
        """
        Create quarterly plan from recommendations

        Args:
            recommendations: List of Recommendation objects

        Returns:
            List of Recommendation objects with quarter assignments
        """
        # Sort by adjusted score (descending)
        sorted_recs = sorted(recommendations, key=lambda r: r.adjusted_score, reverse=True)

        # Track constraints
        site_switches = {}  # site_id -> count
        vendor_quarter_count = {}  # (vendor_id, quarter) -> count
        site_quarter_category = {}  # (site_id, quarter) -> set of categories

        # Assign quarters
        for rec in sorted_recs:
            # Find best quarter based on risk
            quarter = self._assign_quarter(rec, site_switches, vendor_quarter_count,
                                            site_quarter_category)
            rec.quarter = quarter

            # Update constraints
            if quarter is not None:
                site_switches[rec.site_id] = site_switches.get(rec.site_id, 0) + 1
                key = (rec.vendor_id, quarter)
                vendor_quarter_count[key] = vendor_quarter_count.get(key, 0) + 1

                sq_key = (rec.site_id, quarter)
                if sq_key not in site_quarter_category:
                    site_quarter_category[sq_key] = set()
                site_quarter_category[sq_key].add(rec.category)

        return sorted_recs

    def _assign_quarter(self, rec, site_switches, vendor_quarter_count, site_quarter_category):
        """
        Assign quarter to recommendation based on risk and constraints

        Risk-based staging:
        - Green (low risk): Q1 or Q2
        - Amber (medium risk): Q2 or Q3
        - Red (high risk): Q3 or Q4
        """
        # Determine preferred quarters based on risk
        if rec.risk_label == 'Green':
            preferred_quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        elif rec.risk_label == 'Amber':
            preferred_quarters = ['Q2', 'Q3', 'Q4', 'Q1']
        else:  # Red
            preferred_quarters = ['Q3', 'Q4', 'Q2', 'Q1']

        # Check site capacity
        if site_switches.get(rec.site_id, 0) >= self.max_switches_per_site:
            return None  # Site has too many switches this year

        # Find first valid quarter
        for quarter in preferred_quarters:
            # Check vendor capacity
            key = (rec.vendor_id, quarter)
            if vendor_quarter_count.get(key, 0) >= self.max_sites_per_vendor_per_quarter:
                continue

            # Check EHR+Lab constraint
            sq_key = (rec.site_id, quarter)
            current_categories = site_quarter_category.get(sq_key, set())

            # Can't switch both EHR and Lab in same quarter
            if rec.category == 'EHR' and 'Lab' in current_categories:
                continue
            if rec.category == 'Lab' and 'EHR' in current_categories:
                continue

            return quarter

        return 'Q4'  # Fallback to Q4 if no valid quarter found


def generate_synthetic_recommendations(sites_df, vendors_df, n_recommendations=50):
    """
    Generate synthetic recommendations for testing

    In production, these would come from the R-GCN model
    """
    np.random.seed(42)

    recommendations = []
    categories = vendors_df['category'].unique()

    for _ in range(n_recommendations):
        site = sites_df.sample(1).iloc[0]
        vendor = vendors_df.sample(1).iloc[0]

        # Synthetic model outputs
        p_adoption = np.random.beta(3, 2)  # Skewed toward higher probabilities
        days_ar_delta = np.random.normal(-2, 4)  # Mean improvement of 2 days

        # Prices
        current_price = np.random.uniform(500, 2000)
        new_price = vendor['price'] if 'price' in vendor else np.random.uniform(300, 1500)

        # Integration quality
        integration = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])

        rec = Recommendation(
            site_id=site['site_id'],
            vendor_id=vendor['vendor_id'],
            category=vendor['category'],
            p_adoption=p_adoption,
            days_ar_delta=days_ar_delta,
            current_vendor_price=current_price,
            new_vendor_price=new_price,
            integration_quality=integration
        )
        recommendations.append(rec)

    return recommendations


def main():
    print("="*70)
    print("GREEDY PLANNER: QUARTERLY VENDOR CONSOLIDATION PLAN")
    print("="*70)

    # Load data
    print("\n1. Loading site and vendor data...")
    base_path = Path(__file__).parent.parent.parent / "v0.section2.data_design" / "2_synthetic_practice_data"

    sites_df = pd.read_csv(base_path / "sites.csv")
    vendors_df = pd.read_csv(base_path / "vendors.csv")

    print(f"   Sites: {len(sites_df)}")
    print(f"   Vendors: {len(vendors_df)}")

    # Generate recommendations (in production, these come from R-GCN)
    print("\n2. Generating recommendations...")
    recommendations = generate_synthetic_recommendations(sites_df, vendors_df, n_recommendations=100)
    print(f"   Generated {len(recommendations)} recommendations")

    # Create planner
    print("\n3. Running greedy planner...")
    planner = GreedyPlanner(
        max_switches_per_site=3,
        max_sites_per_vendor_per_quarter=20
    )

    planned_recs = planner.plan(recommendations)

    # Create output DataFrame
    plan_df = pd.DataFrame([rec.to_dict() for rec in planned_recs])

    # Summary statistics
    print("\n" + "="*50)
    print("PLAN SUMMARY")
    print("="*50)

    # By quarter
    print("\n   By Quarter:")
    quarter_summary = plan_df.groupby('quarter').agg({
        'site_id': 'count',
        'adjusted_score': 'mean',
        'price_delta': 'sum'
    }).rename(columns={
        'site_id': 'count',
        'adjusted_score': 'avg_score',
        'price_delta': 'total_savings'
    })
    print(quarter_summary.to_string())

    # By risk
    print("\n   By Risk Label:")
    risk_summary = plan_df.groupby('risk_label').agg({
        'site_id': 'count',
        'adjusted_score': 'mean'
    }).rename(columns={'site_id': 'count', 'adjusted_score': 'avg_score'})
    print(risk_summary.to_string())

    # By category
    print("\n   By Category:")
    category_summary = plan_df.groupby('category').agg({
        'site_id': 'count',
        'p_adoption': 'mean'
    }).rename(columns={'site_id': 'count', 'p_adoption': 'avg_adoption_prob'})
    print(category_summary.to_string())

    # Risk-based staging verification
    print("\n   Risk-Based Staging Check:")
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = plan_df[plan_df['quarter'] == quarter]
        if len(q_data) > 0:
            green_pct = (q_data['risk_label'] == 'Green').mean() * 100
            amber_pct = (q_data['risk_label'] == 'Amber').mean() * 100
            red_pct = (q_data['risk_label'] == 'Red').mean() * 100
            print(f"   {quarter}: Green={green_pct:.0f}%, Amber={amber_pct:.0f}%, Red={red_pct:.0f}%")

    # Save outputs
    print("\n4. Saving plan...")
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    plan_df.to_csv(output_dir / "plan_raw.csv", index=False)

    # Summary by quarter
    plan_df.to_csv(output_dir / "quarterly_plan.csv", index=False)

    # Constraint violations
    violations = []

    # Check EHR+Lab violations
    for (site_id, quarter), group in plan_df.groupby(['site_id', 'quarter']):
        categories = set(group['category'])
        if 'EHR' in categories and 'Lab' in categories:
            violations.append({
                'type': 'EHR+Lab',
                'site_id': site_id,
                'quarter': quarter
            })

    # Check vendor capacity violations
    for (vendor_id, quarter), group in plan_df.groupby(['vendor_id', 'quarter']):
        if len(group) > planner.max_sites_per_vendor_per_quarter:
            violations.append({
                'type': 'VendorCapacity',
                'vendor_id': vendor_id,
                'quarter': quarter,
                'count': len(group)
            })

    print(f"\n   Constraint violations: {len(violations)}")
    if violations:
        print("   Violations:")
        for v in violations:
            print(f"     - {v}")

    print(f"\n   plan_raw.csv saved ({len(plan_df)} recommendations)")
    print(f"   quarterly_plan.csv saved")

    print("\n" + "="*70)
    print("GREEDY PLANNER COMPLETE")
    print("="*70)

    return plan_df, violations


if __name__ == "__main__":
    plan_df, violations = main()
