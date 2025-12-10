"""
build_hetero_graph.py -- Bipartite Graph Construction for Site-Vendor Link Prediction

Converts CSV data (sites, vendors, contracts) to PyTorch Geometric HeteroData format
for use with Graph Neural Networks.

Author: Gregory E. Schwartz
Last Revised: December 2025
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np
from pathlib import Path


class BipartiteGraphBuilder:
    """
    Build bipartite graph from CSV files

    Graph Structure:
        - Site nodes (N=100): dental practices with features
        - Vendor nodes (N=20): vendors with features
        - Edges: historical contracts (2019-2024)

    Node Features:
        - Sites: region (one-hot), EHR (one-hot), revenue (numeric)
        - Vendors: category (one-hot), tier (numeric), price (numeric)
    """

    def __init__(self):
        self.site_id_to_idx = {}   # Map site_id (e.g., 'S001') → 0, 1, 2, ...
        self.vendor_id_to_idx = {} # Map vendor_id (e.g., 'V001') → 0, 1, 2, ...

        # Feature encoding maps
        self.region_map = {'South': 0, 'Midwest': 1, 'Northeast': 2, 'West': 3}
        self.ehr_map = {'Dentrix': 0, 'OpenDental': 1, 'Eaglesoft': 2, 'Curve': 3, 'Denticon': 4}
        self.category_map = {
            'Lab': 0, 'RCM': 1, 'Telephony': 2, 'Scheduling': 3,
            'Clearinghouse': 4, 'IT_MSP': 5, 'Supplies': 6
        }

    def build_graph(self, sites_path, vendors_path, contracts_path, integration_path=None):
        """
        Build bipartite graph from CSV files

        Args:
            sites_path: Path to sites.csv
            vendors_path: Path to vendors.csv
            contracts_path: Path to contracts_2019_2024.csv
            integration_path: Optional path to integration_matrix.csv

        Returns:
            HeteroData: PyG bipartite graph with:
                - data['site'].x: Site node features [N_sites, site_feat_dim]
                - data['vendor'].x: Vendor node features [N_vendors, vendor_feat_dim]
                - data['site', 'contracts', 'vendor'].edge_index: Edge connections [2, N_edges]
        """
        print("="*60)
        print("BUILDING BIPARTITE GRAPH")
        print("="*60)

        # Load data
        print("\n1. Loading CSV files...")
        sites_df = pd.read_csv(sites_path)
        vendors_df = pd.read_csv(vendors_path)
        contracts_df = pd.read_csv(contracts_path)

        print(f"   - Sites: {len(sites_df)} rows")
        print(f"   - Vendors: {len(vendors_df)} rows")
        print(f"   - Contracts: {len(contracts_df)} rows")

        # Create HeteroData object
        data = HeteroData()

        # 2. Build site nodes
        print("\n2. Encoding site node features...")
        data['site'].x, self.site_id_to_idx = self._encode_site_features(sites_df)
        print(f"   - Site nodes: {data['site'].x.shape[0]}")
        print(f"   - Site feature dim: {data['site'].x.shape[1]}")

        # 3. Build vendor nodes
        print("\n3. Encoding vendor node features...")
        data['vendor'].x, self.vendor_id_to_idx = self._encode_vendor_features(vendors_df)
        print(f"   - Vendor nodes: {data['vendor'].x.shape[0]}")
        print(f"   - Vendor feature dim: {data['vendor'].x.shape[1]}")

        # 4. Build edges from contracts
        print("\n4. Building edge index from contracts...")
        data['site', 'contracts', 'vendor'].edge_index = self._build_edge_index(
            contracts_df, self.site_id_to_idx, self.vendor_id_to_idx
        )
        print(f"   - Edges: {data['site', 'contracts', 'vendor'].edge_index.shape[1]}")

        # 5. Optional: Add edge features (integration quality)
        if integration_path:
            print("\n5. Adding edge features (integration quality)...")
            integration_df = pd.read_csv(integration_path)
            data['site', 'contracts', 'vendor'].edge_attr = self._build_edge_features(
                contracts_df, integration_df, self.site_id_to_idx, self.vendor_id_to_idx
            )
            print(f"   - Edge features shape: {data['site', 'contracts', 'vendor'].edge_attr.shape}")

        print("\n" + "="*60)
        print("✅ GRAPH CONSTRUCTION COMPLETE")
        print("="*60)

        return data

    def _encode_site_features(self, sites_df):
        """
        Convert site DataFrame to feature tensor

        Features (10 dims total):
            - Region one-hot (4 dims): South, Midwest, Northeast, West
            - EHR one-hot (5 dims): Dentrix, OpenDental, Eaglesoft, Curve, Denticon
            - Annual revenue normalized (1 dim): revenue / 3M

        Returns:
            features: torch.Tensor [N_sites, 10]
            site_id_to_idx: dict mapping site_id -> index
        """
        site_id_to_idx = {sid: i for i, sid in enumerate(sites_df['site_id'])}

        features = []
        for _, site in sites_df.iterrows():
            # One-hot encode region (4 dims)
            region_onehot = [0, 0, 0, 0]
            if site['region'] in self.region_map:
                region_onehot[self.region_map[site['region']]] = 1

            # One-hot encode EHR system (5 dims)
            ehr_onehot = [0, 0, 0, 0, 0]
            if site['ehr_system'] in self.ehr_map:
                ehr_onehot[self.ehr_map[site['ehr_system']]] = 1

            # Normalize revenue (1 dim)
            # Assume max revenue ~3M based on data
            revenue_norm = site['annual_revenue'] / 3_000_000.0
            revenue_norm = min(revenue_norm, 1.0)  # Clip to [0, 1]

            # Combine: 4 + 5 + 1 = 10 dims
            feature_vec = region_onehot + ehr_onehot + [revenue_norm]
            features.append(feature_vec)

        return torch.tensor(features, dtype=torch.float), site_id_to_idx

    def _encode_vendor_features(self, vendors_df):
        """
        Convert vendor DataFrame to feature tensor

        Features (9 dims total):
            - Category one-hot (7 dims): Lab, RCM, Telephony, Scheduling, Clearinghouse, IT_MSP, Supplies
            - Tier normalized (1 dim): tier / 3.0
            - Price normalized (1 dim): monthly_price / 15000

        Returns:
            features: torch.Tensor [N_vendors, 9]
            vendor_id_to_idx: dict mapping vendor_id -> index
        """
        vendor_id_to_idx = {vid: i for i, vid in enumerate(vendors_df['vendor_id'])}

        features = []
        for _, vendor in vendors_df.iterrows():
            # One-hot encode category (7 dims)
            cat_onehot = [0, 0, 0, 0, 0, 0, 0]

            # Handle category (check if column exists and has valid value)
            if 'category' in vendor.index:
                category = vendor['category']
                if pd.notna(category) and category in self.category_map:
                    cat_onehot[self.category_map[category]] = 1

            # Normalize tier (1 dim)
            # Assume tiers are 1, 2, or 3
            tier_norm = vendor.get('tier', 2) / 3.0

            # Normalize monthly price (1 dim)
            # Assume max price ~15K per site per month
            price = vendor.get('monthly_price_per_site', 5000)
            price_norm = price / 15_000.0
            price_norm = min(price_norm, 1.0)  # Clip to [0, 1]

            # Combine: 7 + 1 + 1 = 9 dims
            feature_vec = cat_onehot + [tier_norm, price_norm]
            features.append(feature_vec)

        return torch.tensor(features, dtype=torch.float), vendor_id_to_idx

    def _build_edge_index(self, contracts_df, site_map, vendor_map):
        """
        Build edge_index tensor from contracts

        Args:
            contracts_df: DataFrame with columns [site_id, vendor_id, ...]
            site_map: dict mapping site_id -> node index
            vendor_map: dict mapping vendor_id -> node index

        Returns:
            edge_index: torch.Tensor [2, N_edges]
                - edge_index[0, :] = source site indices
                - edge_index[1, :] = target vendor indices
        """
        edge_list = []
        skipped = 0

        for _, contract in contracts_df.iterrows():
            site_id = contract['site_id']
            vendor_id = contract['vendor_id']

            # Skip if IDs not in mapping (shouldn't happen but safety check)
            if site_id not in site_map:
                skipped += 1
                continue
            if vendor_id not in vendor_map:
                skipped += 1
                continue

            site_idx = site_map[site_id]
            vendor_idx = vendor_map[vendor_id]
            edge_list.append([site_idx, vendor_idx])

        if skipped > 0:
            print(f"   WARNING: Skipped {skipped} contracts due to missing IDs")

        # Convert to tensor [2, num_edges]
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return edge_index

    def _build_edge_features(self, contracts_df, integration_df, site_map, vendor_map):
        """
        Build edge feature matrix (integration quality per edge)

        Args:
            contracts_df: Contracts with (site_id, vendor_id)
            integration_df: Integration matrix with (site_id, vendor_id, integration_quality)
            site_map: site_id -> index mapping
            vendor_map: vendor_id -> index mapping

        Returns:
            edge_attr: torch.Tensor [N_edges, 1] with integration quality {0, 1, 2}
        """
        # Create lookup dict: (site_id, vendor_id) -> integration_quality
        integration_lookup = {}
        for _, row in integration_df.iterrows():
            key = (row['site_id'], row['vendor_id'])
            integration_lookup[key] = row['integration_quality']

        # Build edge features in same order as edge_index
        edge_features = []
        for _, contract in contracts_df.iterrows():
            site_id = contract['site_id']
            vendor_id = contract['vendor_id']

            # Skip if not in mappings
            if site_id not in site_map or vendor_id not in vendor_map:
                continue

            # Lookup integration quality (default to 1 if not found)
            key = (site_id, vendor_id)
            integration_quality = integration_lookup.get(key, 1)

            # Normalize to [0, 1]: {0, 1, 2} -> {0, 0.5, 1}
            integration_norm = integration_quality / 2.0

            edge_features.append([integration_norm])

        return torch.tensor(edge_features, dtype=torch.float)


def main():
    """
    Test the graph builder with actual data
    """
    # Define paths (relative to v0.section3.model_architecture)
    base_path = Path(__file__).parent.parent.parent.parent / "v0.section2.data_design" / "2_synthetic_practice_data"

    sites_path = base_path / "sites.csv"
    vendors_path = base_path / "vendors.csv"
    contracts_path = base_path / "contracts_2019_2024.csv"
    integration_path = base_path / "integration_matrix.csv"

    # Build graph
    builder = BipartiteGraphBuilder()
    graph = builder.build_graph(
        sites_path=sites_path,
        vendors_path=vendors_path,
        contracts_path=contracts_path,
        integration_path=integration_path
    )

    # Print summary
    print("\n" + "="*60)
    print("GRAPH SUMMARY")
    print("="*60)
    print(f"Site nodes: {graph['site'].x.shape}")
    print(f"Vendor nodes: {graph['vendor'].x.shape}")
    print(f"Edges: {graph['site', 'contracts', 'vendor'].edge_index.shape}")
    if hasattr(graph['site', 'contracts', 'vendor'], 'edge_attr'):
        print(f"Edge features: {graph['site', 'contracts', 'vendor'].edge_attr.shape}")
    print("="*60)

    return graph


if __name__ == "__main__":
    graph = main()
