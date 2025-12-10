"""
Bipartite Graph Neural Network for Site-Vendor Link Prediction

Architecture:
    1. BipartiteGNNEncoder: 2-layer SAGEConv message passing
    2. MLPDecoder: 3-layer MLP for edge prediction
    3. BipartiteGNN: Full model combining encoder + decoder

References:
    - Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs" (GraphSAGE)
    - PyTorch Geometric SAGEConv: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Linear
from torch_geometric.data import HeteroData


class BipartiteGNNEncoder(nn.Module):
    """
    2-layer GraphSAGE encoder for bipartite graph

    Architecture:
        Input projection: Linear layers to project site (10) and vendor (9) features to hidden_dim
        Layer 1: SAGEConv(hidden_dim → hidden_dim) + ReLU + Dropout
        Layer 2: SAGEConv(hidden_dim → out_dim) + ReLU

    Args:
        site_in_dim: Site feature dimension (default: 10)
        vendor_in_dim: Vendor feature dimension (default: 9)
        hidden_dim: Hidden layer dimension (default: 64)
        out_dim: Output embedding dimension (default: 32)
        dropout: Dropout rate (default: 0.3)

    Input:
        x_dict: Dictionary of node features
        edge_index_dict: Dictionary of edge indices

    Output:
        Dictionary of node embeddings
    """

    def __init__(self, site_in_dim=10, vendor_in_dim=9, hidden_dim=64, out_dim=32, dropout=0.3):
        super().__init__()

        # Input projection layers (different dims for site vs vendor)
        self.site_lin_in = Linear(site_in_dim, hidden_dim)
        self.vendor_lin_in = Linear(vendor_in_dim, hidden_dim)

        # Layer 1: SAGEConv for each edge type
        self.conv1_site_to_vendor = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        self.conv1_vendor_to_site = SAGEConv((hidden_dim, hidden_dim), hidden_dim)

        # Layer 2: SAGEConv for each edge type
        self.conv2_site_to_vendor = SAGEConv((hidden_dim, hidden_dim), out_dim)
        self.conv2_vendor_to_site = SAGEConv((hidden_dim, hidden_dim), out_dim)

        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through 2-layer heterogeneous GraphSAGE

        Args:
            x_dict: {'site': [100, 10], 'vendor': [20, 9]}
            edge_index_dict: {('site', 'contracts', 'vendor'): [2, E],
                              ('vendor', 'rev_contracts', 'site'): [2, E]}

        Returns:
            {'site': [100, out_dim], 'vendor': [20, out_dim]}
        """
        # Project input features to hidden_dim
        x_site = self.site_lin_in(x_dict['site'])
        x_vendor = self.vendor_lin_in(x_dict['vendor'])

        # Get edge indices (handle both with and without reverse edges)
        edge_index_s2v = edge_index_dict[('site', 'contracts', 'vendor')]
        if ('vendor', 'rev_contracts', 'site') in edge_index_dict:
            edge_index_v2s = edge_index_dict[('vendor', 'rev_contracts', 'site')]
        else:
            # Create reverse edges if not present
            edge_index_v2s = edge_index_s2v.flip([0])

        # Layer 1: Convolution + ReLU + Dropout
        # Update vendor nodes from site messages
        x_vendor_new = self.conv1_site_to_vendor((x_site, x_vendor), edge_index_s2v)
        x_vendor_new = F.relu(x_vendor_new)
        x_vendor_new = F.dropout(x_vendor_new, p=self.dropout, training=self.training)

        # Update site nodes from vendor messages
        x_site_new = self.conv1_vendor_to_site((x_vendor, x_site), edge_index_v2s)
        x_site_new = F.relu(x_site_new)
        x_site_new = F.dropout(x_site_new, p=self.dropout, training=self.training)

        # Layer 2: Convolution + ReLU
        # Update vendor nodes
        x_vendor_final = self.conv2_site_to_vendor((x_site_new, x_vendor_new), edge_index_s2v)
        x_vendor_final = F.relu(x_vendor_final)

        # Update site nodes
        x_site_final = self.conv2_vendor_to_site((x_vendor_new, x_site_new), edge_index_v2s)
        x_site_final = F.relu(x_site_final)

        return {'site': x_site_final, 'vendor': x_vendor_final}


class MLPDecoder(nn.Module):
    """
    3-layer MLP for link prediction

    Architecture:
        Input: Concatenated site + vendor embeddings [2 * out_dim]
        Layer 1: Linear(2*out_dim → hidden_dim) + ReLU + Dropout
        Layer 2: Linear(hidden_dim → hidden_dim//2) + ReLU + Dropout
        Layer 3: Linear(hidden_dim//2 → 1) + Sigmoid

    Args:
        in_dim: Input dimension (2 * encoder_out_dim)
        hidden_dim: Hidden layer dimension (default: 64)
        dropout: Dropout rate (default: 0.3)

    Input:
        z_site: Site embeddings [batch_size, out_dim]
        z_vendor: Vendor embeddings [batch_size, out_dim]

    Output:
        Link probabilities [batch_size, 1] in range [0, 1]
    """

    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()

        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = nn.Linear(hidden_dim // 2, 1)

        self.dropout = dropout

    def forward(self, z_site, z_vendor):
        """
        Predict link probability from site and vendor embeddings

        Args:
            z_site: Site embeddings [batch_size, out_dim]
            z_vendor: Vendor embeddings [batch_size, out_dim]

        Returns:
            Probabilities [batch_size, 1] in [0, 1]
        """
        # Concatenate site + vendor embeddings
        z = torch.cat([z_site, z_vendor], dim=-1)  # [batch_size, 2*out_dim]

        # Layer 1: Linear → ReLU → Dropout
        z = self.lin1(z)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # Layer 2: Linear → ReLU → Dropout
        z = self.lin2(z)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # Layer 3: Linear → Sigmoid (output probabilities)
        z = self.lin3(z)
        z = torch.sigmoid(z)

        return z


class BipartiteGNN(nn.Module):
    """
    Full bipartite GNN model for site-vendor link prediction

    Architecture:
        1. BipartiteGNNEncoder: Learn node embeddings via 2-layer GraphSAGE
        2. MLPDecoder: Predict link probabilities from concatenated embeddings

    Args:
        site_in_dim: Site feature dimension (default: 10)
        vendor_in_dim: Vendor feature dimension (default: 9)
        hidden_dim: Hidden dimension for both encoder and decoder (default: 64)
        out_dim: Embedding dimension (default: 32)
        dropout: Dropout rate (default: 0.3)

    Training:
        - Positive samples: Historical contracts from contracts_2019_2024.csv
        - Negative samples: Non-existent site-vendor pairs (sampled)
        - Loss: Weighted binary cross-entropy
        - Optimization: AdamW with weight decay

    Usage:
        >>> model = BipartiteGNN(site_in_dim=10, vendor_in_dim=9, hidden_dim=64, out_dim=32)
        >>> model = model.to('cuda')
        >>> embeddings = model.encode(graph.x_dict, graph.edge_index_dict)
        >>> probs = model.decode(embeddings['site'][edge_index[0]],
        ...                      embeddings['vendor'][edge_index[1]])
    """

    def __init__(self, site_in_dim=10, vendor_in_dim=9, hidden_dim=64, out_dim=32, dropout=0.3):
        super().__init__()

        # Heterogeneous encoder (handles site + vendor nodes separately)
        self.encoder = BipartiteGNNEncoder(
            site_in_dim=site_in_dim,
            vendor_in_dim=vendor_in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout
        )

        # Decoder: predicts links from concatenated embeddings
        self.decoder = MLPDecoder(in_dim=2 * out_dim, hidden_dim=hidden_dim, dropout=dropout)

        self.out_dim = out_dim

    def encode(self, x_dict, edge_index_dict):
        """
        Encode nodes to embeddings via 2-layer GraphSAGE

        Args:
            x_dict: Dictionary of node features
                    {'site': [100, 10], 'vendor': [20, 9]}
            edge_index_dict: Dictionary of edge indices
                    {('site', 'contracts', 'vendor'): [2, 866]}

        Returns:
            Dictionary of node embeddings
                    {'site': [100, out_dim], 'vendor': [20, out_dim]}
        """
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z_site, z_vendor):
        """
        Decode link probabilities from site and vendor embeddings

        Args:
            z_site: Site embeddings [batch_size, out_dim]
            z_vendor: Vendor embeddings [batch_size, out_dim]

        Returns:
            Link probabilities [batch_size, 1] in [0, 1]
        """
        return self.decoder(z_site, z_vendor)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        """
        Full forward pass: encode nodes → decode edges

        Args:
            x_dict: Node features
            edge_index_dict: Edge connectivity (message passing)
            edge_label_index: Edges to predict [2, batch_size]
                              edge_label_index[0] = site indices
                              edge_label_index[1] = vendor indices

        Returns:
            Link probabilities [batch_size, 1]
        """
        # Encode: Learn node embeddings
        z_dict = self.encode(x_dict, edge_index_dict)

        # Extract embeddings for edges to predict
        z_site = z_dict['site'][edge_label_index[0]]      # [batch_size, out_dim]
        z_vendor = z_dict['vendor'][edge_label_index[1]]  # [batch_size, out_dim]

        # Decode: Predict link probabilities
        return self.decode(z_site, z_vendor)


def count_parameters(model):
    """
    Count trainable parameters in model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """
    Test the BipartiteGNN model with dummy data
    """
    print("="*70)
    print("BIPARTITE GNN MODEL TEST")
    print("="*70)

    # Create dummy heterogeneous graph
    data = HeteroData()

    # Node features
    data['site'].x = torch.randn(100, 10)      # 100 sites, 10 features
    data['vendor'].x = torch.randn(20, 9)      # 20 vendors, 9 features

    # Edge index (dummy: 50 random edges)
    site_indices = torch.randint(0, 100, (50,))
    vendor_indices = torch.randint(0, 20, (50,))
    data['site', 'contracts', 'vendor'].edge_index = torch.stack([site_indices, vendor_indices])

    print("\n1. Dummy Graph Created")
    print(f"   Sites: {data['site'].x.shape}")
    print(f"   Vendors: {data['vendor'].x.shape}")
    print(f"   Edges: {data['site', 'contracts', 'vendor'].edge_index.shape}")

    # Initialize model
    model = BipartiteGNN(
        site_in_dim=10,
        vendor_in_dim=9,
        hidden_dim=64,
        out_dim=32,
        dropout=0.3
    )

    print("\n2. Model Initialized")
    print(f"   Total parameters: {count_parameters(model):,}")

    # Test encoding
    print("\n3. Testing Encoder...")
    z_dict = model.encode(data.x_dict, data.edge_index_dict)
    print(f"   Site embeddings: {z_dict['site'].shape} (expected: [100, 32])")
    print(f"   Vendor embeddings: {z_dict['vendor'].shape} (expected: [20, 32])")

    # Test decoding
    print("\n4. Testing Decoder...")
    test_edges = data['site', 'contracts', 'vendor'].edge_index[:, :10]  # First 10 edges
    probs = model(data.x_dict, data.edge_index_dict, test_edges)
    print(f"   Predictions: {probs.shape} (expected: [10, 1])")
    print(f"   Probability range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")

    # Test GPU compatibility
    if torch.cuda.is_available():
        print("\n5. Testing GPU Compatibility...")
        device = torch.device('cuda')
        model = model.to(device)
        data = data.to(device)

        z_dict = model.encode(data.x_dict, data.edge_index_dict)
        test_edges = data['site', 'contracts', 'vendor'].edge_index[:, :10]
        probs = model(data.x_dict, data.edge_index_dict, test_edges)

        print(f"   ✅ Model runs on GPU")
        print(f"   GPU Memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    else:
        print("\n5. GPU not available (skipping GPU test)")

    print("\n" + "="*70)
    print("✅ MODEL TEST COMPLETE")
    print("="*70)

    return model


if __name__ == "__main__":
    model = main()
