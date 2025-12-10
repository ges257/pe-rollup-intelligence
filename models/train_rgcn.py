"""
train_rgcn.py -- R-GCN for Temporal Link Prediction

Relational Graph Convolutional Network with edge-type-specific learning.
Uses integration_quality (0/1/2) as edge relation types to learn separate
weight matrices for each quality level.

Architecture:
    - FastRGCNConv for relation-aware message passing
    - Integration quality as edge types (0=none, 1=partial, 2=full_api)
    - MLP decoder with edge features

Hyperparameters (from R-GCN paper arXiv:1703.06103v4):
    - Edge dropout: 0.4
    - Hidden channels: 128
    - Output channels: 64
    - Learning rate: 0.01

Author: Gregory E. Schwartz
Last Revised: December 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "graph_construction"))

import torch
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np

from torch_geometric.nn import RGCNConv, FastRGCNConv
from torch_geometric.utils import dropout_edge
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import trange

from build_temporal_graph import TemporalBipartiteGraphBuilder


class RGCNEncoder(torch.nn.Module):
    """
    R-GCN encoder for bipartite heterogeneous graphs

    Uses relation-aware convolutions where each integration quality level
    (0=none, 1=partial, 2=full_api) gets its own transformation.
    """

    def __init__(self, num_relations, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Use FastRGCNConv for efficiency on small graphs
        self.conv1 = FastRGCNConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_relations=num_relations
        )
        self.conv2 = FastRGCNConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=num_relations
        )

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge relation type [num_edges] (0, 1, or 2)
        """
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return x


class EdgeDecoderWithFeatures(torch.nn.Module):
    """
    MLP decoder that uses both node embeddings AND edge features

    This is the key improvement: concatenate integration_quality with embeddings
    """

    def __init__(self, hidden_channels, use_edge_features=True):
        super().__init__()
        self.use_edge_features = use_edge_features

        # Input: site_emb (hidden_channels) + vendor_emb (hidden_channels) + edge_feat (1)
        input_dim = 2 * hidden_channels + (1 if use_edge_features else 0)

        self.lin1 = Linear(input_dim, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_site, z_vendor, edge_attr=None):
        """
        Args:
            z_site: Site embeddings [num_edges, hidden_channels]
            z_vendor: Vendor embeddings [num_edges, hidden_channels]
            edge_attr: Edge features [num_edges, 1] (integration_quality normalized)
        """
        if self.use_edge_features and edge_attr is not None:
            z = torch.cat([z_site, z_vendor, edge_attr], dim=-1)
        else:
            z = torch.cat([z_site, z_vendor], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z).sigmoid()
        return z


def hetero_to_homogeneous(data, node_type_map={'site': 0, 'vendor': 1}):
    """
    Convert HeteroData to homogeneous representation for R-GCN

    Returns:
        x: Concatenated node features [num_sites + num_vendors, feat_dim]
        edge_index: Homogeneous edge index [2, num_edges]
        edge_type: Edge relation types [num_edges]
        site_mask: Boolean mask for site nodes
        vendor_mask: Boolean mask for vendor nodes
    """
    num_sites = data['site'].x.shape[0]
    num_vendors = data['vendor'].x.shape[0]

    # Concatenate node features (pad if needed)
    site_feat_dim = data['site'].x.shape[1]
    vendor_feat_dim = data['vendor'].x.shape[1]

    if site_feat_dim == vendor_feat_dim:
        x = torch.cat([data['site'].x, data['vendor'].x], dim=0)
    else:
        # Pad to match dimensions
        max_dim = max(site_feat_dim, vendor_feat_dim)
        site_x = F.pad(data['site'].x, (0, max_dim - site_feat_dim))
        vendor_x = F.pad(data['vendor'].x, (0, max_dim - vendor_feat_dim))
        x = torch.cat([site_x, vendor_x], dim=0)

    # Edge index: vendor indices need offset
    edge_index = data['site', 'contracts', 'vendor'].edge_index.clone()
    edge_index[1] += num_sites  # Offset vendor indices

    # Edge types
    edge_type = data['site', 'contracts', 'vendor'].edge_type

    # Add reverse edges (vendor -> site) with same edge types
    edge_index_rev = edge_index.flip([0])
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    edge_type = torch.cat([edge_type, edge_type], dim=0)

    # Node type masks
    site_mask = torch.zeros(num_sites + num_vendors, dtype=torch.bool)
    site_mask[:num_sites] = True
    vendor_mask = ~site_mask

    return x, edge_index, edge_type, site_mask, vendor_mask, num_sites


def sample_negative_edges(num_sites, num_vendors, pos_edges, num_samples):
    """Sample negative edges (site-vendor pairs without contracts)"""
    neg_edges = []
    while len(neg_edges) < num_samples:
        site_idx = np.random.randint(0, num_sites)
        vendor_idx = np.random.randint(0, num_vendors)
        if (site_idx, vendor_idx) not in pos_edges:
            neg_edges.append([site_idx, vendor_idx])

    return torch.tensor(neg_edges, dtype=torch.long).t()


def get_edge_features_for_pairs(site_vendor_pairs, data, num_sites):
    """
    Get integration_quality edge features for arbitrary site-vendor pairs

    For positive edges: lookup from data
    For negative edges: use default value (1 = partial integration)
    """
    device = site_vendor_pairs.device

    # Build lookup from existing edges
    edge_index = data['site', 'contracts', 'vendor'].edge_index
    edge_attr = data['site', 'contracts', 'vendor'].edge_attr

    edge_lookup = {}
    for i in range(edge_index.shape[1]):
        site_idx = edge_index[0, i].item()
        vendor_idx = edge_index[1, i].item()
        edge_lookup[(site_idx, vendor_idx)] = edge_attr[i]

    # Get features for requested pairs
    features = []
    default_feature = torch.tensor([0.5], dtype=torch.float32, device=device)

    for i in range(site_vendor_pairs.shape[1]):
        site_idx = site_vendor_pairs[0, i].item()
        vendor_idx = site_vendor_pairs[1, i].item()

        if (site_idx, vendor_idx) in edge_lookup:
            features.append(edge_lookup[(site_idx, vendor_idx)].to(device))
        else:
            # Default: partial integration (0.5)
            features.append(default_feature)

    return torch.stack(features)


def train_epoch(encoder, decoder, train_data, optimizer, device, neg_ratio=1.0, edge_dropout=0.4):
    """
    Train for one epoch using R-GCN

    Args:
        edge_dropout: Edge dropout rate (paper uses 0.4 for regular edges, 0.2 for self-loops)
                     We use 0.4 as we don't have explicit self-loops
    """
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    # Convert to homogeneous
    x, edge_index, edge_type, site_mask, vendor_mask, num_sites = hetero_to_homogeneous(train_data)
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    # Apply edge dropout (paper: 0.4 for regular edges, 0.2 for self-loops)
    # This acts like a denoising autoencoder - forces robustness
    edge_index_dropped, edge_mask = dropout_edge(
        edge_index,
        p=edge_dropout,
        training=True
    )
    # Apply same mask to edge_type
    edge_type_dropped = edge_type[edge_mask]

    # Get positive edges (in original hetero format)
    pos_edge_index = train_data['site', 'contracts', 'vendor'].edge_index
    num_pos = pos_edge_index.shape[1]

    # Create set of positive edges
    pos_edges = set()
    for i in range(num_pos):
        pos_edges.add((pos_edge_index[0, i].item(), pos_edge_index[1, i].item()))

    # Sample negative edges
    num_neg = int(num_pos * neg_ratio)
    neg_edge_index = sample_negative_edges(
        num_sites=train_data['site'].x.shape[0],
        num_vendors=train_data['vendor'].x.shape[0],
        pos_edges=pos_edges,
        num_samples=num_neg
    ).to(device)

    # Combine positive and negative
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_labels = torch.cat([
        torch.ones(num_pos),
        torch.zeros(num_neg)
    ]).to(device)

    # Get edge features for decoder
    edge_features = get_edge_features_for_pairs(edge_label_index, train_data, num_sites).to(device)

    # Shuffle
    perm = torch.randperm(edge_label_index.shape[1])
    edge_label_index = edge_label_index[:, perm]
    edge_labels = edge_labels[perm]
    edge_features = edge_features[perm]

    # Encode with R-GCN (using dropped edges for denoising)
    z = encoder(x, edge_index_dropped, edge_type_dropped)

    # Extract site and vendor embeddings
    z_site = z[edge_label_index[0]]  # Site embeddings
    z_vendor = z[edge_label_index[1] + num_sites]  # Vendor embeddings (with offset)

    # Decode
    pred = decoder(z_site, z_vendor, edge_features)

    # Compute loss
    loss = F.binary_cross_entropy(pred.view(-1), edge_labels)

    # Backward
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_temporal(encoder, decoder, train_data, val_data, device):
    """
    Temporal evaluation: Predict FUTURE edges given PAST graph structure

    Uses R-GCN encoding of TRAIN graph to predict VAL edges
    """
    encoder.eval()
    decoder.eval()

    # Convert train graph to homogeneous and encode
    x, edge_index, edge_type, site_mask, vendor_mask, num_sites = hetero_to_homogeneous(train_data)
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    # Encode train graph
    z = encoder(x, edge_index, edge_type)

    # Get validation edges (future contracts)
    val_pos_edges = val_data['site', 'contracts', 'vendor'].edge_index
    num_val_pos = val_pos_edges.shape[1]

    # Create set of train edges
    train_edges = set()
    train_edge_index = train_data['site', 'contracts', 'vendor'].edge_index
    for i in range(train_edge_index.shape[1]):
        train_edges.add((train_edge_index[0, i].item(), train_edge_index[1, i].item()))

    # Create set of val edges
    val_edges = set()
    for i in range(num_val_pos):
        val_edges.add((val_pos_edges[0, i].item(), val_pos_edges[1, i].item()))

    # Sample negative edges
    all_edges = train_edges | val_edges
    neg_edge_index = sample_negative_edges(
        num_sites=train_data['site'].x.shape[0],
        num_vendors=train_data['vendor'].x.shape[0],
        pos_edges=all_edges,
        num_samples=num_val_pos
    ).to(device)

    # Combine for evaluation
    edge_label_index = torch.cat([val_pos_edges, neg_edge_index], dim=1)
    edge_labels = torch.cat([
        torch.ones(num_val_pos),
        torch.zeros(num_val_pos)
    ]).to(device)

    # Get edge features
    # For val edges, we need to get integration_quality from val_data
    # For train embeddings, we use the train graph structure
    edge_features = get_edge_features_for_pairs(edge_label_index, val_data, num_sites).to(device)

    # Extract embeddings
    z_site = z[edge_label_index[0]]
    z_vendor = z[edge_label_index[1] + num_sites]

    # Predict
    pred = decoder(z_site, z_vendor, edge_features)

    # Metrics
    preds = pred.view(-1).cpu().numpy()
    labels = edge_labels.cpu().numpy()

    roc_auc = roc_auc_score(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    accuracy = ((preds > 0.5) == labels).mean()

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy
    }


def main():
    print("="*70)
    print("R-GCN TEMPORAL LINK PREDICTION - OPTIMIZED")
    print("="*70)
    print("\nKEY IMPROVEMENTS:")
    print("  1. Integration_quality as edge relation types (3 separate weight matrices)")
    print("  2. Edge dropout: 0.4 (from R-GCN paper)")
    print("  3. Larger embeddings: 128→64 (2x increase)")
    print("  4. Higher learning rate: 0.01 (10x increase)")
    print("  5. Decoder L2: 0.01 (stronger regularization)")
    print("\nBaseline: 0.8343 PR-AUC (unoptimized R-GCN)")
    print("Expected: 0.87-0.90 PR-AUC (with paper optimizations)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 1. Build temporal graphs with edge types
    print("\n1. Building temporal graphs with edge types...")
    base_path = Path(__file__).parent.parent.parent.parent / "v0.section2.data_design" / "2_synthetic_practice_data"

    builder = TemporalBipartiteGraphBuilder()
    train_data, val_data, metadata = builder.build_temporal_graphs(
        sites_path=base_path / "sites.csv",
        vendors_path=base_path / "vendors.csv",
        contracts_path=base_path / "contracts_2019_2024.csv",
        integration_path=base_path / "integration_matrix.csv"
    )

    # Move to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    print(f"\n   Train: {train_data['site', 'contracts', 'vendor'].edge_index.shape[1]} edges (2019-2022)")
    print(f"   Val: {val_data['site', 'contracts', 'vendor'].edge_index.shape[1]} NEW edges (2023-2024)")

    # Check edge type distribution
    train_edge_types = train_data['site', 'contracts', 'vendor'].edge_type
    print(f"\n   Edge type distribution (train):")
    for quality in [0, 1, 2]:
        count = (train_edge_types == quality).sum().item()
        pct = count / train_edge_types.shape[0] * 100
        quality_name = ['none', 'partial', 'full_api'][quality]
        print(f"     Quality {quality} ({quality_name}): {count} edges ({pct:.1f}%)")

    # 2. Initialize R-GCN model
    print("\n2. Initializing R-GCN model...")
    num_relations = 3  # Integration quality: 0, 1, 2

    # Paper uses 500-dim for FB15k-237 (272K edges)
    # We have 866 edges, so we use 128→64 (2x increase from original 64→32)
    hidden_channels = 128
    out_channels = 64

    # Compute input dimension (max of site and vendor feature dims)
    in_channels = max(train_data['site'].x.shape[1], train_data['vendor'].x.shape[1])
    print(f"   Input channels: {in_channels}")

    encoder = RGCNEncoder(
        num_relations=num_relations,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels
    ).to(device)

    decoder = EdgeDecoderWithFeatures(
        hidden_channels=out_channels,
        use_edge_features=True
    ).to(device)

    print(f"   Encoder: FastRGCNConv with {num_relations} relations")
    print(f"   Hidden channels: {hidden_channels}")
    print(f"   Output channels: {out_channels}")
    print(f"   Decoder: MLP with edge features (integration_quality)")

    # 3. Training setup (following R-GCN paper hyperparameters)
    # Paper: lr=0.01, encoder weight_decay=0, decoder weight_decay=0.01
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'weight_decay': 5e-4},  # Light L2 on encoder
        {'params': decoder.parameters(), 'weight_decay': 0.01}   # Stronger L2 on decoder (paper)
    ], lr=0.01)  # Paper uses 0.01 (10x higher than our original 0.001)

    print(f"\n   Learning rate: 0.01 (paper default)")
    print(f"   Encoder L2: 5e-4")
    print(f"   Decoder L2: 0.01 (paper default)")
    print(f"   Edge dropout: 0.4 (paper default)")

    # 4. Training loop
    print("\n3. Training...")
    print("="*70)

    best_val_pr_auc = 0
    patience = 20
    patience_counter = 0

    for epoch in trange(200, desc="Training R-GCN"):
        # Train
        train_loss = train_epoch(
            encoder=encoder,
            decoder=decoder,
            train_data=train_data,
            optimizer=optimizer,
            device=device,
            neg_ratio=1.0
        )

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 0:
            val_metrics = evaluate_temporal(encoder, decoder, train_data, val_data, device)

            print(f"\nEpoch {epoch:03d}:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val PR-AUC: {val_metrics['pr_auc']:.4f} (predicting 2023-2024)")
            print(f"   Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
            print(f"   Val Accuracy: {val_metrics['accuracy']:.4f}")

            # Early stopping
            if val_metrics['pr_auc'] > best_val_pr_auc:
                best_val_pr_auc = val_metrics['pr_auc']
                patience_counter = 0

                # Save (use separate directory for optimized version)
                checkpoint_dir = Path(__file__).parent / 'checkpoints_rgcn_optimized'
                checkpoint_dir.mkdir(exist_ok=True, parents=True)
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'val_pr_auc': best_val_pr_auc,
                }, checkpoint_dir / 'best_model.pt')

                improvement_baseline = val_metrics['pr_auc'] - 0.8343
                print(f"   ✅ Best model saved! Improvement over baseline: {improvement_baseline:+.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break

    # 5. Final evaluation
    print("\n" + "="*70)
    print("4. Final Evaluation")
    print("="*70)

    # Load best model
    checkpoint = torch.load(
        Path(__file__).parent / 'checkpoints_rgcn_optimized' / 'best_model.pt',
        weights_only=False
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    val_metrics = evaluate_temporal(encoder, decoder, train_data, val_data, device)

    print(f"\n📊 R-GCN VALIDATION RESULTS:")
    print(f"   Task: Predict 2023-2024 contracts using 2019-2022 graph")
    print(f"   PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"   ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"   Accuracy: {val_metrics['accuracy']:.4f}")

    print(f"\n📈 COMPARISON:")
    print(f"   Random baseline: 0.500")
    print(f"   SAGEConv (temporal): 0.6867 ± 0.0528")
    print(f"   R-GCN (unoptimized): 0.8343")
    print(f"   R-GCN (OPTIMIZED): {val_metrics['pr_auc']:.4f} ⭐")
    print(f"   GBM baseline: 0.937 (target)")

    improvement_sage = val_metrics['pr_auc'] - 0.6867
    improvement_baseline = val_metrics['pr_auc'] - 0.8343
    progress = (val_metrics['pr_auc'] - 0.500) / (0.937 - 0.500) * 100

    print(f"\n   Improvement over SAGEConv: {improvement_sage:+.4f}")
    print(f"   Improvement over unoptimized R-GCN: {improvement_baseline:+.4f}")
    print(f"   Progress toward GBM: {progress:.1f}%")

    if val_metrics['pr_auc'] > 0.90:
        print(f"\n🎉 OUTSTANDING! Optimized R-GCN beats GBM baseline!")
        print(f"   Paper hyperparameters + integration quality = winning combo!")
    elif val_metrics['pr_auc'] > 0.87:
        print(f"\n✅ Excellent! Paper optimizations worked as expected!")
        print(f"   Close to GBM target - feature engineering could push us over!")
    elif val_metrics['pr_auc'] > 0.84:
        print(f"\n✅ Good improvement! Optimizations helped.")
        print(f"   Consider: Basis decomposition or block decomposition next.")
    elif val_metrics['pr_auc'] > 0.83:
        print(f"\n⚠️  Modest improvement. May need:")
        print(f"   - More epochs (increase patience)")
        print(f"   - Basis decomposition (num_bases parameter)")
        print(f"   - Feature engineering")
    else:
        print(f"\n⚠️  Performance degraded. Possible overfitting from:")
        print(f"   - Learning rate too high (try 0.005)")
        print(f"   - Edge dropout too aggressive (try 0.3)")
        print(f"   - Need to tune hyperparameters individually")

    print("\n" + "="*70)

    return encoder, decoder, val_metrics


if __name__ == "__main__":
    encoder, decoder, metrics = main()
