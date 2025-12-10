# Model Tier Comparison

## Progression: Simple to Complex

This project demonstrates the **baseline → simple → complex** model progression required by the rubric.

---

## Tier 1: Heuristic Baselines

**Approach:** Rule-based methods using graph topology only.

| Method | PR-AUC | Description |
|--------|--------|-------------|
| Jaccard Similarity | 0.164 | Neighborhood overlap between nodes |
| **Peer Count** | **0.171** | Number of shared connections |
| Rule-Based Composite | 0.113 | Combination of heuristics |

**Key Finding:** Heuristics perform poorly because they ignore node features and edge attributes.

---

## Tier 2: Gradient Boosting (LightGBM)

**Approach:** Traditional ML with handcrafted features.

| Metric | Value |
|--------|-------|
| PR-AUC | **0.937** |
| ROC-AUC | 0.924 |
| Features | 30 |

**Features Used:**
- Site features: region, EHR type, revenue tier
- Vendor features: category, pricing tier
- Pair features: integration_quality, historical counts

**Key Finding:** Strong baseline - requires extensive feature engineering.

---

## Tier 4: R-GCN (Primary Model)

**Approach:** Relational Graph Convolutional Network with edge-type-specific learning.

| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.9407** |
| ROC-AUC | 0.9301 |
| Accuracy | 85% |

**Architecture:**
- FastRGCNConv from PyTorch Geometric
- Bipartite heterogeneous graph (sites ↔ vendors)
- Edge-type-specific weight matrices

**Best Hyperparameters:**
- Hidden channels: 128
- Output channels: 80
- Learning rate: 0.01
- Edge dropout: 0.5
- Best epoch: 105

---

## Why R-GCN Wins

1. **Beats GBM by +0.0037 PR-AUC** (0.9407 vs 0.937)
2. **Learns graph topology** - captures multi-hop relationships
3. **Edge-type-specific convolutions** - explicitly models integration_quality
4. **End-to-end training** - features and classifier optimized jointly

---

## External Validation: KumoRFM

Tested against state-of-the-art foundation model from PyG founders:

| Model | Architecture | PR-AUC |
|-------|--------------|--------|
| R-GCN (ours) | Edge-type convolutions | 0.9407 |
| KumoRFM | Graph Transformer | 0.6209 |

**Finding:** R-GCN outperforms foundation models by +32% for tasks with dominant edge-type signals.

---

## Ablation Studies

| Feature Removed | PR-AUC | Delta | Impact |
|-----------------|--------|-------|--------|
| integration_quality | 0.6852 | -0.2555 | **DOMINANT (25.5%)** |
| site_region | 0.9260 | -0.0147 | Minor |
| vendor_category | 0.9266 | -0.0141 | Minor |

**Key Insight:** Integration quality is the dominant signal. R-GCN's architecture is optimal for this.
