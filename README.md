# Healthcare Vendor Consolidation ML System

**Course:** AIM 5004-1 Predictive Modeling
**Author:** Gregory
**Date:** December 2025

---

## Problem Statement

**Stakeholder:** Private Equity (PE) Operating Partner managing 200-500 dental practice sites.

**Problem:** Fragmented vendor landscape across acquired dental practices. Need to identify optimal vendor consolidation opportunities.

**Value Proposition:**
- Projected annual savings: **$40,486**
- Risk-adjusted savings: **$38,075**
- Days-A/R improvement: **-1.5 days**
- Total value: **$61,596**

---

## Key Results

| Model | PR-AUC | Notes |
|-------|--------|-------|
| **R-GCN** | **0.9407** | Primary model - BEATS GBM |
| LightGBM | 0.937 | Tier 2 baseline |
| KumoRFM | 0.621 | External validation |
| Heuristics | 0.171 | Tier 1 baseline |

**Key Finding:** Integration quality contributes **25.5%** of model performance.

---

## Directory Structure

```
final_project_notebooks_dir/
├── data/           # 8 files - Sites, vendors, contracts, train/val
├── models/         # 6 files - R-GCN, baselines, checkpoint
├── results/        # 4 files - Metrics, ablations, comparisons
├── business/       # 5 files - Recommendations, simulation
└── notebooks/      # Presentation notebook
```

---

## Quick Start

```python
# Load data
import pandas as pd
sites = pd.read_csv('data/sites.csv')
vendors = pd.read_csv('data/vendors.csv')

# Load results
import json
with open('results/test_metrics.json') as f:
    metrics = json.load(f)
print(f"R-GCN PR-AUC: {metrics['link_prediction']['pr_auc']}")
```

---

## Model Progression

1. **Tier 1 - Heuristics** (PR-AUC: 0.171)
2. **Tier 2 - LightGBM** (PR-AUC: 0.937)
3. **Tier 4 - R-GCN** (PR-AUC: 0.9407) - Best model

See `results/tier_comparison.md` for detailed analysis.
