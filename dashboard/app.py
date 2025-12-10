"""
app.py -- PE Rollup Intelligence Dashboard

Interactive Streamlit dashboard for vendor consolidation recommendations.
Translates ML model outputs into business-friendly decisions for PE operating
partners who don't have a CS background.

Key UX Decisions:
    - Traffic light risk indicators (Green/Amber/Red)
    - Dollar-first presentation (lead with savings)
    - Expandable "why" explanations
    - Portfolio-level KPIs at top

Author: Gregory E. Schwartz
Last Revised: December 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="PE Rollup Intelligence",
    page_icon="🏥",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    """Load recommendation data from plan_table.csv"""
    base = Path(__file__).parent.parent
    plan_df = pd.read_csv(base / "business" / "plan_table.csv")
    return plan_df

df = load_data()

# Header
st.title("🏥 PE Rollup Intelligence Platform")
st.subheader("Vendor Consolidation Recommendations")
st.markdown("---")

# Calculate KPIs
total_savings = df['price_delta'].sum()
days_ar_value = -df['days_ar_delta'].sum() * 1500  # $1500 per day improved
total_value = total_savings + days_ar_value
avg_days_ar = df['days_ar_delta'].mean()
low_risk_pct = (df['risk_label'].isin(['Green', 'Amber'])).mean() * 100
unique_sites = df['site_id'].nunique()

# KPI Row (4 metrics)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Value",
        value=f"${total_value:,.0f}",
        delta=f"+${days_ar_value:,.0f} from Days-A/R"
    )

with col2:
    st.metric(
        label="Recommendations",
        value=f"{len(df)}",
        delta=f"{unique_sites} sites covered"
    )

with col3:
    st.metric(
        label="Avg Days-A/R",
        value=f"{avg_days_ar:.1f} days",
        delta="Improvement" if avg_days_ar < 0 else "Increase"
    )

with col4:
    st.metric(
        label="Low Risk",
        value=f"{low_risk_pct:.0f}%",
        delta="Green + Amber"
    )

st.markdown("---")

# Filters
st.subheader("Filters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    quarter_filter = st.multiselect(
        "Quarter",
        options=sorted(df['quarter'].unique()),
        default=sorted(df['quarter'].unique())
    )

with col2:
    risk_filter = st.multiselect(
        "Risk Level",
        options=['Green', 'Amber', 'Red'],
        default=['Green', 'Amber', 'Red']
    )

with col3:
    category_filter = st.multiselect(
        "Category",
        options=sorted(df['category'].unique()),
        default=sorted(df['category'].unique())
    )

with col4:
    region_filter = st.multiselect(
        "Region",
        options=sorted(df['region'].unique()),
        default=sorted(df['region'].unique())
    )

# Apply filters
filtered_df = df[
    (df['quarter'].isin(quarter_filter)) &
    (df['risk_label'].isin(risk_filter)) &
    (df['category'].isin(category_filter)) &
    (df['region'].isin(region_filter))
]

st.markdown(f"**Showing {len(filtered_df)} of {len(df)} recommendations**")
st.markdown("---")

# Charts row
col1, col2 = st.columns(2)

with col1:
    st.subheader("Quarterly Savings")
    quarterly = filtered_df.groupby('quarter')['price_delta'].sum().reset_index()
    quarterly = quarterly.sort_values('quarter')

    fig = px.bar(
        quarterly,
        x='quarter',
        y='price_delta',
        labels={'price_delta': 'Savings ($)', 'quarter': 'Quarter'},
        color_discrete_sequence=['#3498db']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Risk Distribution")
    risk_counts = filtered_df['risk_label'].value_counts()

    # Ensure order: Green, Amber, Red
    risk_order = ['Green', 'Amber', 'Red']
    risk_counts = risk_counts.reindex(risk_order).fillna(0)

    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={
            'Green': '#2ecc71',
            'Amber': '#f39c12',
            'Red': '#e74c3c'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Recommendations table
st.subheader("Recommendations")

# Sort by fit_score descending
sorted_df = filtered_df.sort_values('fit_score', ascending=False)

# Risk emoji mapping
risk_emoji = {'Green': '🟢', 'Amber': '🟡', 'Red': '🔴'}

for idx, row in sorted_df.iterrows():
    emoji = risk_emoji.get(row['risk_label'], '⚪')
    savings_str = f"${row['price_delta']:,.0f}" if row['price_delta'] >= 0 else f"-${abs(row['price_delta']):,.0f}"

    with st.expander(
        f"{emoji} {row['site_name']} → {row['vendor_name']} | "
        f"Fit: {row['fit_score']} | {savings_str} | {row['quarter']}"
    ):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Site Details**")
            st.write(f"Site ID: {row['site_id']}")
            st.write(f"Region: {row['region']}")
            st.write(f"EHR System: {row['ehr_system']}")

        with col2:
            st.markdown("**Vendor Details**")
            st.write(f"Vendor ID: {row['vendor_id']}")
            st.write(f"Category: {row['category']}")
            integration_labels = {0: 'None', 1: 'Partial', 2: 'Full API'}
            st.write(f"Integration: {integration_labels.get(row['integration_quality'], 'Unknown')}")

        with col3:
            st.markdown("**Model Outputs**")
            st.write(f"Adoption Probability: {row['p_adoption']*100:.0f}%")
            st.write(f"Days-A/R Change: {row['days_ar_delta']:.1f} days")
            st.write(f"Risk: {row['risk_label']} ({row['risk_score']:.0f})")

        st.markdown("---")
        st.markdown(f"**Why this recommendation:** {row['top_reason']}")

# Footer
st.markdown("---")
st.markdown(
    "*PE Rollup Intelligence Platform | "
    "R-GCN Link Prediction Model (PR-AUC: 0.9407) | "
    "Gregory E. Schwartz | December 2025*"
)
