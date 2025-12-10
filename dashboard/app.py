"""
app.py -- PE Rollup Intelligence Platform | Synergy Intelligence Suite

Three-tab dashboard for PE stakeholder decision-making:
    Tab 1: Synergy Capture Roadmap - Investment thesis & value creation
    Tab 2: Actionable Recommendations - Filter/sort optimization opportunities
    Tab 3: Implementation Timeline - Quarterly execution plan

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
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    """Load recommendation data from local copy"""
    data_path = Path(__file__).parent / "data" / "plan_table.csv"
    return pd.read_csv(data_path)

df = load_data()

# ============================================================================
# SENSITIVITY CONTROLS (Sidebar)
# ============================================================================

with st.sidebar:
    st.header("What-If Analysis")
    st.caption("Adjust assumptions to see impact on projected value")

    # Default values
    DEFAULT_AMBER = 10
    DEFAULT_RED = 25
    DEFAULT_COST = 2000
    DEFAULT_AR = 500

    # Initialize session state with defaults if not set
    if 'amber' not in st.session_state:
        st.session_state.amber = DEFAULT_AMBER
    if 'red' not in st.session_state:
        st.session_state.red = DEFAULT_RED
    if 'cost' not in st.session_state:
        st.session_state.cost = DEFAULT_COST
    if 'ar' not in st.session_state:
        st.session_state.ar = DEFAULT_AR

    # Reset button at top
    if st.button("↺ Reset to Defaults"):
        st.session_state.amber = DEFAULT_AMBER
        st.session_state.red = DEFAULT_RED
        st.session_state.cost = DEFAULT_COST
        st.session_state.ar = DEFAULT_AR
        st.rerun()

    st.markdown("---")

    st.subheader("Risk Discounts")
    st.caption("How much to reduce expected savings for riskier recommendations")

    amber_penalty = st.slider(
        "🟡 Medium-risk discount %",
        min_value=0, max_value=50,
        step=5,
        key='amber',
        help="Vendors with some integration challenges or moderate complexity"
    )
    red_penalty = st.slider(
        "🔴 High-risk discount %",
        min_value=0, max_value=75,
        step=5,
        key='red',
        help="Vendors with significant barriers - poor integration, resistance to change"
    )

    st.markdown("---")

    st.subheader("Cost Assumptions")
    cost_per_switch = st.slider(
        "Cost per vendor switch ($)",
        min_value=500, max_value=5000,
        step=250,
        key='cost',
        help="Staff time, training, system setup for each transition"
    )
    days_ar_value = st.slider(
        "Value per A/R day saved ($/day)",
        min_value=100, max_value=1000,
        step=50,
        key='ar',
        help="Cash flow benefit of faster collections"
    )

# ============================================================================
# KPI CALCULATIONS (V3: 3-Year NPV, Payback, Enterprise Value)
# ============================================================================

# Constants
DISCOUNT_RATE = 0.10
ANNUITY_FACTOR_3Y = 2.486  # (1 - 1.1^-3) / 0.10
EBITDA_MULTIPLE = 10  # PE valuation multiple

# Risk factors from research (p1.md): Use calibrated p_adoption with risk overlays
# These are "governance haircuts", not the primary probability
RISK_FACTORS = {'Green': 1.0, 'Amber': 1 - amber_penalty/100, 'Red': 1 - red_penalty/100}
COST_PER_SWITCH = cost_per_switch
DAYS_AR_VALUE_PER_DAY = days_ar_value

# Per-recommendation calculations
# price_delta: POSITIVE = savings (switching saves money)
# days_ar_delta: NEGATIVE = improvement (fewer days in A/R)
df['annual_cost_savings'] = df['price_delta']  # Positive = savings
df['annual_ar_value'] = -df['days_ar_delta'] * DAYS_AR_VALUE_PER_DAY  # Negative delta = improvement
df['gross_annual_value'] = df['annual_cost_savings'] + df['annual_ar_value']

# Risk-adjusted expected value (using p_adoption AND risk factor)
df['risk_factor'] = df['risk_label'].map(RISK_FACTORS)
df['ev_risk_adj'] = df['p_adoption'] * df['gross_annual_value'] * df['risk_factor']

# Per-recommendation NPV and payback
df['impl_cost'] = COST_PER_SWITCH
df['npv_3y'] = (df['ev_risk_adj'] * ANNUITY_FACTOR_3Y) - df['impl_cost']
df['payback_years'] = df['impl_cost'] / df['ev_risk_adj'].replace(0, float('inf'))

# Portfolio totals
total_switches = len(df)
total_impl_cost = total_switches * COST_PER_SWITCH
total_ev_annual = df['ev_risk_adj'].sum()
portfolio_npv_3y = (total_ev_annual * ANNUITY_FACTOR_3Y) - total_impl_cost
portfolio_roi_3y = (portfolio_npv_3y / total_impl_cost) * 100 if total_impl_cost > 0 else 0
portfolio_payback = total_impl_cost / total_ev_annual if total_ev_annual > 0 else float('inf')
enterprise_value = total_ev_annual * EBITDA_MULTIPLE

# Legacy metrics for breakdown display
expected_savings = (df['p_adoption'] * df['price_delta']).sum()
days_ar_total = df['annual_ar_value'].sum() * df['p_adoption'].mean()  # Weighted
risk_penalty_amount = total_ev_annual * ANNUITY_FACTOR_3Y - df['ev_risk_adj'].sum() * ANNUITY_FACTOR_3Y  # Approx

# Risk distribution
risk_counts = df['risk_label'].value_counts()
n_green = risk_counts.get('Green', 0)
n_amber = risk_counts.get('Amber', 0)
n_red = risk_counts.get('Red', 0)
low_risk_pct = (n_green + n_amber) / len(df) * 100

# Quarterly breakdown (using new ev_risk_adj column)
quarterly = df.groupby('quarter').agg({
    'ev_risk_adj': 'sum',
    'npv_3y': 'sum',
    'site_id': 'count'
}).reset_index()
quarterly.columns = ['quarter', 'ev_annual', 'npv_3y', 'n_switches']
quarterly['impl_cost'] = quarterly['n_switches'] * COST_PER_SWITCH
quarterly = quarterly.sort_values('quarter')

# Cumulative cash flow for J-Curve (starting negative, then growing)
quarterly['cumulative_ev'] = quarterly['ev_annual'].cumsum()
quarterly['cumulative_impl'] = quarterly['impl_cost'].cumsum()

# Labels
confidence_label = "Medium" if 0.5 <= df['p_adoption'].mean() <= 0.8 else ("High" if df['p_adoption'].mean() > 0.8 else "Low")
risk_level = "Low" if low_risk_pct >= 80 else ("Medium" if low_risk_pct >= 60 else "High")

# ============================================================================
# POD AGGREGATION METRICS (For Tab 2 Synergy Capture Playbook)
# ============================================================================

pod_metrics = df.groupby('pod_id').agg({
    'site_id': 'nunique',           # n_sites_in_pod
    'ev_risk_adj': 'sum',           # total_ev_annual
    'npv_3y': 'sum',                # total_npv_3y
    'impl_cost': 'sum',             # total_impl_cost
    'risk_label': list,             # for risk mix calculation
    'region': lambda x: list(x.unique()),
    'ehr_system': lambda x: list(x.unique()),
}).reset_index()
pod_metrics.columns = ['pod_id', 'n_sites', 'total_ev_annual', 'total_npv_3y', 'total_impl_cost', 'risk_labels', 'regions', 'ehr_systems']

# Calculate pod ROI and payback
pod_metrics['pod_roi_pct'] = ((pod_metrics['total_ev_annual'] * ANNUITY_FACTOR_3Y - pod_metrics['total_impl_cost']) / pod_metrics['total_impl_cost']) * 100
pod_metrics['pod_payback_years'] = pod_metrics['total_impl_cost'] / pod_metrics['total_ev_annual'].replace(0, float('inf'))

# Risk mix counts
pod_metrics['n_green'] = pod_metrics['risk_labels'].apply(lambda x: x.count('Green'))
pod_metrics['n_amber'] = pod_metrics['risk_labels'].apply(lambda x: x.count('Amber'))
pod_metrics['n_red'] = pod_metrics['risk_labels'].apply(lambda x: x.count('Red'))

def get_category_standardization(pod_data, n_sites_in_pod):
    """Calculate standardization metrics for each category in a pod"""
    results = {}
    for category in pod_data['category'].unique():
        cat_data = pod_data[pod_data['category'] == category]
        if len(cat_data) == 0:
            continue
        top_vendor = cat_data['vendor_name'].mode().iloc[0] if len(cat_data['vendor_name'].mode()) > 0 else "Unknown"
        n_sites_top = cat_data[cat_data['vendor_name'] == top_vendor]['site_id'].nunique()
        coverage_pct = (n_sites_top / n_sites_in_pod) * 100 if n_sites_in_pod > 0 else 0
        avg_adoption = cat_data['p_adoption'].mean()
        total_npv = cat_data['npv_3y'].sum()

        # Badge logic
        if coverage_pct >= 80:
            badge = "🟢"
            status = "EXECUTE" if avg_adoption >= 0.85 else "REVIEW"
        elif coverage_pct >= 50:
            badge = "🟡"
            status = "REVIEW"
        else:
            badge = "🔴"
            status = "HOLD"

        results[category] = {
            'top_vendor': top_vendor,
            'coverage_pct': coverage_pct,
            'avg_adoption': avg_adoption,
            'total_npv': total_npv,
            'badge': badge,
            'status': status
        }
    return results

# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================

st.title("📊 PE Rollup Intelligence Platform")
st.markdown("### 🦉 Synergy Intelligence Suite")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🗺️ Value Creation Summary", "📡 Synergy Discovery Radar", "📈 Implementation Timeline"])

# ============================================================================
# TAB 1: PORTFOLIO OVERVIEW
# ============================================================================
with tab1:
    st.markdown("---")

    # Headline - PE Value Creation focus
    payback_months = int(portfolio_payback * 12) if portfolio_payback < 10 else 999
    headline = f"""
    **Standardize {df['vendor_id'].nunique()} vendors across {df['site_id'].nunique()} sites** → **${enterprise_value:,.0f} Enterprise Value Created**
    """
    st.markdown(headline)
    st.markdown("---")

    # Hero metrics - PE Priority Order: Enterprise Value, Run-Rate EBITDA, Payback, Confidence
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Enterprise Value Created", f"${enterprise_value:,.0f}", f"10x Run-Rate EBITDA")
    with col2:
        st.metric("Run-Rate EBITDA", f"${total_ev_annual:,.0f}/yr", f"Recurring annual value")
    with col3:
        st.metric("Payback Period", f"{payback_months} months", f"${total_impl_cost:,.0f} investment")
    with col4:
        st.metric("Confidence", confidence_label, f"{low_risk_pct:.0f}% low risk")

    st.markdown("---")

    # J-Curve Chart: Cumulative Cash Position over 3 Years
    st.subheader("Investment J-Curve: Cumulative Cash Position")

    # Build J-curve data: Start at -impl_cost, add value each period
    jcurve_periods = ['Start', 'Q1', 'Q2', 'Q3', 'Q4', 'Y2', 'Y3']

    # Calculate cumulative positions
    q_values = quarterly.set_index('quarter')
    ev_q1 = q_values.loc['Q1', 'ev_annual'] if 'Q1' in q_values.index else 0
    ev_q2 = q_values.loc['Q2', 'ev_annual'] if 'Q2' in q_values.index else 0
    ev_q3 = q_values.loc['Q3', 'ev_annual'] if 'Q3' in q_values.index else 0
    ev_q4 = q_values.loc['Q4', 'ev_annual'] if 'Q4' in q_values.index else 0

    jcurve_values = [
        -total_impl_cost,                          # Start: pay implementation
        -total_impl_cost + ev_q1,                  # Q1: first value realized
        -total_impl_cost + ev_q1 + ev_q2,          # Q2
        -total_impl_cost + ev_q1 + ev_q2 + ev_q3,  # Q3
        -total_impl_cost + total_ev_annual,        # Q4: Year 1 complete
        -total_impl_cost + total_ev_annual * 2,    # Y2: second year of value
        -total_impl_cost + total_ev_annual * 3,    # Y3: third year of value
    ]

    fig = go.Figure()

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")

    # J-curve line
    fig.add_trace(go.Scatter(
        x=jcurve_periods, y=jcurve_values,
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='#3498db', width=3),
        marker=dict(size=12),
        name='Cumulative Cash Position'
    ))

    # Color zones
    fig.add_hrect(y0=0, y1=max(jcurve_values)*1.1, fillcolor="rgba(46, 204, 113, 0.1)",
                  line_width=0, annotation_text="Profit Zone", annotation_position="top right")
    fig.add_hrect(y0=min(jcurve_values)*1.1, y1=0, fillcolor="rgba(231, 76, 60, 0.1)",
                  line_width=0, annotation_text="Investment Phase", annotation_position="bottom right")

    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Cumulative Cash Position ($)",
        yaxis_tickformat='$,.0f',
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Break-even at approximately month {payback_months}. Implementation costs recovered by Year 2.")

    st.markdown("---")

    # Two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("3-Year Value Breakdown")
        annual_cost_savings_total = df['annual_cost_savings'].sum() * df['p_adoption'].mean()
        annual_ar_total = df['annual_ar_value'].sum() * df['p_adoption'].mean()
        gross_3y = total_ev_annual * ANNUITY_FACTOR_3Y
        st.markdown(f"""
        | Component | Annual | 3-Year (NPV) |
        |-----------|--------|--------------|
        | Run-Rate Value | ${total_ev_annual:,.0f} | ${gross_3y:,.0f} |
        | Implementation Cost | | -${total_impl_cost:,.0f} |
        | **Net Present Value** | | **${portfolio_npv_3y:,.0f}** |

        ---
        **ROI Metrics:**
        - 3-Year ROI: **{portfolio_roi_3y:+.0f}%**
        - Payback: **{portfolio_payback:.1f} years**
        - Enterprise Value (10x): **${enterprise_value:,.0f}**
        """)

    with col2:
        st.subheader("Risk Distribution")
        risk_data = pd.DataFrame({'Risk': ['Green', 'Amber', 'Red'], 'Count': [n_green, n_amber, n_red]})
        fig = px.pie(risk_data, values='Count', names='Risk', color='Risk',
                     color_discrete_map={'Green': '#2ecc71', 'Amber': '#f39c12', 'Red': '#e74c3c'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Pod Strategy Summary (NEW)
    st.subheader("Pod Strategy Overview")
    st.caption("Sites clustered by ML embeddings into operational pods for coordinated standardization")

    # Calculate pod-level aggregates
    pod_summary = df.groupby('pod_id').agg({
        'site_id': 'nunique',
        'ev_risk_adj': 'sum',
        'npv_3y': 'sum',
        'category': lambda x: x.value_counts().index[0],  # Most common category
        'risk_label': lambda x: (x == 'Green').sum()  # Count of green
    }).reset_index()
    pod_summary.columns = ['Pod', 'Sites', 'Annual EV', '3Y NPV', 'Top Category', 'Green Count']
    pod_summary = pod_summary.sort_values('3Y NPV', ascending=False)

    # Show top 5 pods
    st.markdown("**Top Opportunities by Pod:**")
    for _, row in pod_summary.head(5).iterrows():
        status = "Ready" if row['Green Count'] > row['Sites'] * 0.5 else "Review"
        status_color = "🟢" if status == "Ready" else "🟡"
        st.markdown(
            f"- {status_color} **Pod {int(row['Pod'])}**: {row['Sites']} sites | "
            f"3Y NPV: ${row['3Y NPV']:,.0f} | Top: {row['Top Category']}"
        )

    st.markdown("---")
    st.subheader("Current Assumptions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Switch Cost:** ${COST_PER_SWITCH:,}")
    with col2:
        st.info(f"**A/R Day Value:** ${DAYS_AR_VALUE_PER_DAY:,}")
    with col3:
        st.info(f"**Risk Discounts:** 🟡{amber_penalty}% 🔴{red_penalty}%")
    st.caption("Use sidebar to run what-if scenarios")

# ============================================================================
# TAB 2: SYNERGY CAPTURE PLAYBOOK (Pod-Orchestrated)
# ============================================================================
with tab2:
    st.markdown("### 📡 Synergy Discovery Radar")
    st.caption("Identify high-value opportunities across your portfolio. Pod-based clustering shows you where to focus first.")
    st.markdown("---")

    # Filters - 4 columns now including Pod filter
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pod_options = ['All Pods'] + [f"Pod {int(p)}" for p in sorted(df['pod_id'].unique())]
        pod_filter = st.selectbox("Filter by Pod", pod_options)
    with col2:
        risk_filter = st.multiselect(
            "Filter by Risk",
            options=['Green', 'Amber', 'Red'],
            default=['Green', 'Amber', 'Red']
        )
    with col3:
        category_filter = st.multiselect(
            "Filter by Category",
            options=sorted(df['category'].unique()),
            default=sorted(df['category'].unique())
        )
    with col4:
        sort_by = st.selectbox(
            "Sort by",
            options=['Pod Priority', 'Fit Score (High→Low)', 'NPV (High→Low)', 'Risk (Low→High)']
        )

    # Apply filters
    filtered = df.copy()
    if pod_filter != 'All Pods':
        selected_pod = int(pod_filter.replace('Pod ', ''))
        filtered = filtered[filtered['pod_id'] == selected_pod]
    filtered = filtered[
        (filtered['risk_label'].isin(risk_filter)) &
        (filtered['category'].isin(category_filter))
    ]

    # Apply sort
    if sort_by == 'Pod Priority':
        filtered = filtered.sort_values(['pod_id', 'npv_3y'], ascending=[True, False])
    elif sort_by == 'Fit Score (High→Low)':
        filtered = filtered.sort_values('fit_score', ascending=False)
    elif sort_by == 'NPV (High→Low)':
        filtered = filtered.sort_values('npv_3y', ascending=False)
    else:  # Risk Low→High
        risk_order = {'Green': 0, 'Amber': 1, 'Red': 2}
        filtered['risk_order'] = filtered['risk_label'].map(risk_order)
        filtered = filtered.sort_values('risk_order')

    # Summary
    st.markdown(f"**Showing {len(filtered)} recommendations across {filtered['pod_id'].nunique()} pods**")

    # Risk Key Legend (matches sidebar What-If Analysis definitions)
    st.caption(
        f"**Risk Key:** "
        f"🟢 **Green** (Low Risk: strong fit, good integration, 0% discount) | "
        f"🟡 **Amber** (Medium Risk: some integration challenges, {amber_penalty}% discount) | "
        f"🔴 **Red** (High Risk: significant barriers, {red_penalty}% discount)"
    )
    st.markdown("---")

    # Find highest-value pod for default expansion
    if len(filtered) > 0:
        pod_npv_totals = filtered.groupby('pod_id')['npv_3y'].sum()
        highest_value_pod = pod_npv_totals.idxmax()
    else:
        highest_value_pod = None

    # Risk emoji mapping
    risk_emoji = {'Green': '🟢', 'Amber': '🟡', 'Red': '🔴'}

    # Render by pod
    for pod_id in sorted(filtered['pod_id'].unique()):
        pod_data = filtered[filtered['pod_id'] == pod_id]
        pm_row = pod_metrics[pod_metrics['pod_id'] == pod_id]

        if len(pm_row) == 0:
            continue
        pm = pm_row.iloc[0]

        # Pod header with summary
        risk_str = f"{pm['n_green']}G/{pm['n_amber']}A/{pm['n_red']}R"
        region_str = ", ".join(pm['regions'][:2]) + ("..." if len(pm['regions']) > 2 else "")

        with st.expander(
            f"📦 **POD {int(pod_id)}** ({region_str}) | ${pm['total_npv_3y']:,.0f} NPV | {pm['n_sites']} sites | {risk_str}",
            expanded=(pod_id == highest_value_pod)
        ):
            # Pod Summary Card - 6 metrics in 2 rows
            st.markdown("**Pod Transformation Summary**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Pod 3Y NPV", f"${pm['total_npv_3y']:,.0f}")
            col2.metric("Annual EV", f"${pm['total_ev_annual']:,.0f}/yr")
            payback_display = f"{pm['pod_payback_years']:.1f} yrs" if pm['pod_payback_years'] < 10 else "N/A"
            col3.metric("Payback", payback_display)

            col1, col2, col3 = st.columns(3)
            col1.metric("Pod ROI", f"{pm['pod_roi_pct']:,.0f}%")
            col2.metric("Impl Cost", f"${pm['total_impl_cost']:,.0f}")
            col3.metric("Risk Mix", risk_str)

            st.markdown("---")

            # Category Standardization Badges
            st.markdown("**Category Standardization Opportunities:**")
            cat_std = get_category_standardization(pod_data, pm['n_sites'])

            if len(cat_std) > 0:
                cols = st.columns(min(3, len(cat_std)))
                for i, (cat, metrics) in enumerate(cat_std.items()):
                    with cols[i % 3]:
                        st.markdown(f"""
{metrics['badge']} **{cat}**
- Coverage: {metrics['coverage_pct']:.0f}%
- Top: {metrics['top_vendor']}
- Adoption: {metrics['avg_adoption']*100:.0f}%
- NPV: ${metrics['total_npv']:,.0f}
- Status: **{metrics['status']}**
                        """)

            st.markdown("---")

            # Action Recommendation
            if pm['pod_roi_pct'] > 500 and pm['n_green'] >= 4:
                action = "GREENLIGHT for Q1 (Slam Dunk)"
                action_color = "success"
            elif pm['pod_roi_pct'] > 100:
                action = "APPROVE for Q1-Q2 (High Value)"
                action_color = "success"
            elif pm['pod_roi_pct'] > 50:
                action = "REVIEW for Q2-Q3 (Staged)"
                action_color = "warning"
            elif pm['pod_roi_pct'] > 0:
                action = "REVIEW for Later Phases"
                action_color = "warning"
            else:
                action = "HOLD (Negative/Fragmented)"
                action_color = "error"

            if action_color == "success":
                st.success(f"**Action:** {action}")
            elif action_color == "warning":
                st.warning(f"**Action:** {action}")
            else:
                st.error(f"**Action:** {action}")

            st.markdown("---")

            # Individual Recommendation Cards (nested)
            st.markdown(f"**Individual Recommendations ({len(pod_data)}):**")
            for _, row in pod_data.sort_values('fit_score', ascending=False).iterrows():
                emoji = risk_emoji.get(row['risk_label'], '⚪')
                with st.expander(
                    f"{emoji} {row['vendor_name']} → {row['site_name']} ({row['category']}) | "
                    f"Fit: {row['fit_score']} | NPV: ${row['npv_3y']:,.0f} | {row['quarter']}"
                ):
                    # Two-column layout
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Model Prediction**")
                        st.write(f"- Adoption Prob: {row['p_adoption']*100:.0f}%")
                        st.write(f"- Fit Score: {row['fit_score']} ({row['fit_label']})")
                        st.write(f"- Days A/R Impact: {row['days_ar_delta']:+.1f} days")
                        st.write(f"- Integration: {['None', 'Partial', 'Full'][int(row['integration_quality'])]}")
                        st.write(f"- Risk: {row['risk_label']}")
                    with c2:
                        st.markdown("**Business Value**")
                        st.write(f"- Annual Cost Savings: ${row['annual_cost_savings']:,.0f}")
                        st.write(f"- Annual A/R Value: ${row['annual_ar_value']:,.0f}")
                        st.write(f"- Risk-Adj EV: ${row['ev_risk_adj']:,.0f}/yr")
                        st.write(f"- 3Y NPV: ${row['npv_3y']:,.0f}")
                        st.write(f"- Impl Cost: ${row['impl_cost']:,.0f}")
                    st.caption(f"**Why:** {row['top_reason']}")

# ============================================================================
# TAB 3: IMPLEMENTATION PLAN
# ============================================================================
with tab3:
    st.markdown("---")

    # Timeline summary
    st.markdown(f"""
    ### Implementation Timeline

    **Strategy:** High-impact, lower-risk consolidations in Q1-Q2; more complex changes in Q3-Q4.

    Total: **{total_switches} switches** over **4 quarters** = ~{total_switches//4} per quarter average.
    """)

    st.markdown("---")

    # Quarter-by-quarter expanders
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = df[df['quarter'] == quarter].copy()
        q_stats = quarterly[quarterly['quarter'] == quarter].iloc[0] if len(quarterly[quarterly['quarter'] == quarter]) > 0 else None

        if len(q_data) == 0:
            continue

        # Count by risk
        q_green = (q_data['risk_label'] == 'Green').sum()
        q_amber = (q_data['risk_label'] == 'Amber').sum()
        q_red = (q_data['risk_label'] == 'Red').sum()

        q_annual_ev = q_stats['ev_annual'] if q_stats is not None else 0
        q_npv = q_stats['npv_3y'] if q_stats is not None else 0
        q_cost = len(q_data) * COST_PER_SWITCH

        with st.expander(
            f"**{quarter}** — {len(q_data)} switches | "
            f"3Y NPV: ${q_npv:,.0f} | "
            f"🟢{q_green} 🟡{q_amber} 🔴{q_red}"
        ):
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Switches", len(q_data))
            col2.metric("Annual EV", f"${q_annual_ev:,.0f}")
            col3.metric("Implementation Cost", f"${q_cost:,.0f}")
            col4.metric("3-Year NPV", f"${q_npv:,.0f}")

            st.markdown("---")

            # Vendor list
            st.markdown("**Vendors in this quarter:**")

            for _, row in q_data.sort_values('fit_score', ascending=False).iterrows():
                emoji = risk_emoji.get(row['risk_label'], '⚪')
                st.markdown(
                    f"- {emoji} **{row['vendor_name']}** → {row['site_name']} (Pod {int(row['pod_id'])}) | "
                    f"Fit: {row['fit_score']} | NPV: ${row['npv_3y']:,.0f}"
                )

    st.markdown("---")

    # Quarterly comparison chart
    st.subheader("Quarterly 3-Year NPV Comparison")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=quarterly['quarter'],
        y=quarterly['npv_3y'],
        name='3-Year NPV',
        marker_color='#2ecc71'
    ))
    fig.add_trace(go.Bar(
        x=quarterly['quarter'],
        y=quarterly['impl_cost'],
        name='Implementation Cost',
        marker_color='#e74c3c'
    ))
    fig.update_layout(
        barmode='group',
        xaxis_title="Quarter",
        yaxis_title="Amount ($)",
        yaxis_tickformat='$,.0f',
        height=300,
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("PE Rollup Intelligence Platform | R-GCN Link Prediction Model (PR-AUC: 0.9407) | Gregory E. Schwartz | December 2025")
