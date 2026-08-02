import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_all_clean_data
from analysis import (
    calculate_forecast_accuracy,
    calculate_pipeline_coverage,
    calculate_rep_crm_gap
)

st.set_page_config(
    page_title="Q3 Forecast & Planning",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Base */
    .main { background-color: #F3F2EF; }
    .block-container { padding: 2rem 3rem; max-width: 1200px; }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Force all text black */
    html, body, [class*="css"], p, span, div, label, 
    input, textarea, select, option {
        color: #000000 !important;
    }

    /* Page header */
    .page-header {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #0A66C2;
        border: 1px solid #E0E0E0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .page-header h1 {
        color: #000000 !important;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
    }
    .page-header p {
        color: #333333 !important;
        font-size: 0.9rem;
        margin: 0;
    }

    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid #E0E0E0;
        margin: 1.5rem 0;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] { 
        font-size: 0.75rem !important; 
        color: #333333 !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] { 
        font-size: 1.3rem !important; 
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* App background */
    [data-testid="stAppViewContainer"] {
        background-color: #F3F2EF;
    }

    /* Dropdown fix */
    [data-testid="stSelectbox"] select,
    [data-testid="stSelectbox"] div,
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }

    /* Dropdown options */
    [role="option"] {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    [role="listbox"] {
        background-color: #FFFFFF !important;
    }

    /* Tabs */
    [data-testid="stTab"] button {
        color: #000000 !important;
    }
    [data-testid="stTab"] button[aria-selected="true"] {
        color: #0A66C2 !important;
        border-bottom: 2px solid #0A66C2;
    }

    /* Table */
    .dataframe {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 6px !important;
    }
    [data-testid="stExpander"] summary {
        color: #000000 !important;
    }

    /* Input widgets */
    [data-baseweb="input"] {
        background-color: #FFFFFF !important;
    }
    [data-baseweb="select"] {
        background-color: #FFFFFF !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #666666 !important;
        font-size: 0.75rem;
        padding: 2rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    opps, bookings, forecast, issues = load_all_clean_data()
    accuracy_df, acc_validations = calculate_forecast_accuracy(forecast, bookings, opps)
    coverage_df, planning_opps, cov_validations = calculate_pipeline_coverage(opps, bookings)
    gap_df, gap_validations = calculate_rep_crm_gap(forecast, opps)
    return (opps, bookings, forecast, issues, accuracy_df, acc_validations,
            coverage_df, planning_opps, cov_validations, gap_df, gap_validations)

with st.spinner("Loading data..."):
    (opps, bookings, forecast, issues, accuracy_df, acc_validations,
     coverage_df, planning_opps, cov_validations, gap_df, gap_validations) = get_data()

st.markdown("""
<div class="page-header">
    <h1>LinkedIn · Q3 2025 Forecast & Planning</h1>
    <p>Automated pipeline ingesting CRM opportunities, historical bookings, and rep forecast submissions · Data as of Q1 2025</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Quality",
    "Forecast Accuracy",
    "Pipeline Coverage",
    "Rep vs CRM Gap"
])

with tab1:

    issue_types = {}
    for issue in issues:
        t = issue['check_type']
        issue_types[t] = issue_types.get(t, 0) + 1

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Issues", len(issues))
    col2.metric("Stale Deals", issue_types.get('Stale Deal', 0))
    col3.metric("Duplicates Removed", issue_types.get('Duplicate', 0) // 2)
    col4.metric("Missing Fields",
        issue_types.get('Missing close_date', 0) +
        issue_types.get('Missing rep_id', 0) +
        issue_types.get('Missing pipeline_amount', 0))
    col5.metric("Zero Likelihood", issue_types.get('Zero likelihood_pct', 0))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Issue Log**")

    issues_df = pd.DataFrame(issues)
    st.dataframe(
        issues_df[['check_type', 'table', 'record_id', 'issue', 'action_taken']].rename(columns={
            'check_type': 'Issue Type',
            'table': 'Table',
            'record_id': 'Record',
            'issue': 'Description',
            'action_taken': 'Action Taken'
        }),
        use_container_width=True,
        hide_index=True,
        height=400
    )
    

with tab2:
    
    st.markdown('<p class="section-title">Forecast Accuracy · 2024-Q3 and 2024-Q4</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Weighted rep forecasts (pipeline × likelihood %) compared against actual historical bookings. Only quarters with both forecast submissions and actuals are shown.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_quarter = st.selectbox("Quarter", ["All"] + sorted(accuracy_df['quarter'].unique().tolist()))
    with col2:
        selected_region = st.selectbox("Region", ["All"] + sorted(accuracy_df['region'].unique().tolist()))
    with col3:
        selected_segment = st.selectbox("Segment", ["All"] + sorted(accuracy_df['segment'].unique().tolist()))

    filtered = accuracy_df.copy()
    if selected_quarter != "All":
        filtered = filtered[filtered['quarter'] == selected_quarter]
    if selected_region != "All":
        filtered = filtered[filtered['region'] == selected_region]
    if selected_segment != "All":
        filtered = filtered[filtered['segment'] == selected_segment]

    filtered_with_data = filtered[filtered['total_weighted_forecast'] > 0]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Actual Bookings", f"${filtered['total_bookings'].sum():,.0f}")
    col2.metric("Weighted Forecast", f"${filtered['total_weighted_forecast'].sum():,.0f}")
    col3.metric("Over Forecasted", len(filtered[filtered['over_under_label'] == 'Over Forecast']))
    col4.metric("Under Forecasted", len(filtered[filtered['over_under_label'] == 'Under Forecast']))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.dataframe(
        filtered_with_data[[
            'quarter', 'region', 'segment',
            'total_bookings', 'total_weighted_forecast',
            'accuracy_pct', 'over_under_label'
        ]].rename(columns={
            'quarter': 'Quarter',
            'region': 'Region',
            'segment': 'Segment',
            'total_bookings': 'Actual Bookings ($)',
            'total_weighted_forecast': 'Weighted Forecast ($)',
            'accuracy_pct': 'Accuracy %',
            'over_under_label': 'Status'
        }),
        use_container_width=True,
        hide_index=True,
        height=450
    )


    # Accuracy trend tables
    from analysis import calculate_accuracy_trend
    region_trend, segment_trend = calculate_accuracy_trend(accuracy_df)

    quarters = sorted([c for c in region_trend.columns
                       if c not in ['region', 'trend', 'trend_label']])

    def color_trend(val):
        if isinstance(val, str) and '↑' in val:
            return 'color: green; font-weight: bold'
        elif isinstance(val, str) and '↓' in val:
            return 'color: red; font-weight: bold'
        return 'color: black'

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Accuracy Trend by Region**")

    display_region = region_trend[['region'] + quarters + ['trend_label']].rename(columns={
        'region': 'Region',
        quarters[0]: f'{quarters[0]} Accuracy %',
        quarters[1]: f'{quarters[1]} Accuracy %',
        'trend_label': 'Trend'
    })
    st.dataframe(
        display_region.style.applymap(color_trend, subset=['Trend']),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("**Accuracy Trend by Segment**")
    display_segment = segment_trend[['segment'] + quarters + ['trend_label']].rename(columns={
        'segment': 'Segment',
        quarters[0]: f'{quarters[0]} Accuracy %',
        quarters[1]: f'{quarters[1]} Accuracy %',
        'trend_label': 'Trend'
    })
    st.dataframe(
        display_segment.style.applymap(color_trend, subset=['Trend']),
        use_container_width=True,
        hide_index=True
    )

    with st.expander("Validation checks"):
        for v in acc_validations:
            if v['status'] == 'PASS':
                st.success(f"✅ {v['check']}: {v['actual']}")
            elif v['status'] == 'WARN':
                st.warning(f"⚠️ {v['check']}: {v['actual']}")
            else:
                st.error(f"❌ {v['check']}: {v['actual']}")

    

with tab3:
    
    st.markdown('<p class="section-title">Pipeline Coverage · Q3 2025 Planning</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Active pipeline benchmarked against the average of 2023-Q3 and 2024-Q3 historical bookings. Coverage ratio of 2x or above is considered healthy.</p>', unsafe_allow_html=True)

    total_pipeline = coverage_df['total_pipeline'].sum()
    total_target = coverage_df['avg_q3_bookings'].sum()
    total_gap = coverage_df['gap'].sum()
    overall_coverage = total_pipeline / total_target if total_target > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Pipeline", f"${total_pipeline:,.0f}")
    col2.metric("Q3 Historical Target", f"${total_target:,.0f}")
    col3.metric("Overall Coverage", f"{overall_coverage:.2f}x")
    col4.metric("Pipeline Gap", f"${total_gap:,.0f}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.dataframe(
        coverage_df[[
            'region', 'segment', 'avg_q3_bookings',
            'total_pipeline', 'deal_count',
            'coverage_ratio', 'coverage_status', 'gap'
        ]].rename(columns={
            'region': 'Region',
            'segment': 'Segment',
            'avg_q3_bookings': 'Q3 Target ($)',
            'total_pipeline': 'Active Pipeline ($)',
            'deal_count': 'Deals',
            'coverage_ratio': 'Coverage Ratio',
            'coverage_status': 'Status',
            'gap': 'Gap ($)'
        }),
        use_container_width=True,
        hide_index=True
    )

    if len(planning_opps) > 0:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Active Deals in Planning Window**")
        st.dataframe(
            planning_opps[[
                'opp_id', 'account_id', 'region', 'segment',
                'stage', 'pipeline_amount', 'close_date', 'rep_id'
            ]].rename(columns={
                'opp_id': 'Opp ID',
                'account_id': 'Account',
                'region': 'Region',
                'segment': 'Segment',
                'stage': 'Stage',
                'pipeline_amount': 'Pipeline ($)',
                'close_date': 'Close Date',
                'rep_id': 'Rep'
            }),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("Validation checks"):
        for v in cov_validations:
            if v['status'] == 'PASS':
                st.success(f"✅ {v['check']}: {v['actual']}")
            elif v['status'] == 'WARN':
                st.warning(f"⚠️ {v['check']}: {v['actual']}")
            else:
                st.error(f"❌ {v['check']}: {v['actual']}")

    

with tab4:
    
    st.markdown('<p class="section-title">Rep vs CRM Pipeline Gap</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Where rep forecast submissions disagree with CRM opportunity values. Large or persistent gaps are leading indicators of forecast inaccuracy going into Q3. Sorted by absolute gap size.</p>', unsafe_allow_html=True)

    total_disagreement = gap_df['gap'].abs().sum()
    rep_higher = len(gap_df[gap_df['gap_direction'] == 'Rep Higher'])
    rep_lower = len(gap_df[gap_df['gap_direction'] == 'Rep Lower'])
    aligned = len(gap_df[gap_df['gap_direction'] == 'Aligned'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Disagreement", f"${total_disagreement:,.0f}")
    col2.metric("Rep Higher Than CRM", rep_higher)
    col3.metric("Rep Lower Than CRM", rep_lower)
    col4.metric("Aligned", aligned)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    gap_filter = st.selectbox("Filter by direction", ["All", "Rep Higher", "Rep Lower", "Aligned"])
    filtered_gap = gap_df if gap_filter == "All" else gap_df[gap_df['gap_direction'] == gap_filter]

    st.dataframe(
        filtered_gap[[
            'rep_id', 'account_id', 'quarter', 'region', 'segment',
            'stage', 'pipeline_amount_rep', 'pipeline_amount_crm',
            'gap', 'gap_pct', 'gap_direction'
        ]].rename(columns={
            'rep_id': 'Rep',
            'account_id': 'Account',
            'quarter': 'Quarter',
            'region': 'Region',
            'segment': 'Segment',
            'stage': 'Stage',
            'pipeline_amount_rep': 'Rep View ($)',
            'pipeline_amount_crm': 'CRM Value ($)',
            'gap': 'Gap ($)',
            'gap_pct': 'Gap %',
            'gap_direction': 'Direction'
        }),
        use_container_width=True,
        hide_index=True,
        height=500
    )

    with st.expander("Validation checks"):
        for v in gap_validations:
            if v['status'] == 'PASS':
                st.success(f"✅ {v['check']}: {v['actual']}")
            elif v['status'] == 'WARN':
                st.warning(f"⚠️ {v['check']}: {v['actual']}")
            else:
                st.error(f"❌ {v['check']}: {v['actual']}")

    
