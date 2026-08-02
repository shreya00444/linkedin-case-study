import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# ANALYSIS 1: FORECAST ACCURACY
# Uses: rep_forecast_submissions + historical_bookings + opportunities
# Quarters: 2024-Q3 and 2024-Q4 (only quarters with both forecast AND actuals)
# Answers: How accurate have reps been at forecasting by region and segment?
# ============================================================

def calculate_forecast_accuracy(clean_forecast, clean_bookings, clean_opps):
    validations = []

    # Step 1 - Filter valid forecasts only
    valid_forecast = clean_forecast[
        (clean_forecast['pipeline_amount'].notnull()) &
        (clean_forecast['pipeline_amount'] > 0) &
        (clean_forecast['likelihood_pct'] > 0)
    ].copy()

    # Step 2 - Only keep quarters that exist in historical bookings
    valid_quarters = clean_bookings['quarter'].unique()
    valid_forecast = valid_forecast[valid_forecast['quarter'].isin(valid_quarters)]

    # Step 3 - Join rep_forecast to opportunities to get region and segment
    forecast_enriched = pd.merge(
        valid_forecast,
        clean_opps[['account_id', 'region', 'segment']],
        on='account_id',
        how='left'
    )

    matched = forecast_enriched[forecast_enriched['region'].notnull()]
    unmatched = forecast_enriched[forecast_enriched['region'].isnull()]

    validations.append({
        'check': 'Rep forecast to CRM join',
        'expected': 'All records matched',
        'actual': f'{len(matched)} matched, {len(unmatched)} unmatched',
        'status': 'PASS' if len(unmatched) == 0 else 'WARN'
    })

    # Step 4 - Sum weighted forecasts by quarter, region, segment
    forecast_by_dims = matched.groupby(
        ['quarter', 'region', 'segment']
    ).agg(
        total_weighted_forecast=('weighted_forecast', 'sum'),
        submission_count=('rep_id', 'count')
    ).reset_index()

    # Step 5 - Get actual bookings by quarter, region, segment
    bookings_by_dims = clean_bookings.groupby(
        ['quarter', 'region', 'segment']
    ).agg(
        total_bookings=('bookings', 'sum')
    ).reset_index()

    # Step 6 - Merge forecast vs actuals on all three dimensions
    accuracy_df = pd.merge(
        bookings_by_dims,
        forecast_by_dims,
        on=['quarter', 'region', 'segment'],
        how='left'
    )

    accuracy_df['total_weighted_forecast'] = accuracy_df['total_weighted_forecast'].fillna(0)
    accuracy_df['submission_count'] = accuracy_df['submission_count'].fillna(0)

    # Step 7 - Calculate forecast error and accuracy
    accuracy_df['forecast_error'] = abs(
        accuracy_df['total_weighted_forecast'] - accuracy_df['total_bookings']
    )
    accuracy_df['accuracy_pct'] = (
        1 - accuracy_df['forecast_error'] / accuracy_df['total_bookings']
    ) * 100
    accuracy_df['over_under'] = (
        accuracy_df['total_weighted_forecast'] - accuracy_df['total_bookings']
    )
    accuracy_df['over_under_label'] = accuracy_df['over_under'].apply(
        lambda x: 'Over Forecast' if x > 0 else 'Under Forecast' if x < 0 else 'Accurate'
    )

    accuracy_df['accuracy_pct'] = accuracy_df['accuracy_pct'].round(1)
    accuracy_df['total_weighted_forecast'] = accuracy_df['total_weighted_forecast'].round(0)
    accuracy_df['forecast_error'] = accuracy_df['forecast_error'].round(0)
    accuracy_df['over_under'] = accuracy_df['over_under'].round(0)

    # ---- VALIDATION CHECKS ----
    # Dynamically detect quarters with both forecast AND bookings data
    meaningful_rows = accuracy_df[accuracy_df['total_weighted_forecast'] > 0]
    forecast_quarters = list(meaningful_rows['quarter'].unique())
    n_forecast_quarters = len(forecast_quarters)
    expected_meaningful = n_forecast_quarters * 4 * 3

    validations.append({
        'check': f'Row count check ({n_forecast_quarters} quarters with forecast data)',
        'expected': f'{expected_meaningful} rows',
        'actual': f'{len(meaningful_rows)} rows',
        'status': 'PASS' if len(meaningful_rows) == expected_meaningful else 'WARN'
    })

    regions_in_output = meaningful_rows['region'].nunique()
    validations.append({
        'check': 'All 4 regions present in forecast quarters',
        'expected': '4',
        'actual': str(regions_in_output),
        'status': 'PASS' if regions_in_output == 4 else 'FAIL'
    })

    segments_in_output = meaningful_rows['segment'].nunique()
    validations.append({
        'check': 'All 3 segments present in forecast quarters',
        'expected': '3',
        'actual': str(segments_in_output),
        'status': 'PASS' if segments_in_output == 3 else 'FAIL'
    })

    forecast_bookings = accuracy_df[
        accuracy_df['quarter'].isin(forecast_quarters)
    ]['total_bookings'].sum()
    source_bookings = clean_bookings[
        clean_bookings['quarter'].isin(forecast_quarters)
    ]['bookings'].sum()
    validations.append({
        'check': f'Bookings reconcile to source ({", ".join(sorted(forecast_quarters))})',
        'expected': f'${source_bookings:,.0f}',
        'actual': f'${forecast_bookings:,.0f}',
        'status': 'PASS' if forecast_bookings == source_bookings else 'FAIL'
    })


    return accuracy_df, validations


# ============================================================
# ANALYSIS 2: PIPELINE COVERAGE FOR Q3 2025
# Uses: opportunities (2025-Q1 active pipeline) + historical_bookings (2023-Q3 and 2024-Q3)
# Answers: Do we have enough pipeline to hit Q3 2025 targets?
# Note: Using 2025-Q1 pipeline as proxy since Q3 2025 deals not yet in CRM
# In production would dynamically pick up Q3 2025 close dates as reps add them
# ============================================================

def calculate_pipeline_coverage(clean_opps, clean_bookings):
    validations = []

    # Step 1 - Filter to most forward looking active pipeline (2025-Q1)
    clean_opps = clean_opps.copy()
    clean_opps['close_date'] = pd.to_datetime(clean_opps['close_date'], errors='coerce')

    q1_start = pd.Timestamp('2025-01-01')
    q1_end = pd.Timestamp('2025-03-31')

    planning_opps = clean_opps[
        (clean_opps['close_date'] >= q1_start) &
        (clean_opps['close_date'] <= q1_end) &
        (clean_opps['is_closed_won'] == 0) &
        (clean_opps['stage'] != 'Closed Lost') &
        (clean_opps['close_date'].notnull())
    ].copy()

    # Step 2 - Calculate current pipeline by region and segment
    pipeline_by_region_segment = planning_opps.groupby(
        ['region', 'segment']
    ).agg(
        deal_count=('opp_id', 'count'),
        total_pipeline=('pipeline_amount', 'sum')
    ).reset_index()

    # Step 3 - Use Q3 historical bookings as target baseline
    # Average of 2023-Q3 and 2024-Q3 gives realistic Q3 target
    # We use Q3 not Q1 because we are planning FOR Q3 2025
    historical_q3 = clean_bookings[
        clean_bookings['quarter'].isin(['2023-Q3', '2024-Q3'])
    ].groupby(['region', 'segment']).agg(
        avg_q3_bookings=('bookings', 'mean')
    ).reset_index()

    # Step 4 - Merge pipeline vs Q3 target
    coverage_df = pd.merge(
        historical_q3,
        pipeline_by_region_segment,
        on=['region', 'segment'],
        how='left'
    )

    coverage_df['total_pipeline'] = coverage_df['total_pipeline'].fillna(0)
    coverage_df['deal_count'] = coverage_df['deal_count'].fillna(0)

    # Step 5 - Calculate coverage ratio and gap
    coverage_df['coverage_ratio'] = (
        coverage_df['total_pipeline'] / coverage_df['avg_q3_bookings']
    ).round(2)

    coverage_df['gap'] = (
        coverage_df['avg_q3_bookings'] - coverage_df['total_pipeline']
    ).round(0)

    coverage_df['coverage_status'] = coverage_df['coverage_ratio'].apply(
        lambda x: 'Strong' if x >= 2.0
        else 'At Risk' if x >= 1.0
        else 'Critical'
    )

    coverage_df['avg_q3_bookings'] = coverage_df['avg_q3_bookings'].round(0)
    coverage_df['total_pipeline'] = coverage_df['total_pipeline'].round(0)

    # ---- VALIDATION CHECKS ----
    validations.append({
        'check': 'Planning quarter pipeline exists',
        'expected': '> 0 deals',
        'actual': f'{len(planning_opps)} active deals found',
        'status': 'PASS' if len(planning_opps) > 0 else 'FAIL'
    })

    negative_coverage = coverage_df[coverage_df['coverage_ratio'] < 0]
    validations.append({
        'check': 'Coverage ratios are positive',
        'expected': 'All >= 0',
        'actual': f'{len(negative_coverage)} negative ratios',
        'status': 'PASS' if len(negative_coverage) == 0 else 'FAIL'
    })

    expected_combinations = 4 * 3
    validations.append({
        'check': 'All region/segment combinations present',
        'expected': f'{expected_combinations}',
        'actual': f'{len(coverage_df)}',
        'status': 'PASS' if len(coverage_df) == expected_combinations else 'WARN'
    })

    total_pipeline = coverage_df['total_pipeline'].sum()
    validations.append({
        'check': 'Total planning pipeline is positive',
        'expected': '> 0',
        'actual': f'${total_pipeline:,.0f}',
        'status': 'PASS' if total_pipeline > 0 else 'FAIL'
    })

    return coverage_df, planning_opps, validations


# ============================================================
# ANALYSIS 3: REP VS CRM GAP
# Uses: rep_forecast_submissions (2025-Q1) + opportunities
# Answers: Where are reps and CRM most misaligned on deal values?
# These gaps are leading indicators of Q3 forecast inaccuracy
# ============================================================

def calculate_rep_crm_gap(clean_forecast, clean_opps):
    validations = []

    # Step 1 - Get latest rep forecast per account per quarter
    latest_forecast = clean_forecast.sort_values(
        'submission_date'
    ).groupby(['account_id', 'quarter']).last().reset_index()

    # Step 2 - Join to opportunities on account_id
    gap_df = pd.merge(
        latest_forecast[['rep_id', 'account_id', 'pipeline_amount',
                        'likelihood_pct', 'quarter']],
        clean_opps[['account_id', 'opp_id', 'pipeline_amount',
                   'region', 'segment', 'stage']],
        on='account_id',
        how='inner',
        suffixes=('_rep', '_crm')
    )

    # Step 3 - Calculate gap between rep view and CRM view
    gap_df['gap'] = gap_df['pipeline_amount_rep'] - gap_df['pipeline_amount_crm']
    gap_df['gap_pct'] = (
        gap_df['gap'] / gap_df['pipeline_amount_crm'].replace(0, np.nan) * 100
    ).round(1)
    gap_df['gap_direction'] = gap_df['gap'].apply(
        lambda x: 'Rep Higher' if x > 0
        else 'Rep Lower' if x < 0
        else 'Aligned'
    )

    gap_df['gap'] = gap_df['gap'].round(0)
    gap_df['pipeline_amount_rep'] = gap_df['pipeline_amount_rep'].round(0)
    gap_df['pipeline_amount_crm'] = gap_df['pipeline_amount_crm'].round(0)
    gap_df['abs_gap'] = gap_df['gap'].abs()
    gap_df = gap_df.sort_values('abs_gap', ascending=False)

    # ---- VALIDATION CHECKS ----
    validations.append({
        'check': 'Rep/CRM join returned records',
        'expected': '> 0 records',
        'actual': f'{len(gap_df)} records matched',
        'status': 'PASS' if len(gap_df) > 0 else 'FAIL'
    })

    if len(gap_df) > 0:
        sample_gap = gap_df.iloc[0]['gap']
        sample_rep = gap_df.iloc[0]['pipeline_amount_rep']
        sample_crm = gap_df.iloc[0]['pipeline_amount_crm']
        expected_gap = round(sample_rep - sample_crm, 0)
        validations.append({
            'check': 'Gap calculation correct',
            'expected': f'${expected_gap:,.0f}',
            'actual': f'${sample_gap:,.0f}',
            'status': 'PASS' if abs(sample_gap - expected_gap) < 1 else 'FAIL'
        })

    total_gaps = gap_df['gap'].abs().sum()
    validations.append({
        'check': 'Total gap volume calculated',
        'expected': '> 0',
        'actual': f'${total_gaps:,.0f} total disagreement',
        'status': 'PASS' if total_gaps > 0 else 'FAIL'
    })

    return gap_df, validations


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    from data_loader import load_all_clean_data

    opps, bookings, forecast, issues = load_all_clean_data()

    print("=== ANALYSIS 1: FORECAST ACCURACY (2024-Q3 and 2024-Q4) ===")
    accuracy_df, acc_validations = calculate_forecast_accuracy(
        forecast, bookings, opps
    )
    quarters_with_data = accuracy_df[accuracy_df['total_weighted_forecast'] > 0]
    print(f"Quarters with forecast data: {sorted(quarters_with_data['quarter'].unique())}")
    print(quarters_with_data[[
        'quarter', 'region', 'segment',
        'total_bookings', 'total_weighted_forecast',
        'accuracy_pct', 'over_under_label'
    ]].to_string())
    print("\nValidations:")
    for v in acc_validations:
        print(f"  {v['status']} — {v['check']}: {v['actual']}")

    print("\n=== ANALYSIS 2: PIPELINE COVERAGE VS Q3 HISTORICAL TARGET ===")
    coverage_df, planning_opps, cov_validations = calculate_pipeline_coverage(
        opps, bookings
    )
    print(coverage_df[[
        'region', 'segment', 'avg_q3_bookings',
        'total_pipeline', 'coverage_ratio', 'coverage_status', 'gap'
    ]].to_string())
    print(f"\nTotal active pipeline: ${coverage_df['total_pipeline'].sum():,.0f}")
    print(f"Total Q3 target: ${coverage_df['avg_q3_bookings'].sum():,.0f}")
    print(f"Total gap: ${coverage_df['gap'].sum():,.0f}")
    print("\nValidations:")
    for v in cov_validations:
        print(f"  {v['status']} — {v['check']}: {v['actual']}")

    print("\n=== ANALYSIS 3: REP VS CRM GAP ===")
    gap_df, gap_validations = calculate_rep_crm_gap(forecast, opps)
    print(gap_df[[
        'rep_id', 'account_id', 'quarter', 'region', 'segment',
        'pipeline_amount_rep', 'pipeline_amount_crm',
        'gap', 'gap_direction'
    ]].head(10).to_string())
    print(f"\nTotal accounts with gaps: {len(gap_df)}")
    print(f"Rep Higher: {len(gap_df[gap_df['gap_direction'] == 'Rep Higher'])}")
    print(f"Rep Lower: {len(gap_df[gap_df['gap_direction'] == 'Rep Lower'])}")
    print(f"Aligned: {len(gap_df[gap_df['gap_direction'] == 'Aligned'])}")
    print("\nValidations:")
    for v in gap_validations:
        print(f"  {v['status']} — {v['check']}: {v['actual']}")

    print("\nAnalysis complete!")


# ============================================================
# ANALYSIS 4: ACCURACY TREND BY REGION AND SEGMENT
# Uses: accuracy_df (output of calculate_forecast_accuracy)
# Answers: Is accuracy improving or declining quarter over quarter?
# ============================================================

def calculate_accuracy_trend(accuracy_df):
    """
    Pivots accuracy data to show Q3 vs Q4 side by side.
    Only includes rows with actual forecast submissions.
    """
    # Filter to only rows with forecast data
    with_data = accuracy_df[accuracy_df['total_weighted_forecast'] > 0].copy()

    # Pivot by region
    region_trend = with_data.groupby(['quarter', 'region'])['accuracy_pct'].mean().reset_index()
    region_pivot = region_trend.pivot(index='region', columns='quarter', values='accuracy_pct').reset_index()
    region_pivot.columns.name = None

    # Add trend column
    quarters = sorted([c for c in region_pivot.columns if c != 'region'])
    if len(quarters) >= 2:
        region_pivot['trend'] = region_pivot[quarters[-1]] - region_pivot[quarters[-2]]
        region_pivot['trend_label'] = region_pivot['trend'].apply(
            lambda x: f'↑ +{x:.1f}%' if x > 0 else f'↓ {x:.1f}%'
        )

    # Round
    for q in quarters:
        region_pivot[q] = region_pivot[q].round(1)
    if 'trend' in region_pivot.columns:
        region_pivot['trend'] = region_pivot['trend'].round(1)
    # Ensure all numeric columns are rounded to 1 decimal
    for col in region_pivot.columns:
        if col not in ['region', 'trend_label']:
            region_pivot[col] = pd.to_numeric(region_pivot[col], errors='ignore')
            if region_pivot[col].dtype in ['float64', 'float32']:
                region_pivot[col] = region_pivot[col].round(1)

    # Pivot by segment
    segment_trend = with_data.groupby(['quarter', 'segment'])['accuracy_pct'].mean().reset_index()
    segment_pivot = segment_trend.pivot(index='segment', columns='quarter', values='accuracy_pct').reset_index()
    segment_pivot.columns.name = None

    if len(quarters) >= 2:
        segment_pivot['trend'] = segment_pivot[quarters[-1]] - segment_pivot[quarters[-2]]
        segment_pivot['trend_label'] = segment_pivot['trend'].apply(
            lambda x: f'↑ +{x:.1f}%' if x > 0 else f'↓ {x:.1f}%'
        )

    for q in quarters:
        segment_pivot[q] = segment_pivot[q].round(1)
    if 'trend' in segment_pivot.columns:
        segment_pivot['trend'] = segment_pivot['trend'].round(1)
    # Ensure all numeric columns are rounded to 1 decimal
    for col in segment_pivot.columns:
        if col not in ['segment', 'trend_label']:
            segment_pivot[col] = pd.to_numeric(segment_pivot[col], errors='ignore')
            if segment_pivot[col].dtype in ['float64', 'float32']:
                segment_pivot[col] = segment_pivot[col].round(1)

    return region_pivot, segment_pivot


# ============================================================
# ANALYSIS 5: FORECAST SUBMISSION TIMELINESS
# Uses: rep_forecast_submissions
# Answers: How many days after quarter start did reps submit?
# ============================================================

def calculate_submission_timeliness(clean_forecast):
    """
    Calculates how many days after quarter start each rep submitted.
    Late submissions = long planning cycles.
    """
    # Quarter start dates
    quarter_starts = {
        '2024-Q3': pd.Timestamp('2024-07-01'),
        '2024-Q4': pd.Timestamp('2024-10-01'),
        '2025-Q1': pd.Timestamp('2025-01-01')
    }

    df = clean_forecast.copy()
    df['submission_date'] = pd.to_datetime(df['submission_date'], errors='coerce')
    df['quarter_start'] = df['quarter'].map(quarter_starts)
    df['days_after_quarter_start'] = (
        df['submission_date'] - df['quarter_start']
    ).dt.days

    # Remove rows where we can't calculate
    df = df[df['days_after_quarter_start'].notnull()]
    df = df[df['days_after_quarter_start'] >= 0]

    # Summarize by rep and quarter
    timeliness = df.groupby(['rep_id', 'quarter']).agg(
        submissions=('account_id', 'count'),
        avg_days_after_start=('days_after_quarter_start', 'mean'),
        earliest_submission=('days_after_quarter_start', 'min'),
        latest_submission=('days_after_quarter_start', 'max')
    ).reset_index()

    timeliness['avg_days_after_start'] = timeliness['avg_days_after_start'].round(0)
    timeliness['earliest_submission'] = timeliness['earliest_submission'].round(0)
    timeliness['latest_submission'] = timeliness['latest_submission'].round(0)

    # Add timeliness label
    timeliness['timeliness_label'] = timeliness['avg_days_after_start'].apply(
        lambda x: 'Early (0-15 days)' if x <= 15
        else 'On Time (16-30 days)' if x <= 30
        else 'Late (31-60 days)' if x <= 60
        else 'Very Late (60+ days)'
    )

    # Overall summary by quarter
    quarter_summary = df.groupby('quarter').agg(
        avg_days=('days_after_quarter_start', 'mean'),
        total_submissions=('account_id', 'count')
    ).reset_index()
    quarter_summary['avg_days'] = quarter_summary['avg_days'].round(0)

    return timeliness, quarter_summary


# ============================================================
# ANALYSIS 6: REP LEVEL ACCURACY
# Uses: accuracy_df + rep_forecast_submissions
# Answers: Which reps are most and least accurate?
# ============================================================

def calculate_rep_accuracy(clean_forecast, clean_bookings, clean_opps):
    """
    Calculates forecast accuracy at the rep level.
    Joins rep forecasts to opportunities to get region/segment.
    """
    # Filter valid forecasts
    valid_forecast = clean_forecast[
        (clean_forecast['pipeline_amount'].notnull()) &
        (clean_forecast['pipeline_amount'] > 0) &
        (clean_forecast['likelihood_pct'] > 0)
    ].copy()

    # Only quarters with historical bookings
    valid_quarters = clean_bookings['quarter'].unique()
    valid_forecast = valid_forecast[valid_forecast['quarter'].isin(valid_quarters)]

    # Join to opportunities to get region
    forecast_with_region = pd.merge(
        valid_forecast,
        clean_opps[['account_id', 'region', 'segment']],
        on='account_id',
        how='left'
    )

    # Sum weighted forecast by rep and quarter
    rep_forecast = forecast_with_region.groupby(
        ['rep_id', 'quarter', 'region']
    ).agg(
        total_weighted_forecast=('weighted_forecast', 'sum'),
        submission_count=('account_id', 'count')
    ).reset_index()

    # Get actual bookings by region and quarter
    bookings_by_region = clean_bookings.groupby(
        ['quarter', 'region']
    ).agg(
        total_bookings=('bookings', 'sum')
    ).reset_index()

    # Merge
    rep_accuracy = pd.merge(
        rep_forecast,
        bookings_by_region,
        on=['quarter', 'region'],
        how='left'
    )

    # Calculate accuracy
    rep_accuracy['forecast_error'] = abs(
        rep_accuracy['total_weighted_forecast'] - rep_accuracy['total_bookings']
    )
    rep_accuracy['accuracy_pct'] = (
        1 - rep_accuracy['forecast_error'] / rep_accuracy['total_bookings']
    ) * 100

    rep_accuracy['accuracy_pct'] = rep_accuracy['accuracy_pct'].round(1)
    rep_accuracy['total_weighted_forecast'] = rep_accuracy['total_weighted_forecast'].round(0)

    rep_accuracy['accuracy_label'] = rep_accuracy['accuracy_pct'].apply(
        lambda x: 'Strong' if x >= 70
        else 'Moderate' if x >= 40
        else 'Needs Attention'
    )

    return rep_accuracy


# ============================================================
# ANALYSIS 7: STALE DEAL DOLLAR IMPACT
# Uses: opportunities + issues list
# Answers: How much pipeline is inflated by stale deals?
# ============================================================

def calculate_stale_deal_impact(opps_raw, issues):
    """
    Quantifies the dollar impact of stale deals on pipeline numbers.
    """
    # Get stale deal record IDs from issues
    stale_ids = [
        i['record_id'] for i in issues
        if i['check_type'] == 'Stale Deal'
    ]

    # Get stale deals from raw opportunities
    stale_deals = opps_raw[opps_raw['opp_id'].isin(stale_ids)].copy()

    # Total dollar impact
    total_stale_value = stale_deals['pipeline_amount'].sum()

    # By region
    by_region = stale_deals.groupby('region').agg(
        stale_count=('opp_id', 'count'),
        stale_pipeline=('pipeline_amount', 'sum')
    ).reset_index().sort_values('stale_pipeline', ascending=False)

    # By segment
    by_segment = stale_deals.groupby('segment').agg(
        stale_count=('opp_id', 'count'),
        stale_pipeline=('pipeline_amount', 'sum')
    ).reset_index().sort_values('stale_pipeline', ascending=False)

    return total_stale_value, stale_deals, by_region, by_segment


# ============================================================
# ANALYSIS 8: FORECAST COVERAGE GAP
# Uses: accuracy_df
# Answers: What % of actual bookings was formally forecasted?
# ============================================================

def calculate_forecast_coverage_gap(accuracy_df):
    """
    Shows what percentage of actual bookings was captured
    in rep forecast submissions.
    The gap shows how much revenue booked without being forecasted.
    """
    # Only quarters with forecast data
    with_data = accuracy_df[accuracy_df['total_weighted_forecast'] > 0].copy()

    # Overall
    total_forecast = with_data['total_weighted_forecast'].sum()
    total_bookings = with_data['total_bookings'].sum()
    overall_coverage = (total_forecast / total_bookings * 100).round(1) if total_bookings > 0 else 0

    # By region
    by_region = with_data.groupby('region').agg(
        total_forecast=('total_weighted_forecast', 'sum'),
        total_bookings=('total_bookings', 'sum')
    ).reset_index()
    by_region['coverage_pct'] = (
        by_region['total_forecast'] / by_region['total_bookings'] * 100
    ).round(1)
    by_region['gap_pct'] = (100 - by_region['coverage_pct']).round(1)

    # By segment
    by_segment = with_data.groupby('segment').agg(
        total_forecast=('total_weighted_forecast', 'sum'),
        total_bookings=('total_bookings', 'sum')
    ).reset_index()
    by_segment['coverage_pct'] = (
        by_segment['total_forecast'] / by_segment['total_bookings'] * 100
    ).round(1)
    by_segment['gap_pct'] = (100 - by_segment['coverage_pct']).round(1)

    return overall_coverage, by_region, by_segment


