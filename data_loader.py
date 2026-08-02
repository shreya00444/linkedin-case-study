import pandas as pd
import numpy as np
from datetime import datetime
import os

# File path to Excel data
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "case_interview_data.xlsx")

# Scalable test data exclusion list
KNOWN_TEST_REPS = ['REP-099', 'REP-000', 'REP-999']
KNOWN_TEST_ACCOUNTS = ['ACC-999', 'ACC-000', 'ACC-TEST']

def load_raw_data():
    """Load all three sheets from the Excel file"""
    opportunities = pd.read_excel(DATA_PATH, sheet_name="opportunities")
    historical_bookings = pd.read_excel(DATA_PATH, sheet_name="historical_bookings")
    rep_forecast = pd.read_excel(DATA_PATH, sheet_name="rep_forecast_submissions")
    return opportunities, historical_bookings, rep_forecast

def clean_opportunities(df):
    """Clean and flag issues in the opportunities table"""
    issues = []

    # 1. Remove duplicate opp_ids
    duplicates = df[df.duplicated(subset=['opp_id'], keep=False)]
    if len(duplicates) > 0:
        for _, row in duplicates.iterrows():
            issues.append({
                'check_type': 'Duplicate',
                'table': 'opportunities',
                'record_id': row['opp_id'],
                'issue': 'Duplicate opp_id found',
                'action_taken': 'Kept first instance, removed duplicate'
            })
        df = df.drop_duplicates(subset=['opp_id'], keep='first')

    # 2. Remove test data using scalable exclusion list
    test_mask = (
        df['rep_id'].isin(KNOWN_TEST_REPS) |
        df['account_id'].isin(KNOWN_TEST_ACCOUNTS)
    )
    test_records = df[test_mask]
    if len(test_records) > 0:
        for _, row in test_records.iterrows():
            issues.append({
                'check_type': 'Test Data',
                'table': 'opportunities',
                'record_id': row['opp_id'],
                'issue': 'Suspected test record based on rep/account ID pattern',
                'action_taken': 'Excluded from all analysis'
            })
        df = df[~test_mask].copy()

    # 3. Flag missing close dates
    missing_close = df[df['close_date'].isnull()]
    if len(missing_close) > 0:
        for _, row in missing_close.iterrows():
            issues.append({
                'check_type': 'Missing close_date',
                'table': 'opportunities',
                'record_id': row['opp_id'],
                'issue': 'Close date is null',
                'action_taken': 'Excluded from Q3 pipeline analysis'
            })

    # 4. Flag missing rep IDs
    missing_rep = df[df['rep_id'].isnull()]
    if len(missing_rep) > 0:
        for _, row in missing_rep.iterrows():
            issues.append({
                'check_type': 'Missing rep_id',
                'table': 'opportunities',
                'record_id': row['opp_id'],
                'issue': 'Rep ID is null',
                'action_taken': 'Kept in pipeline totals, excluded from rep analysis'
            })

    # 5. Flag stale deals
    today = datetime.today()
    df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
    stale_mask = (
        (df['close_date'] < today) &
        (df['is_closed_won'] == 0) &
        (df['stage'] != 'Closed Lost') &
        (df['close_date'].notnull())
    )
    stale_deals = df[stale_mask]
    if len(stale_deals) > 0:
        for _, row in stale_deals.iterrows():
            issues.append({
                'check_type': 'Stale Deal',
                'table': 'opportunities',
                'record_id': row['opp_id'],
                'issue': f"Close date {row['close_date'].date()} is in the past but deal is still open",
                'action_taken': 'Flagged and excluded from Q3 pipeline'
            })

    return df, issues

def clean_rep_forecast(df):
    """Clean and flag issues in rep_forecast_submissions table"""
    issues = []

    # 1. Remove test data using scalable exclusion list
    test_mask = (
        df['rep_id'].isin(KNOWN_TEST_REPS) |
        df['account_id'].isin(KNOWN_TEST_ACCOUNTS)
    )
    test_records = df[test_mask]
    if len(test_records) > 0:
        for _, row in test_records.iterrows():
            issues.append({
                'check_type': 'Test Data',
                'table': 'rep_forecast_submissions',
                'record_id': f"{row['rep_id']}/{row['account_id']}",
                'issue': 'Suspected test record based on rep/account ID pattern',
                'action_taken': 'Excluded from all analysis'
            })
        df = df[~test_mask].copy()

    # 2. Flag missing pipeline amounts
    missing_amount = df[df['pipeline_amount'].isnull()]
    if len(missing_amount) > 0:
        for _, row in missing_amount.iterrows():
            issues.append({
                'check_type': 'Missing pipeline_amount',
                'table': 'rep_forecast_submissions',
                'record_id': f"{row['rep_id']}/{row['account_id']}",
                'issue': 'Pipeline amount is null',
                'action_taken': 'Excluded from weighted forecast calculation'
            })

    # 3. Flag zero pipeline amounts
    zero_amount = df[df['pipeline_amount'] == 0]
    if len(zero_amount) > 0:
        for _, row in zero_amount.iterrows():
            issues.append({
                'check_type': 'Zero pipeline_amount',
                'table': 'rep_forecast_submissions',
                'record_id': f"{row['rep_id']}/{row['account_id']}",
                'issue': 'Pipeline amount is zero',
                'action_taken': 'Excluded from weighted forecast calculation'
            })

    # 4. Flag zero likelihood
    zero_likelihood = df[df['likelihood_pct'] == 0]
    if len(zero_likelihood) > 0:
        for _, row in zero_likelihood.iterrows():
            issues.append({
                'check_type': 'Zero likelihood_pct',
                'table': 'rep_forecast_submissions',
                'record_id': f"{row['rep_id']}/{row['account_id']}",
                'issue': 'Likelihood is 0% - rep has no confidence in this deal',
                'action_taken': 'Excluded from weighted forecast - contributes zero value'
            })

    # 5. Calculate weighted forecast
    df = df.copy()
    df['weighted_forecast'] = df['pipeline_amount'].fillna(0) * df['likelihood_pct'] / 100

    return df, issues

def clean_historical_bookings(df):
    """Clean and validate historical bookings table"""
    issues = []

    # 1. Check for nulls
    for col in ['quarter', 'region', 'segment', 'bookings']:
        null_rows = df[df[col].isnull()]
        if len(null_rows) > 0:
            for _, row in null_rows.iterrows():
                issues.append({
                    'check_type': f'Missing {col}',
                    'table': 'historical_bookings',
                    'record_id': f"{row.get('quarter', 'unknown')}/{row.get('region', 'unknown')}",
                    'issue': f'{col} is null',
                    'action_taken': 'Flagged for review'
                })

    # 2. Check for negative bookings
    negative_bookings = df[df['bookings'] < 0]
    if len(negative_bookings) > 0:
        for _, row in negative_bookings.iterrows():
            issues.append({
                'check_type': 'Negative bookings',
                'table': 'historical_bookings',
                'record_id': f"{row['quarter']}/{row['region']}",
                'issue': f"Negative booking value: {row['bookings']}",
                'action_taken': 'Flagged for review - may indicate reversal or data error'
            })

    return df, issues

def load_all_clean_data():
    """Master function - loads and cleans all three tables"""
    opps_raw, bookings_raw, forecast_raw = load_raw_data()

    clean_opps, opp_issues = clean_opportunities(opps_raw)
    clean_forecast, forecast_issues = clean_rep_forecast(forecast_raw)
    clean_bookings, bookings_issues = clean_historical_bookings(bookings_raw)

    all_issues = opp_issues + forecast_issues + bookings_issues

    return clean_opps, clean_bookings, clean_forecast, all_issues

if __name__ == "__main__":
    opps, bookings, forecast, all_issues = load_all_clean_data()

    print(f"Opportunities: {len(opps)} rows")
    print(f"Historical Bookings: {len(bookings)} rows")
    print(f"Rep Forecast: {len(forecast)} rows")
    print(f"\nTotal issues found: {len(all_issues)}")

    print("\n=== ISSUES BY TYPE ===")
    issue_types = {}
    for issue in all_issues:
        t = issue['check_type']
        issue_types[t] = issue_types.get(t, 0) + 1
    for k, v in issue_types.items():
        print(f"  {k}: {v}")

    print("\nData loaded and cleaned successfully!")