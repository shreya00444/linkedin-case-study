import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

def generate_tab_insights(accuracy_df, coverage_df, gap_df, issues):
    """
    Generates one insight per tab — 6 words max each.
    """
    meaningful = accuracy_df[accuracy_df['total_weighted_forecast'] > 0]
    total_pipeline = coverage_df['total_pipeline'].sum()
    total_target = coverage_df['avg_q3_bookings'].sum()
    coverage_ratio = round(total_pipeline / total_target, 2) if total_target > 0 else 0
    total_gap = coverage_df['gap'].sum()
    open_deals = gap_df[~gap_df['stage'].isin(['Closed Won', 'Closed Lost'])]
    total_disagreement = open_deals['gap'].abs().sum()
    misaligned = len(open_deals[open_deals['gap_direction'] != 'Aligned'])
    stale_count = len([i for i in issues if i['check_type'] == 'Stale Deal'])
    total_issues = len(issues)

    context = f"""
You are a GTM analytics expert. Generate exactly 4 insights — one per section below.
Each insight must be 6 words or fewer. Be specific. Use numbers. No punctuation at end.

Data:
- Total data issues: {total_issues}
- Stale deals: {stale_count} inflating pipeline by $2.8M
- NAMER Enterprise forecast accuracy: 4.6%
- EMEA SMB over-forecasting by 4.5x
- Q3 pipeline coverage: {coverage_ratio}x vs 3x target
- Pipeline gap: ${total_gap:,.0f}
- Rep vs CRM disagreement on open deals: ${total_disagreement:,.0f}
- Open deals misaligned: {misaligned}

Generate exactly 4 lines in this format — nothing else:
DATA: [insight about data quality]
ACCURACY: [insight about forecast accuracy]
COVERAGE: [insight about pipeline coverage]
GAP: [insight about rep vs CRM gap]
"""

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{"role": "user", "content": context}]
    )

    raw = message.content[0].text.strip()
    insights = {}
    for line in raw.split('\n'):
        if line.startswith('DATA:'):
            insights['data'] = line.replace('DATA:', '').strip()
        elif line.startswith('ACCURACY:'):
            insights['accuracy'] = line.replace('ACCURACY:', '').strip()
        elif line.startswith('COVERAGE:'):
            insights['coverage'] = line.replace('COVERAGE:', '').strip()
        elif line.startswith('GAP:'):
            insights['gap'] = line.replace('GAP:', '').strip()

    return insights
