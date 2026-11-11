"""Audit and summarize a completed, versioned alternative accounting experiment."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from volrisk.io import read_json, write_json, sha256
from volrisk.fundamentals import FUND_FEATURES
from volrisk.fundamentals_v3 import V3_FEATURES

root = Path(__file__).resolve().parents[1]
folder = root/'results/robustness/v3'
assert (folder/'COMPLETED.json').exists(), 'All registered runs must finish first'
old = pd.read_pickle(root/'data/processed/main/panel.pkl')
new = pd.read_pickle(root/'data/processed/financial-v3/panel.pkl')
pd.testing.assert_frame_equal(old, new[old.columns])
manifest = read_json(root/'data/processed/financial-v3/manifest.json')
assert sha256(root/'data/processed/main/panel.pkl') == manifest['parent_panel_sha256']
assert sha256(root/'data/processed/financial-v3/panel.pkl') == manifest['panel_sha256']
metrics = pd.read_csv(folder/'metrics.csv')
previous = pd.read_csv(root/'results/robustness/v2/metrics.csv')
checks = []
for case in read_json(root/'config/sec-spotchecks.json')['cases']:
    snapshots = pd.read_csv(root/f'data/processed/financial-v3/{case["ticker"]}_snapshots.csv')
    row = snapshots.loc[snapshots.filing_date.eq(case['filing_date']) & snapshots.report_end.eq(case['report_end'])].iloc[0]
    expected = case['current_liabilities']/case['assets']
    assert np.isclose(row.current_liabilities_to_assets, expected, rtol=1e-9)
    checks.append({'ticker':case['ticker'], 'feature':'current_liabilities_to_assets', 'expected':expected, 'actual':float(row.current_liabilities_to_assets), 'source':case['url']})
market_checks = {}
for protocol in metrics.protocol.unique():
    a = pd.read_csv(folder/protocol/'predictions.csv')
    b = pd.read_csv(root/'results/robustness/v2'/protocol/'predictions.csv')
    cols = ['date','ticker','target','target_end','Persistence','Ridge_sequence_market','Boosting_market','LSTM_market']
    pd.testing.assert_frame_equal(a[cols],b[cols])
    market_checks[protocol] = 'Identical evaluation rows, targets and market-only forecasts to v2'
write_json(root/'results/FINANCIAL_V3_AUDIT.json', {'original_columns_unchanged':True,'parent_panel_hash_matches':True,'panel_hash_matches':True,'market_controls':market_checks,'original_statement_checks':checks,'scope':'Two current-liability ratios in two original filings; not full accounting-panel certification.'})

coverage = []
for split in ['train','validation','test']:
    for version, features in [('v2',FUND_FEATURES),('v3',V3_FEATURES)]:
        for feature in features:
            coverage.append({'split':split,'schema':version,'feature':feature,'coverage':new.loc[new.split.eq(split),feature].notna().mean()})
pd.DataFrame(coverage).to_csv(root/'results/FINANCIAL_V3_COVERAGE.csv',index=False)

lines = ['# Alternative accounting features — exploratory v3', '',
    'Completed three registered protocols, thirteen model specifications each and twenty-seven neural seed fits. No company or text expansion. All original panel columns and all matched market-only forecasts are identical to v2. Original v1/v2 artifacts remain intact.', '',
    '## Definitions and coverage', '',
    'Replace total liabilities/assets with parent stockholders’ equity/assets; replace operating margin with current liabilities/assets. These are different economic features, not filled versions of the old features. The parent-equity measure excludes noncontrolling interests and must not be used to infer total liabilities. Negative equity is retained. The same eight-dimensional accounting input, missingness indicators, architecture, seeds and evaluation observations are used.', '',
    '| Field | Mean coverage across thirty issuers |', '|---|---:|']
c = pd.read_csv(root/'data/processed/financial-v3/coverage.csv').drop(columns='ticker').mean()
for f in ['liabilities_to_assets','operating_margin','parent_equity_to_assets','current_liabilities_to_assets','pretax_margin','liability_components_to_assets']:
    lines.append(f'| {f} | {c[f]:.1%} |')
lines += ['', 'Coverage is measured over eligible calendar-split rows and averaged across issuers, not independent observations. Pretax margin was rejected on coverage alone before model fitting. Current plus noncurrent liabilities are not used to fill total liabilities: both same-period components are available too rarely. Split-level coverage is in FINANCIAL_V3_COVERAGE.csv.', '',
    '## Matched QLIKE comparisons', '', 'Lower is better. Each LSTM forecast averages the same three registered seeds, without selecting evaluation winners.', '',
    '| Evaluation | Market LSTM (both schemas) | V2 missingness | V3 missingness | V2 full ratios | V3 full ratios |', '|---|---:|---:|---:|---:|---:|']
for protocol in metrics.protocol.unique():
    a = metrics[metrics.protocol.eq(protocol)].set_index('model')
    b = previous[previous.protocol.eq(protocol)].set_index('model')
    lines.append(f'| {protocol} | {a.loc["LSTM_market","qlike"]:.4f} | {b.loc["LSTM_missingness","qlike"]:.4f} | {a.loc["LSTM_missingness","qlike"]:.4f} | {b.loc["LSTM_fundamentals","qlike"]:.4f} | {a.loc["LSTM_fundamentals","qlike"]:.4f} |')
lines += ['', '## Within-v3 paired uncertainty', '', 'Daily mean loss differences; all firms on a date stay together in circular block resampling. Negative favors full financial values. These intervals are exploratory, conditional on the sample/fitted models, and do not capture retraining or unseen-company population uncertainty. Twenty-day labels overlap; no firm-day independence assumption.', '', '| Evaluation | Full LSTM minus missingness | 20-day block 95% interval | 60-day block 95% interval |', '|---|---:|---:|---:|']
for protocol in metrics.protocol.unique():
    ds = read_json(folder/protocol/'diagnostics.json')['paired_qlike_intervals']['LSTM_fundamentals minus LSTM_missingness']
    a,b = ds['20'],ds['60']
    lines.append(f'| {protocol} | {a["mean_difference"]:.4f} | [{a["ci95_low"]:.4f}, {a["ci95_high"]:.4f}] | [{b["ci95_low"]:.4f}, {b["ci95_high"]:.4f}] |')
lines += ['', '## Interpretation', '',
    'Higher coverage alone has not established a financial-value benefit. Feature content and missingness patterns change together, so this experiment cannot identify a pure coverage effect or a causal mechanism. Compare the full ratios with their own schema’s missingness/filing-age control, not only the market baseline. Missingness may encode issuer/reporting differences.', '',
    'The eight excluded companies were absent from this run’s fitting and validation but previously examined in v1. The 2020 protocol deliberately revisits a known failure period. The static and company-holdout evaluation periods overlap; do not pool them as independent replications. All three protocols were chosen after earlier results, so none is an untouched confirmatory test.', '',
    'Next: audit missing tags by issuer and accounting definition, then preregister a comparable-feature common-coverage sensitivity and a genuinely new company/time holdout. Do not add text until there is a defensible incremental-information question. A larger sample alone cannot fix feature semantics or reporting bias.', '',
    '## Verification', '',
    'Original columns and parent panel hashes match. Market-only predictions match v2 exactly. Two current-liability ratios match previously transcribed original SEC statements. Synthetic timing/negative-equity tests cover the new concepts. This is limited verification, not certification of the full point-in-time archive. Ridge emitted ill-conditioned float32 matrix warnings in the registered alpha grid; fitted results are retained and this numerical limitation should be checked before confirmatory use.', '',
    '## All registered model specifications', '']
for protocol, group in metrics.groupby('protocol',sort=False):
    lines += [f'### {protocol}', '', '| Model | MAE (pp) | QLIKE | Top-five overlap |', '|---|---:|---:|---:|']
    for row in group.itertuples(): lines.append(f'| {row.model} | {100*row.mae:.2f} | {row.qlike:.4f} | {row.top5_overlap:.1%} |')
    lines += ['', 'Eight-firm raw top-five overlap must not be compared with thirty-firm overlap; random expectations are respectively 62.5% and 16.7%.', '']
(root/'results/FINANCIAL_V3.md').write_text('\n'.join(lines))
metrics.to_csv(root/'results/FINANCIAL_V3.csv',index=False)
os.environ.setdefault('MPLCONFIGDIR',str(root/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,protocol in zip(axes,metrics.protocol.unique()):
    names=['LSTM_market','LSTM_missingness','LSTM_fundamentals']
    g=metrics[metrics.protocol.eq(protocol)].set_index('model').loc[names]
    ax.bar(['Market','Age + masks','Full ratios'],g.qlike,color=['#6783a3','#8798ac','#315f90'])
    ax.set(title=protocol,ylabel='QLIKE (lower is better)');ax.spines[['top','right']].set_visible(False)
fig.suptitle('Alternative accounting schema · exploratory v3');fig.tight_layout()
fig.savefig(root/'results/FINANCIAL_V3.png',dpi=180);plt.close(fig)
print(root/'results/FINANCIAL_V3.md')
