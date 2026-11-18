"""Display fixed-policy decision tradeoffs without selecting favorable model variants."""
from pathlib import Path
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from volrisk.io import read_json, write_json, sha256, utc_now


def main():
    path = ROOT / 'results/MULTIMODAL_PORTFOLIO_V1.csv'
    audit = read_json(ROOT / 'results/MULTIMODAL_PORTFOLIO_AUDIT_V1.json')
    frame = pd.read_csv(path)
    if len(frame) != audit['strategy_records'] or frame.duplicated(['basket', 'model', 'cost_bps', 'cash_annual', 'strategy']).any():
        raise ValueError('Incomplete/duplicate decision grid')
    primary = frame.loc[frame.cost_bps.eq(10) & frame.cash_annual.eq(0)]
    selected = [f'{b}_{g}_calibrated' for b in ['Ridge', 'LSTM'] for g in ['market', 'financial', 'text', 'all']]
    checks = []
    for comparison in audit['comparisons']:
        case = primary.loc[primary.basket.eq(comparison['basket']) & primary.model.eq(comparison['model'])].set_index('strategy')
        for metric, field in [('risk_target_mae', 'risk_error_difference'), ('total_return', 'return_difference'), ('mean_equity_fraction', 'equity_difference')]:
            checks.append(abs(case.loc['forecast', metric] - case.loc[comparison['benchmark'], metric] - comparison[field]))
    if max(checks) > 1e-10:
        raise ValueError('Decision contrast reconciliation failed')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    labels = [name.replace('_calibrated', '').replace('_', ' · ') for name in selected]
    for axis, basket in zip(axes, audit['plan']['baskets']):
        for i, name in enumerate(selected):
            row = next(r for r in audit['comparisons'] if r['basket'] == basket and r['model'] == name and r['benchmark'] == 'fixed_validation')
            interval = row['intervals']['60']
            point = 100 * row['risk_error_difference']
            axis.errorbar(point, i, xerr=[[point - 100 * interval['ci95_low']], [100 * interval['ci95_high'] - point]],
                fmt='o', capsize=3, color='#173b67' if name.startswith('Ridge') else '#148691', markersize=5)
        axis.axvline(0, color='#8895a6', linestyle='--', linewidth=1)
        axis.set_title(basket, fontsize=11)
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=9)
        axis.set_xlabel('Dynamic minus fixed-equity risk error (pp)\nNegative favors the dynamic rule', fontsize=9)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.15)
    axes[0].invert_yaxis()
    fig.suptitle('Decision value after costs · 60-date conditional intervals', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(ROOT / 'results/MULTIMODAL_PORTFOLIO_V1.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    lines = ['# Multimodal forecasts to portfolio decisions', '',
        'The frozen application grid evaluates1,632 strategy records:two baskets,34 raw/calibrated forecast procedures,three cost rates,two hypothetical cash yields andfour strategies. Basket membership and policy rules were registered during model training before the matched evaluation scores. The calendar periods were already inspected in prior research, so this remains exploratory.', '',
        'Primary setting:10% annual volatility target,10bps per dollar traded,zero cash return,20-session rebalance and60-session historical correlations. Signal after close,execution at the following close; new holdings earn subsequent returns only. The comparison uses identical prices,dates and stock baskets. Stocks are long-only and unlevered. The constant baseline is frozen from mean planned validation equity; it does not match realized evaluation exposure.', '',
        '## Decision findings', '',
        'In the ten-sector basket, calibrated dynamic rules have modestly lower mean risk-target error than their model-specific fixed-equity controls, but the displayed date-block intervals cross zero. They also earn lower net returns than those controls. In the technology trio, calibrated LSTM dynamic rules have higher mean risk error than fixed equity; the all-modality LSTM difference is positive in both block intervals. The all-modality Ridge interval crosses zero. These results do not establish reliable dynamic superiority, equivalent risk exposure or alpha.', '',
        'Improved forecast loss is not automatically improved allocation. Exposure bounds,historical correlation estimates,asset concentration and calibration mediate the decision. The language extension does not establish incremental practical benefit in this registered application.', '',
        '| Basket | Calibrated model | Risk-error Δ(pp) | Net-return Δ(pp) | Mean-equity Δ(pp) |60-date risk-error95% interval(pp) |', '|---|---|---:|---:|---:|---|']
    for basket in audit['plan']['baskets']:
        for name in selected:
            r = next(x for x in audit['comparisons'] if x['basket'] == basket and x['model'] == name and x['benchmark'] == 'fixed_validation')
            ci = r['intervals']['60']
            lines.append(f'| {basket} | {name.replace("_calibrated", "")} | {100*r["risk_error_difference"]:+.3f} | {100*r["return_difference"]:+.2f} | {100*r["equity_difference"]:+.2f} | [{100*ci["ci95_low"]:+.3f}, {100*ci["ci95_high"]:+.3f}] |')
    lines += ['', 'All differences are dynamic minus that procedure\'s validation-fixed equity control. The same market/financial/text/all information groups are shown for both backbones; metadata controls,raw forecasts and all unfavorable cost/cash outcomes remain in the complete CSV. The audit also reports comparisons against the historical-risk rule.', '',
        '## Complete outcome contract', '',
        'The CSV includes risk-target absolute error,total and annualized return,realized volatility,max drawdown,mean equity fraction,turnover and costs. Zero,10 and25bps costs are retained;4% cash yield is a hypothetical ACT/365 sensitivity,not observed Treasury returns. Risk tracking uses rolling20-session net portfolio simple-return volatility; forecast targets use single-stock log-return volatility. These are related,distinct outcomes.', '',
        'Intervals use paired circular20/60-session blocks of daily risk-error differences,conditional on fixed procedures and selected baskets. They do not include full model retraining uncertainty,multiplicity adjustment or an uninspected temporal sample. Revised prices,provider/survivor selection,no taxes/liquidity/market-impact model and fixed correlation estimates remain limitations. The2026 market holdout remains sealed.', '']
    (ROOT / 'results/MULTIMODAL_PORTFOLIO_V1.md').write_text('\n'.join(lines))
    write_json(ROOT / 'results/MULTIMODAL_PORTFOLIO_CHECKS_V1.json', {'completed_utc': utc_now(),
        'source_sha256': sha256(Path(__file__)), 'results_sha256': sha256(path),
        'strategy_records': len(frame), 'max_contrast_reconciliation_difference': max(checks),
        'market_outcomes_2026_accessed': False})
    print('Reconciled', len(frame), 'strategy records; max contrast difference', max(checks))


if __name__ == '__main__':
    main()
