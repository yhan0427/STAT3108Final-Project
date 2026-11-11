"""Summarize completed v2 runs without fitting or selecting new models."""
from pathlib import Path
import os
import pandas as pd
from volrisk.io import read_json

root=Path(__file__).resolve().parents[1]
folder=root/'results/robustness/v2'
assert (folder/'COMPLETED.json').exists(), 'Wait for all registered experiments to finish'
metrics=pd.read_csv(folder/'metrics.csv')
lines=['# Exploratory robustness results — v2','', 'These experiments were designed after inspecting v1. They are diagnostics, not an untouched final test or a published-paper replication.','']
lines += ['## Main findings', '', 'Matched-history Ridge is a competitive baseline. The LSTM advantage depends on year and metric. Complete accounting values do not show consistent incremental benefits over the missingness/filing-age control in these specifications. These results concern the implemented feature set with incomplete public accounting coverage, not all potential financial information.', '', '| Evaluation | Persistence QLIKE | Sequence Ridge QLIKE | Market LSTM QLIKE |', '|---|---:|---:|---:|']
for protocol in ['origin_2020','origin_2022','origin_2024','unseen_companies']:
    g=metrics[metrics.protocol.eq(protocol)].set_index('model')
    lines.append(f'| {protocol} | {g.loc["Persistence","qlike"]:.3f} | {g.loc["Ridge_sequence_market","qlike"]:.3f} | {g.loc["LSTM_market","qlike"]:.3f} |')
lines += ['', 'For the eight-company evaluation, full-fundamental LSTM has higher QLIKE than market-only LSTM; the paired intervals below quantify the difference. Do not compare its raw top-five overlap with the thirty-company protocols. Further scope expansion should first improve accounting coverage and predefine the new sample/evaluation design.', '']
for protocol,g in metrics.groupby('protocol',sort=False):
    diagnostics=read_json(folder/protocol/'diagnostics.json')
    n=len(diagnostics['evaluation_companies'])
    lines += [f'## {protocol}', '',f'{n} evaluation companies. Random top-five expected overlap: {5/n:.1%}. Compare screening scores only within this company pool.','', '| Model | MAE (pp) | RMSE (pp) | QLIKE | Top-five overlap |','|---|---:|---:|---:|---:|']
    for r in g.itertuples():
        lines.append(f'| {r.model} | {100*r.mae:.2f} | {100*r.rmse:.2f} | {r.qlike:.3f} | {r.top5_overlap:.1%} |')
    lines += ['', 'Paired QLIKE intervals: negative favors the first model.','']
    for pair,blocks in diagnostics['paired_qlike_intervals'].items():
        for block,ci in blocks.items():
            if ci: lines.append(f'- {pair}, {block}-day blocks: {ci["mean_difference"]:.4f}, 95% CI [{ci["ci95_low"]:.4f}, {ci["ci95_high"]:.4f}].')
    lines.append('')
lines += ['## Interpretation limits','','Three selected annual origins do not establish universal robustness. The eight held-out firms were not used for this experiment’s fitting or selection, but were previously examined in v1. Missingness controls can reveal reporting/identity effects; they do not establish causation. SEC accounting coverage, survivorship selection, vendor revisions, non-financial large-cap scope and daily-return volatility measurement remain limitations. A different period or company pool is a different forecasting task. Do not pool overlapping protocol rows or choose a model based on the best evaluation seed.','']
(root/'results/ROBUSTNESS_V2.md').write_text('\n'.join(lines))
metrics.to_csv(root/'results/ROBUSTNESS_V2.csv',index=False)
os.environ.setdefault('MPLCONFIGDIR',str(root/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
chosen=['Persistence','Ridge_sequence_market','Boosting_market','LSTM_market','LSTM_missingness','LSTM_fundamentals']
fig,axs=plt.subplots(1,2,figsize=(15,5))
for model in chosen:
    g=metrics[metrics.model.eq(model)].set_index('protocol').reindex(['origin_2020','origin_2022','origin_2024'])
    axs[0].plot([2020,2022,2024],g.qlike,marker='o',label=model)
axs[0].set(title='Selected expanding-window origins',ylabel='QLIKE (lower is better)',xticks=[2020,2022,2024]);axs[0].legend(fontsize=8)
g=metrics[metrics.protocol.eq('unseen_companies')].set_index('model').loc[chosen]
axs[1].barh(chosen,g.qlike,color=['#879ab0','#526c90','#367f93','#bc893e','#8e72a5','#386cc2']);axs[1].invert_yaxis();axs[1].set(title='Eight companies excluded from fitting',xlabel='QLIKE (lower is better)')
for ax in axs:
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Exploratory robustness diagnostics · not paper replication')
fig.tight_layout();fig.savefig(root/'results/ROBUSTNESS_V2.png',dpi=180);plt.close(fig)
print(root/'results/ROBUSTNESS_V2.md')
