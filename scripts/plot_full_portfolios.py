"""Decision tradeoff chart from the completed primary LSTM portfolio cases."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/han-mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
f=pd.read_csv(ROOT/'results/FULL_PORTFOLIO_V1.csv');f=f[(f.cost_bps==10)&(f.cash_annual==0)&f.model.eq('LSTM60_calibrated')]
fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
styles={'historical':('Historical-risk rule','#64748b','s'),'fixed_validation':('Validation-fixed equity','#296d9a','X'),'forecast':('LSTM dynamic equity','#173b5e','o')}
for ax,(basket,g) in zip(axes,f.groupby('basket',sort=False)):
    for _,r in g[g.strategy.isin(styles)].iterrows():
        label,color,marker=styles[r.strategy];ax.scatter(r.risk_target_mae*100,r.total_return*100,s=85,color=color,marker=marker,label=label,zorder=3)
    ax.set_title(basket,fontsize=11);ax.set_xlabel('Risk-target MAE (percentage points; lower is better)');ax.set_ylabel('Net cumulative return (%)');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.margins(.2)
axes[0].legend(fontsize=8,loc='best')
fig.suptitle('Does dynamic forecasting improve the decision?',fontsize=14)
fig.supxlabel('Exploratory 2024–2025 replay · 10% risk target · 10 bps costs · zero cash yield\nEquity exposures differ; points do not establish timing skill or alpha.',fontsize=8)
fig.savefig(ROOT/'results/FULL_PORTFOLIO_V1.png',dpi=160);plt.close(fig)
