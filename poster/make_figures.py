"""Scientific poster figures derived solely from saved v2 results."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from volrisk.io import read_json
root=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(root/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#19334e','text.color':'#19334e','xtick.color':'#41546a','ytick.color':'#41546a','axes.edgecolor':'#b9c4cf','savefig.facecolor':'white'})
out=root/'poster/assets';out.mkdir(exist_ok=True)
metrics=pd.read_csv(root/'results/ROBUSTNESS_V2.csv')
def save(fig,name):
    fig.savefig(out/f'{name}.png',dpi=240,bbox_inches='tight',pad_inches=.12)
    fig.savefig(out/f'{name}.svg',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)

fig,ax=plt.subplots(figsize=(9.7,5.0))
x=np.arange(3);w=.29
for off,model,label,color in [(-w/2,'Ridge_sequence_market','Sequence Ridge','#7791aa'),(w/2,'LSTM_market','Market LSTM','#245d95')]:
    vals=[]
    for p in ['origin_2020','origin_2022','origin_2024']:
        g=metrics[metrics.protocol.eq(p)].set_index('model')
        vals.append(g.loc[model,'qlike']/g.loc['Persistence','qlike'])
    bars=ax.bar(x+off,vals,w,color=color,label=label)
    ax.bar_label(bars,fmt='%.2f',padding=5,fontsize=15)
ax.axhline(1,color='#a37440',linestyle='--',lw=1.7,label='Persistence = 1')
ax.set(xticks=x,xticklabels=['2020','2022','2024'],ylim=(0,1.52),ylabel='QLIKE / persistence QLIKE',xlabel='Evaluation year')
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=3,frameon=False,fontsize=12)
ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True);fig.tight_layout();save(fig,'time_robustness')

fig,ax=plt.subplots(figsize=(9.6,4.25))
protos=['static_matched','origin_2020','origin_2022','origin_2024','unseen_companies']
labels=['Static 2024–25','2020','2022','2024','8 excluded firms']
for i,p in enumerate(protos):
    ci=read_json(root/f'results/robustness/v2/{p}/diagnostics.json')['paired_qlike_intervals']['LSTM_fundamentals minus LSTM_missingness']['20']
    v,lo,hi=ci['mean_difference'],ci['ci95_low'],ci['ci95_high']
    ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#245d95',capsize=5,lw=2.3,markersize=8)
ax.axvline(0,color='#a37440',linestyle='--',lw=1.5)
ax.set(yticks=np.arange(5),yticklabels=labels,xlim=(-.04,.085),xlabel='QLIKE difference: full financials − coverage control')
ax.invert_yaxis();ax.grid(axis='x',alpha=.13);fig.tight_layout();save(fig,'financial_controls')

fig,ax=plt.subplots(figsize=(9.5,4.1))
g=metrics[metrics.protocol.eq('unseen_companies')].set_index('model')
models=['Persistence','Ridge_sequence_market','LSTM_market','LSTM_missingness','LSTM_fundamentals']
labels=['Persistence','Sequence Ridge','Market LSTM','LSTM + coverage','LSTM + financials']
bars=ax.barh(labels,g.loc[models,'qlike'],color=['#c4cdd5','#7791aa','#245d95','#438f8c','#b18b5c'])
ax.bar_label(bars,fmt='%.3f',padding=6,fontsize=15);ax.invert_yaxis();ax.set(xlim=(0,.47),xlabel='QLIKE (lower is better)');ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True);fig.tight_layout();save(fig,'company_holdout')

fig=plt.figure(figsize=(11,1.25));fig.text(.02,.4,r'$y_{i,t}=\sqrt{\frac{252}{20}\sum_{k=1}^{20}(r_{i,t+k}-\bar{r}_{i,t+1:t+20})^2}$',fontsize=28,color='#19334e');save(fig,'target_formula')
print(out)
