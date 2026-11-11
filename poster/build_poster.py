"""Build the current Portfolio Risk Lab poster from completed application results."""
from pathlib import Path
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/portfolio_risk_lab_poster.pdf';OUT.parent.mkdir(parents=True,exist_ok=True)
W,H=2592,1728;c=canvas.Canvas(str(OUT),pagesize=(W,H));c.setTitle('Portfolio Risk Lab: Do Better Forecasts Improve Portfolio Decisions?');c.setAuthor('Han')
INK='#17324D';BLUE='#2E608E';MUTED='#50677D';PALE='#EDF3F8';LINE='#D7E2EB'
def rect(x,y,w,h,color):
 c.setFillColor(HexColor(color));c.rect(x,H-y-h,w,h,fill=1,stroke=0)
def text(t,x,y,size=25,color=INK,bold=False):
 c.setFillColor(HexColor(color));c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.drawString(x,H-y-size,t)
def para(t,x,y,w=770,size=24,maxh=200):
 p=Paragraph(t,ParagraphStyle('p',fontName='Helvetica',fontSize=size,leading=size*1.32,textColor=HexColor(INK)))
 _,h=p.wrap(w,maxh);assert h<=maxh,(t[:40],h,maxh);p.drawOn(c,x,H-y-h)
def section(num,title,x,y):
 text(num,x,y,22,BLUE,True);text(title,x+45,y-3,28,INK,True);rect(x,y+43,770,2,LINE)
def table(x,y,rows,widths,rowh=52):
 for i,row in enumerate(rows):
  rect(x,y+i*rowh,sum(widths),rowh,PALE if i%2==0 else '#FFFFFF');off=0
  for value,width in zip(row,widths):para(str(value),x+off+9,y+i*rowh+10,width-18,20,maxh=rowh-12);off+=width
metrics=pd.read_csv(ROOT/'results/PORTFOLIO_V1.csv')
g=metrics.query('preset == "Diversified five" and protocol == "static_matched" and model == "Ridge_sequence_market" and cost_bps == 10').set_index('strategy')
rect(0,0,W,H,'#FFFFFF');rect(0,0,W,13,INK)
text('HAN  /  STAT UN3108 - APPLIED DEEP LEARNING AND AI',65,35,23,MUTED)
text('Portfolio Risk Lab',65,88,73,INK,True)
text('Do better forecasts improve portfolio decisions?',65,180,48,INK,True)
rect(65,269,2460,86,PALE)
para('<b>Exploratory finding:</b> forecast-based exposure can change returns without improving risk-target tracking. Read performance together with equity exposure and trading costs.',88,285,2408,28,maxh=70)
X1,X2,X3=65,900,1735
section('01','Problem and research question',X1,396)
para('A portfolio holder needs to decide how much equity risk to carry. Can model-predicted volatility improve risk-target tracking over a historical-volatility rule, after execution delays and trading costs?',X1,457,size=25,maxh=170)
section('02','An auditable decision experiment',X1,649)
steps=[('INPUT','Stock basket, weights, risk target and cost assumption.'),('ESTIMATE','Combine marginal volatilities with historical correlations.'),('COMPARE','Historical-risk versus forecast-risk equity/cash rules.'),('REVEAL','Inspect net returns, risk, drawdown, exposure and costs.')]
for i,(label,body) in enumerate(steps):
 y=714+i*123;rect(X1,y,770,106,PALE);text(label,X1+18,y+13,19,BLUE,True);para(body,X1+18,y+42,732,22,maxh=61)
section('03','Data and scope',X1,1250)
para('Thirty fixed US non-financial large-cap stocks; daily adjusted prices and saved V2 forecasts. Two five-stock baskets, four historical protocols, two forecasting models and three cost assumptions: <b>48 scenarios / 192 strategy records</b>.<br/><br/>Existing Ridge and LSTM are comparators. A recent-method replication and a new holdout remain future work.',X1,1310,size=23,maxh=235)
section('04','Portfolio and execution model',X2,396)
para('<b>Risk estimate:</b> Sigma = diag(s) R diag(s), using sixty-session Ledoit-Wolf historical correlation R. Both risk rules share R; s is either historical or forecast volatility.<br/><br/><b>Equity fraction:</b> min(1, target / portfolio volatility). Keep the selected stock proportions; place the remainder in cash. No leverage or shorting.',X2,457,size=24,maxh=255)
rect(X2,745,770,112,PALE)
para('<b>Timing:</b> observe close t &rarr; execute close t+1 &rarr; new holdings earn returns from t+2. Rebalance every twenty signal sessions.',X2+16,762,738,23,maxh=86)
para('Holdings drift between trades. Deduct proportional costs from traded stock dollars, including initial entry, with self-financing accounting. Cash earns zero; positions are marked to market at the end.',X2,887,size=23,maxh=145)
section('05','Baselines and evaluation',X2,1065)
para('<b>Four strategies:</b> buy and hold; fixed half-equity; historical-risk rule; forecast-risk rule.<br/><br/><b>Primary:</b> average absolute gap between rolling twenty-session net volatility and the target.<br/><br/><b>Secondary:</b> net return, drawdown, turnover, cash costs and average equity exposure. Fixed half-equity illustrates exposure effects; it is not a matched-risk comparison.',X2,1125,size=23,maxh=300)
para('Conditional paired date-block intervals accompany the full scenario report. Overlapping periods and scenarios are not independent replications.',X2,1453,size=20,maxh=85)
section('06','One transparent application case',X3,396)
para('Diversified five / 2024-25 / Sequence Ridge.<br/>Target: 10% annual volatility. Cost: 10 bps per stock dollar traded. This is a declared illustration, not a selected winning strategy.',X3,457,size=23,maxh=130)
rows=[['Strategy','Net return','Volatility','Avg equity']]
labels={'buy_hold':'Buy & hold','fixed_half':'Half equity','historical':'Historical rule','forecast':'Forecast rule'}
for k,label in labels.items():
 r=g.loc[k];rows.append([label,f'{r.total_return:.1%}',f'{r.realized_volatility:.1%}',f'{r.mean_equity_fraction:.1%}'])
table(X3,619,rows,[266,168,168,168])
a,b=g.loc['forecast'],g.loc['historical']
para(f'<b>Read the tradeoff:</b> forecast-rule return is {100*(a.total_return-b.total_return):.2f} percentage points higher, but average equity exposure is also {100*(a.mean_equity_fraction-b.mean_equity_fraction):.2f} points higher. Risk-target error is {100*a.risk_target_mae:.2f} versus {100*b.risk_target_mae:.2f} points. This case does not demonstrate better risk tracking.',X3,916,size=23,maxh=190)
section('07','Interpretation and next test',X3,1150)
para('The application makes model-dependent allocation rules observable and testable. It does not establish universal forecast superiority or alpha.<br/><br/>Next: reproduce one recent time-series method, audit pretraining overlap, and evaluate a fixed decision protocol on new observations. Test cash-yield, execution and cost sensitivity before stronger investment claims.',X3,1210,size=23,maxh=280)
rect(65,1580,2460,2,LINE)
para('<b>Limits.</b> Previously inspected periods; survivor sample; revised vendor prices; historical correlation; zero cash yield; simplified costs; no taxes or liquidity model. Forecast labels and tradable windows differ by the execution delay. Interactive scenarios are exploratory. Lower exposure alone can reduce risk.',65,1600,1530,17,maxh=95)
para('<b>Reproducibility.</b> config/portfolio-v1.json; docs/PORTFOLIO_PROTOCOL.md; results/PORTFOLIO_V1.csv. Forecasting V1-V3 remains preliminary evidence.<br/><b>Method reference.</b> Ledoit-Wolf covariance, scikit-learn. Han; AI coding assistance disclosed. Application draft, not final confirmatory results.',1640,1600,886,16,maxh=95)
c.showPage();c.save();print(OUT)
