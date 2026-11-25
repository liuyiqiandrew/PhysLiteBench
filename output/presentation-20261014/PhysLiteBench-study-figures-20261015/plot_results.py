"""Generate every scientific chart used in the unified presentation.
Run aggregate_results.py first. All paths are relative to this bundle.
Python 3 + matplotlib + numpy. No network, model calls, smoothing or curve fitting.
Frontier segments connect observed nondominated points; cost axes are logarithmic.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
D=json.loads((ROOT/'summary.json').read_text());R=D['summary'];M=D['metadata']
for p in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','font.size':18,'axes.labelsize':20,'axes.titlesize':22,'legend.fontsize':17,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#777777','axes.linewidth':.8,'savefig.facecolor':'white'})
MODELS=['gpt-5.6-luna','gpt-5.6-terra','gpt-5.6-sol','gpt-6-luna','gpt-6-sol','gpt-6-astra','deepseek/deepseek-v4.1-flash']
NAMES={'gpt-5.6-luna':'GPT-5.6 Luna','gpt-5.6-terra':'GPT-5.6 Terra','gpt-5.6-sol':'GPT-5.6 Sol','gpt-6-luna':'GPT-6 Luna','gpt-6-sol':'GPT-6 Sol','gpt-6-astra':'GPT-6 Astra','deepseek/deepseek-v4.1-flash':'DeepSeek V4.1 Flash'}
MC=dict(zip(MODELS,['#477DA5','#B45309','#607D35','#1D4E89','#0F766E','#5B4B81','#B00032']))
COND=['baseline','general','physics'];CC=dict(zip(COND,['#1D4E89','#B45309','#0F766E']));CN=dict(zip(COND,['No hint','General hint','Physics hint']))
EFF=['none','low','medium','high','xhigh','max'];EN=dict(zip(EFF,['None','Low','Medium','High','Extra High','Max']));ES=dict(zip(EFF,['N','L','M','H','X','Max']));EM=dict(zip(EFF,['v','o','D','s','^','P']))
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=240)
 plt.close(fig)
def axis(ax,cost=True):
 ax.grid(axis='y',color='#D8DCE1',lw=.75);ax.set_axisbelow(True)
 ax.set_ylim(-3,109);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('Completed pass rate (%)')
 if cost:
  ax.set_xscale('log');ax.set_xlabel('Estimated API cost per attempt (USD, log scale)')
  ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos:f'${x:g}'))
  ax.xaxis.set_minor_formatter(ticker.NullFormatter())

def front(rows):
 a=[r for r in rows if r['cost_complete']]
 return sorted([r for r in a if not any(q['mean_cost_usd']<=r['mean_cost_usd'] and q['pass_rate']>=r['pass_rate'] and (q['mean_cost_usd']<r['mean_cost_usd'] or q['pass_rate']>r['pass_rate']) for q in a)],key=lambda x:x['mean_cost_usd'])

# 1: one no-hint view of all seven exact model versions. Unknown costs are omitted.
fig,ax=plt.subplots(figsize=(13.2,5.25));fig.subplots_adjust(left=.08,right=.78,top=.96,bottom=.17)
a=[r for r in R if r['condition']=='baseline'];axis(ax);ax.set_xlim(.0032,1.5);ax.set_xticks([.004,.01,.03,.1,.3,1])
f=front(a);ax.plot([r['mean_cost_usd'] for r in f],[100*r['pass_rate'] for r in f],ls='--',lw=2,color='#333333',zorder=1,solid_joinstyle='miter')
for r in a:
 if not r['cost_complete']:continue
 ax.scatter(r['mean_cost_usd'],100*r['pass_rate'],s=95,marker=EM[r['effort']],color=MC[r['model']],edgecolor='white',linewidth=.6,zorder=3)
for r in f:
 txt=NAMES[r['model']].replace('GPT-','')+' '+EN[r['effort']]
 off={'low':(8,11),'high':(-8,12),'xhigh':(9,9)}[r['effort']]
 if r['model']=='gpt-6-sol' and r['effort']=='high':off=(10,-26)
 ax.annotate(txt,(r['mean_cost_usd'],100*r['pass_rate']),xytext=off,textcoords='offset points',fontsize=17,color=MC[r['model']])
leg=[Line2D([],[],marker='o',ls='',color=MC[m],label=NAMES[m],ms=8) for m in MODELS if any(r['model']==m and r['cost_complete'] for r in a)]
l1=ax.legend(handles=leg,loc='upper left',bbox_to_anchor=(1.01,1.02),frameon=False,borderaxespad=0,fontsize=16);ax.add_artist(l1)
ax.legend(handles=[Line2D([],[],marker=EM[e],ls='',color='#555555',label=EN[e],ms=7) for e in EFF],loc='lower left',bbox_to_anchor=(1.01,-.01),frameon=False,ncol=2,columnspacing=.7,handletextpad=.3,fontsize=15,borderaxespad=0)
save(fig,'01-no-hint-frontier')

# 2: fixed model/effort coverage across prompt conditions; exact-cost frontiers only.
keys={(r['model'],r['effort']) for r in R if all(any(s['model']==r['model'] and s['effort']==r['effort'] and s['condition']==c for s in R) for c in COND)}
a=[r for r in R if (r['model'],r['effort']) in keys]
MM={'gpt-5.6-luna':'o','gpt-5.6-terra':'s','gpt-5.6-sol':'^','deepseek/deepseek-v4.1-flash':'D'}
fig,ax=plt.subplots(figsize=(13.2,5.25));fig.subplots_adjust(left=.08,right=.78,top=.96,bottom=.17);axis(ax);ax.set_xlim(.006,.5);ax.set_xticks([.008,.02,.05,.1,.2,.4])
for c in COND:
 f=front([r for r in a if r['condition']==c]);ax.plot([r['mean_cost_usd'] for r in f],[100*r['pass_rate'] for r in f],color=CC[c],lw=2,alpha=.8,zorder=1,solid_joinstyle='miter')
 for r in [r for r in a if r['condition']==c and r['cost_complete']]:
  ax.scatter(r['mean_cost_usd'],100*r['pass_rate'],s=100,marker=MM[r['model']],color=CC[c],edgecolor='white',linewidth=.65,zorder=3)
  # Effort labels are literal observations, never fitted/smoothed values.
  if r['model']=='gpt-5.6-luna' and r['effort'] in ['none','low'] and not (c=='physics' and r['effort']=='low'):continue
  off=(6,4)
  if r['model']=='gpt-5.6-luna' and r['effort']=='none':off=(5,-13)
  if r['model']=='gpt-5.6-sol' and c=='physics':off=(-23,3)
  if r['model']=='gpt-5.6-terra' and c=='physics':off=(7,-2)
  ax.annotate(ES[r['effort']],(r['mean_cost_usd'],100*r['pass_rate']),xytext=off,textcoords='offset points',fontsize=14,color=CC[c])
l1=ax.legend(handles=[Line2D([],[],color=CC[c],lw=2,label=CN[c]) for c in COND],loc='upper left',bbox_to_anchor=(1.01,1.02),frameon=False,borderaxespad=0);ax.add_artist(l1)
ax.legend(handles=[Line2D([],[],marker=MM[m],ls='',color='#555555',label=NAMES[m].replace('GPT-5.6 ',''),ms=9) for m in MM if m!='deepseek/deepseek-v4.1-flash'],title='GPT-5.6 models',loc='upper left',bbox_to_anchor=(1.01,.66),frameon=False,borderaxespad=0,fontsize=16)
fig.text(.80,.10,'Effort labels\nL: Low    M: Medium\nH: High    X: Extra High\nMax: Max',fontsize=15.5,linespacing=1.5)
save(fig,'02-hint-frontiers')

# 3: one Luna model, all effort and hint settings, pooled unique observations.
fig,ax=plt.subplots(figsize=(13.2,5.25));fig.subplots_adjust(left=.08,right=.99,top=.91,bottom=.22);axis(ax,False)
x=np.arange(6);w=.25
for i,c in enumerate(COND):
 a=[next(r for r in R if r['model']=='gpt-5.6-luna' and r['effort']==e and r['condition']==c) for e in EFF]
 xx=x+(i-1)*w;ax.bar(xx,[100*r['pass_rate'] for r in a],width=w*.94,color=CC[c],label=CN[c])
 for xp,r in zip(xx,a):ax.text(xp,100*r['pass_rate']+2.4,f"{r['passes']}/{r['attempts']}",ha='center',va='bottom',fontsize=14.5,color='#333333')
ax.set_xticks(x,[EN[e] for e in EFF]);ax.set_xlim(-.55,5.55);ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False)
save(fig,'03-luna-effort-hints')

# 4: all recorded outcomes by model. Scientific diagnoses appear separately on slide 9.
fig,ax=plt.subplots(figsize=(8.7,5.3));fig.subplots_adjust(left=.255,right=.96,top=.98,bottom=.24)
order=MODELS[::-1];cats=['passes','graded_failures','errors'];colors=['#0F766E','#B45309','#B00032'];labels=['Completed pass','Graded failure','Execution error']
y=np.arange(len(order));left=np.zeros(len(order));counts={m:sum(r['attempts'] for r in R if r['model']==m) for m in order}
for cat,col,label in zip(cats,colors,labels):
 v=np.array([100*sum(r[cat] for r in R if r['model']==m)/counts[m] for m in order]);ax.barh(y,v,left=left,color=col,height=.66,label=label);left+=v
ax.set_yticks(y,[NAMES[m].replace('GPT-','')+f"\n(n={counts[m]})" for m in order]);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Share of attempts (%)');ax.grid(axis='x',color='#D8DCE1',lw=.75);ax.set_axisbelow(True);ax.tick_params(axis='y',labelsize=16);ax.legend(loc='upper center',bbox_to_anchor=(.44,-.16),ncol=3,frameon=False,fontsize=14,handlelength=1,handletextpad=.4,columnspacing=1)
save(fig,'04-outcomes-by-model')

# 5: same analytic data as the user-edited bead chart; no fitted interpolation.
a=json.loads((ROOT/'sources/bead-analytic-series.json').read_text());fig,ax=plt.subplots(figsize=(6.1,4.0));fig.subplots_adjust(left=.14,right=.98,top=.97,bottom=.34)
for r,c,l in zip(a,['#B00032','#0F766E'],['Starter formula','Correct physics']):ax.plot(r['x'],r['y'],color=c,lw=2,label=l,solid_joinstyle='miter')
ax.set(xlim=(1,8),ylim=(0,1.02),xlabel='Later time / earlier time',ylabel='Covariance (normalized)');ax.set_yticks([0,.25,.5,.75,1]);ax.grid(axis='y',color='#D8DCE1',lw=.75);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.29),ncol=2,frameon=False,fontsize=16);ax.tick_params(labelsize=15);ax.xaxis.label.set_fontsize(17);ax.yaxis.label.set_fontsize(17)
save(fig,'05-bead-covariance')

# 6: all available no-hint efforts. Sparse settings remain sparse.
fig,ax=plt.subplots(figsize=(13.2,5.25));fig.subplots_adjust(left=.08,right=.78,top=.96,bottom=.18);axis(ax,False)
for m in MODELS:
 a=sorted([r for r in R if r['model']==m and r['condition']=='baseline'],key=lambda r:EFF.index(r['effort']))
 ax.plot([EFF.index(r['effort']) for r in a],[100*r['pass_rate'] for r in a],marker='o',lw=1.8,ms=7,color=MC[m],label=NAMES[m],solid_joinstyle='miter')
ax.set_xticks(range(6),[EN[e] for e in EFF]);ax.set_xlim(-.2,5.2);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1.02),frameon=False,borderaxespad=0,fontsize=16)
save(fig,'06-no-hint-efforts')
# Full data and records, including figure eligibility, accompany this index.
index={'matplotlib':matplotlib.__version__,'numpy':np.__version__,'cost_axis':'log10 on all frontier figures','curve_smoothing':False,'point_metric':'Completed passes / all attempts, exceptions retained in denominator','aggregation':'Exact model version + reasoning effort + reported prompt condition; remove 38 known duplicate trial identities','figure_names':['01-no-hint-frontier','02-hint-frontiers','03-luna-effort-hints','04-outcomes-by-model','05-bead-covariance','06-no-hint-efforts'],'exact_no_hint_points':sum(r['condition']=='baseline' and r['cost_complete'] for r in R),'exact_hint_comparison_points':sum(r['cost_complete'] for r in a) if False else sum(r['cost_complete'] and (r['model'],r['effort']) in keys for r in R),'hint_comparison_setting_count':3*len(keys)}
(ROOT/'figure-index.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps(index,indent=2))
