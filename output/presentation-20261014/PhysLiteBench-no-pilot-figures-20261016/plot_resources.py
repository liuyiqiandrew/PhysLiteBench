"""Plot all model/effort settings at recorded dollar/token cost, including incomplete usage.
Open markers are lower bounds; dollar bars end at conservative provider reserves.
Dashed envelopes are provisional because unknown usage can move points right.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
from matplotlib.lines import Line2D
U=Path(__file__).resolve().parent;O=U/'figures';D=json.loads((U/'resource-summary.json').read_text());R=D['summary'];audit=[]
for p in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','font.size':18,'axes.labelsize':20,'axes.titlesize':22,'legend.fontsize':16,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#777777','axes.linewidth':.8,'savefig.facecolor':'white'})
MODELS=['gpt-5.6-luna','gpt-5.6-terra','gpt-5.6-sol','gpt-6-luna','gpt-6-sol','gpt-6-astra','deepseek/deepseek-v4.1-flash'];NAMES=dict(zip(MODELS,['GPT-5.6 Luna','GPT-5.6 Terra','GPT-5.6 Sol','GPT-6 Luna','GPT-6 Sol','GPT-6 Astra','DeepSeek V4.1 Flash']))
MC=dict(zip(MODELS,['#477DA5','#B45309','#607D35','#1D4E89','#0F766E','#5B4B81','#B00032']))
C=['baseline','general','physics'];CC=dict(zip(C,['#1D4E89','#B45309','#0F766E']));CN=dict(zip(C,['No hint','General hint','Physics hint']))
E=['none','low','medium','high','xhigh','max'];EN=dict(zip(E,['None','Low','Medium','High','Extra High','Max']));ES=dict(zip(E,['N','L','M','H','X','Max']));EM=dict(zip(E,['v','o','D','s','^','P']));MM=dict(zip([MODELS[0],MODELS[1],MODELS[2],MODELS[-1]],['o','s','^','D']))
keys={(r['model'],r['effort']) for r in R if all(any((q['model'],q['effort'],q['condition'])==(r['model'],r['effort'],c) for q in R) for c in C)}
BASE=[r for r in R if r['condition']=='baseline'];HINT=[r for r in R if (r['model'],r['effort']) in keys]
def key(r):return [r['model'],r['effort'],r['condition']]
def front(a,metric):
 f='plot_'+metric
 return sorted([r for r in a if not any(q[f]<=r[f] and q['pass_rate']>=r['pass_rate'] and (q[f]<r[f] or q['pass_rate']>r['pass_rate']) for q in a)],key=lambda r:r[f])
def axis(ax,metric,scope):
 ax.set_xscale('log');ax.set_ylim(-5,110);ax.set_yticks([0,20,40,60,80,100]);ax.grid(axis='y',color='#D8DCE1',lw=.7);ax.set_axisbelow(True)
 if metric=='usd':
  ax.set_xlim((.003,1.7) if scope=='base' else (.006,.5));ax.set_xticks([.004,.01,.03,.1,.3,1] if scope=='base' else [.008,.02,.05,.1,.2,.4]);ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,p:f'${x:g}'));ax.set_xlabel('Recorded API $ / attempt')
 else:
  ax.set_xlim((.9e5,3.3e6) if scope=='base' else (1.05e5,1.8e6));ax.set_xticks([1e5,3e5,1e6,3e6] if scope=='base' else [1.5e5,3e5,6e5,1.2e6]);ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,p:f'{x/1e6:g}M' if x>=1e6 else f'{x/1e3:g}k'));ax.set_xlabel('Recorded tokens / attempt')
 ax.xaxis.set_minor_formatter(ticker.NullFormatter());ax.tick_params(axis='both',labelsize=17)
def point(ax,r,metric,color,marker):
 x=r['plot_'+metric];y=100*r['pass_rate'];complete=r['cost_complete'] if metric=='usd' else r['token_complete']
 if not complete:
  hi=r.get('mean_cost_upper_usd') if metric=='usd' else None
  if hi is not None:
   ax.plot([x,hi],[y,y],color=color,lw=1.4,zorder=2);ax.plot([hi,hi],[y-1.4,y+1.4],color=color,lw=1.2,zorder=2)
  else:
   ax.annotate('',xy=(x*1.10,y),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=color,lw=1.05),zorder=2)
 ax.scatter(x,y,s=78,marker=marker,facecolor=color if complete else 'white',edgecolor='white' if complete else color,linewidth=.65 if complete else 1.6,zorder=4)
def draw(ax,scope,metric,compact=False):
 a=BASE if scope=='base' else HINT;axis(ax,metric,scope)
 plotted=[];fs=[]
 if scope=='base':
  f=front(a,metric);fs=f;ax.plot([r['plot_'+metric] for r in f],[100*r['pass_rate'] for r in f],ls='--',color='#444444',lw=1.6,zorder=1)
  for r in a:point(ax,r,metric,MC[r['model']],EM[r['effort']]);plotted.append(key(r))
  for r in f:
   label=NAMES[r['model']].replace('GPT-','')+' '+EN[r['effort']]
   if compact: label=label.replace('Extra High','X')
   dx,dy=7,9
   if metric=='usd' and r['model']=='gpt-6-sol' and r['effort']=='high':dx,dy=7,-22
   if metric=='usd' and r['model']=='gpt-6-sol' and r['effort']=='low':dx,dy=-5,10
   if metric=='tokens' and r['model']=='gpt-6-astra':dx,dy=7,9
   if compact and metric=='tokens' and r['model']=='gpt-5.6-luna':continue
   ax.annotate(label,(r['plot_'+metric],100*r['pass_rate']),xytext=(dx,dy),textcoords='offset points',fontsize=13 if compact else 16,color=MC[r['model']])
 else:
  for c in C:
   f=front([r for r in a if r['condition']==c],metric);fs+=f
   ax.plot([r['plot_'+metric] for r in f],[100*r['pass_rate'] for r in f],color=CC[c],ls='--',lw=1.6,zorder=1)
  for r in a:
   point(ax,r,metric,CC[r['condition']],MM[r['model']]);plotted.append(key(r))
   # Label frontier settings plus the previously omitted Luna High points.
   if r not in fs and not (r['model']=='gpt-5.6-luna' and r['effort']=='high'):continue
   if compact and r['effort'] in ['none','low'] and r['model']=='gpt-5.6-luna':continue
   dx,dy=5,6
   if r['condition']=='physics' and r['model']=='gpt-5.6-sol':dx,dy=-15,6
   if r['condition']=='physics' and r['model']=='gpt-5.6-luna' and r['effort']=='high':dx,dy=-2,-19
   if r['condition']=='baseline' and r['model']=='gpt-5.6-luna' and r['effort']=='high':dx,dy=-7,9
   if r['condition']=='physics' and r['effort']=='max':dx,dy=4,-19
   ax.annotate(ES[r['effort']],(r['plot_'+metric],100*r['pass_rate']),xytext=(dx,dy),textcoords='offset points',fontsize=13 if compact else 15,color=CC[r['condition']])
 assert len(plotted)==(24 if scope=='base' else 36) and len({tuple(k) for k in plotted})==len(plotted)
 audit.append({'scope':scope,'metric':metric,'compact':compact,'plotted_primary_settings':plotted,'provisional_frontier':[key(r) for r in fs],'pilot_shown':False})
def legends(fig,scope,pair):
 if pair:
  if scope=='base':
   fig.legend(handles=[Line2D([],[],marker='o',ls='',color=MC[m],label=NAMES[m],ms=7) for m in MODELS],loc='upper center',bbox_to_anchor=(.52,.26),ncol=4,frameon=False,fontsize=16,columnspacing=1.0,handletextpad=.35)
   fig.legend(handles=[Line2D([],[],marker=EM[e],ls='',color='#555555',label=EN[e],ms=7) for e in E],loc='upper center',bbox_to_anchor=(.52,.13),ncol=6,frameon=False,fontsize=15,columnspacing=1.2,handletextpad=.35)
  else:
   fig.legend(handles=[Line2D([],[],color=CC[c],marker='o',ls='--',label=CN[c],ms=7) for c in C],loc='upper center',bbox_to_anchor=(.52,.265),ncol=3,frameon=False,fontsize=17,columnspacing=2)
   fig.legend(handles=[Line2D([],[],marker=MM[m],color='#555555',ls='',label=NAMES[m],ms=7) for m in MM],loc='upper center',bbox_to_anchor=(.52,.19),ncol=4,frameon=False,fontsize=16,columnspacing=1.0,handletextpad=.3)
   fig.text(.17,.085,'Efforts: N None, L Low, M Medium, H High, X Extra High, Max',fontsize=15)
  return
 x=.835 if pair else .79
 if scope=='base':
  fig.legend(handles=[Line2D([],[],marker='o',ls='',color=MC[m],label=NAMES[m],ms=7) for m in MODELS],loc='upper left',bbox_to_anchor=(x,.92),frameon=False,fontsize=15,handletextpad=.4)
  fig.legend(handles=[Line2D([],[],marker=EM[e],ls='',color='#555555',label=EN[e],ms=7) for e in E],loc='upper left',bbox_to_anchor=(x,.45),frameon=False,fontsize=12.5,ncol=2,columnspacing=.25,handletextpad=.1)
 else:
  fig.legend(handles=[Line2D([],[],color=CC[c],marker='o',ls='--',label=CN[c],ms=7) for c in C],loc='upper left',bbox_to_anchor=(x,.92),frameon=False,fontsize=16,handletextpad=.4)
  fig.legend(handles=[Line2D([],[],marker=MM[m],color='#555555',ls='',label=NAMES[m].replace('GPT-5.6 ','').replace('DeepSeek V4.1 Flash','DeepSeek Flash'),ms=7) for m in MM],title='GPT-5.6 / DeepSeek V4.1',loc='upper left',bbox_to_anchor=(x,.67),frameon=False,fontsize=14.5,title_fontsize=14,handletextpad=.4)
  fig.text(x+.015,.19,'Effort labels\nN: None   L: Low\nM: Medium   H: High\nX: Extra High   Max: Max',fontsize=13.5,linespacing=1.45)
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(O/(name+'.'+ext),dpi=240)
 plt.close(fig)
for scope in ['base','hint']:
 fig,axs=plt.subplots(1,2,figsize=(13.2,5.8));fig.subplots_adjust(left=.075,right=.985,top=.91,bottom=.37,wspace=.22)
 for ax,metric,title in zip(axs,['usd','tokens'],['Dollar cost','Token cost']):
  draw(ax,scope,metric,True);ax.set_title(title,pad=10,fontweight='bold')
 axs[0].set_ylabel('Completed pass rate (%)');legends(fig,scope,True)
 fig.text(.075,.025,'Open symbols: recorded minima. Bars: upper cost bounds. Dashed lines: provisional frontiers.',fontsize=15,color='#444444')
 save(fig,'09-no-hint-all-costs' if scope=='base' else '10-hint-all-costs')
 for metric in ['usd','tokens']:
  fig,ax=plt.subplots(figsize=(13.2,5.7));fig.subplots_adjust(left=.08,right=.77,top=.95,bottom=.21);draw(ax,scope,metric);ax.set_ylabel('Completed pass rate (%)');legends(fig,scope,False)
  fig.text(.08,.045,'Open symbols show recorded minima; bars show upper cost bounds. Dashed frontiers are provisional.',fontsize=15,color='#444444')
  save(fig,('11-no-hint-' if scope=='base' else '12-hint-')+metric)
assert {tuple(key(r)) for r in BASE+HINT}=={tuple(key(r)) for r in R}
(U/'resource-plot-audit.json').write_text(json.dumps({'primary_settings_in_union':48,'study_attempts':2068,'pilot_attempts_separate':3,'pilot_displayed':False,'panels':audit,'smoothing':False,'uncertainty':'Open markers at known lower bounds. Dollar bars terminate at conservative reserves. Right arrows indicate unbounded missing usage, not a measured range.'},indent=2)+'\n')
print('Rendered all 24 no-hint and 36 matched hint settings in dollar and token views. Pilot excluded from figures; source observations preserved.')
