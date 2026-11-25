"""Reproduce the outcome and model/effort plots from the pinned report."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
R=Path(__file__).resolve().parent;O=R/'figures';O.mkdir(exist_ok=True)
D=json.loads((R/'failure-breakdown.json').read_text());rows=D['rows'];totals=D['totals']
for p in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','font.size':19,'axes.labelsize':21,'legend.fontsize':16,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#777777','axes.linewidth':.8,'savefig.facecolor':'white'})
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(O/f'{name}.{ext}',dpi=240)
 plt.close(fig)
# The red physics-failure segments start at zero to make their decline easy to read.
fig,ax=plt.subplots(figsize=(8.8,5.2));fig.subplots_adjust(left=.20,right=.97,top=.95,bottom=.39)
y=np.arange(3);left=np.zeros(3)
for key,label,color in [('physics_failure','Physics failure','#B00032'),('passed','Passed','#0F766E'),('other_failure','Other graded failure','#91A0AE'),('timeout','Timeout','#E4B05B'),('execution_error','Other execution error','#3C424A')]:
 vals=np.array([r[key] for r in totals]);ax.barh(y,vals,left=left,height=.58,color=color,label=label)
 for j,(v,l) in enumerate(zip(vals,left)):
  if key in ['passed','physics_failure']:ax.text(l+v/2,j,str(v),ha='center',va='center',color='white',fontweight='bold',fontsize=21)
  elif key=='other_failure' and v>=20:ax.text(l+v/2,j,str(v),ha='center',va='center',color='#18202A',fontsize=17)
 left+=vals
ax.set_yticks(y,['No hint','General hint','Physics hint']);ax.invert_yaxis();ax.set_xlim(0,390);ax.set_xticks([0,100,200,300,390]);ax.set_xlabel('Attempts per condition');ax.grid(axis='x',color='#DEE2E6',lw=.7);ax.set_axisbelow(True)
ax.legend(loc='upper center',bbox_to_anchor=(.43,-.24),ncol=2,frameon=False,handlelength=1.2,columnspacing=1.6,handletextpad=.5)
save(fig,'07-reviewed-outcomes')
# One matrix row per exact model/effort, all 39-attempt cells. No pooled High replication.
conds=['baseline','general','physics'];efforts=['none','low','medium','high','xhigh','max'];names={'none':'None','low':'Low','medium':'Medium','high':'High','xhigh':'Extra High','max':'Max'}
keys=[(m,e) for m in ['gpt-5.6-luna','gpt-5.6-terra','gpt-5.6-sol'] for e in efforts if any(r['model']==m and r['effort']==e for r in rows)]
a=np.array([[next(r['physics_failure'] for r in rows if r['model']==m and r['effort']==e and r['condition']==c) for c in conds] for m,e in keys])
fig,ax=plt.subplots(figsize=(13.2,5.25));fig.subplots_adjust(left=.23,right=.90,top=.86,bottom=.06)
cmap=LinearSegmentedColormap.from_list('physics_failure',['#FFFFFF','#F2C8D3','#B00032'])
im=ax.imshow(a,cmap=cmap,vmin=0,vmax=39,aspect='auto')
ax.set_xticks(range(3),['No hint','General hint','Physics hint']);ax.xaxis.tick_top();ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False,length=0,pad=9)
ax.set_yticks(range(10),[m.replace('gpt-5.6-','').title()+' '+names[e] for m,e in keys]);ax.tick_params(axis='y',pad=12)
for j in range(10):
 for k in range(3):ax.text(k,j,str(a[j,k]),ha='center',va='center',color='white' if a[j,k]>=29 else '#111111',fontsize=19,fontweight='bold')
for y in [5.5,7.5]:ax.axhline(y,color='#777777',lw=1.2)
for sp in ax.spines.values():sp.set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=.032,pad=.035);cb.set_ticks([0,10,20,30,39]);cb.set_label('Physics failures out of 39',labelpad=12)
save(fig,'08-physics-failure-matrix')
print('Saved reviewed outcome and model/effort figures')
