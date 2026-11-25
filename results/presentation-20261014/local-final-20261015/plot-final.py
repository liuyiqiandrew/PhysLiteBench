from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench/.presentation-local-final-20261015')
d=json.loads((W/'data.json').read_text());assert d['status']['recorded_attempts']==351
for f in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Times New Roman','font.size':15,'axes.titlesize':21,'axes.labelsize':16,'svg.fonttype':'none','pdf.fonttype':42})
colors={'baseline':'#1D4E89','general':'#B45309','physics':'#0F766E'};labels={'baseline':'No hint','general':'General hint','physics':'Physics hint'};markers={'baseline':'o','general':'s','physics':'^'}
fig,(ax,bx)=plt.subplots(1,2,figsize=(15,7.5),gridspec_kw={'width_ratios':[1,1.25]})
fig.subplots_adjust(left=.065,right=.98,top=.79,bottom=.29,wspace=.3)
fig.suptitle('Physics hints and benchmark completion at High effort',x=.065,y=.97,ha='left',fontsize=27,fontweight='bold')
fig.text(.065,.90,'13 frozen tasks, three repeats per condition. 351 observations: 313 local and 38 historical controls.',fontsize=17)
for c in colors:
 a=sorted(d['frontiers'][c],key=lambda x:x['mean_estimated_api_usd'])
 ax.plot([s['mean_estimated_api_usd'] for s in a],[s['passes']/39*100 for s in a],marker=markers[c],color=colors[c],lw=2,ms=8,label=labels[c])
 for s in a:
  name='Luna' if 'luna' in s['model'] else 'Terra';offset=(7,8) if c!='physics' else (-42,8)
  ax.annotate(name,(s['mean_estimated_api_usd'],s['passes']/39*100),xytext=offset,textcoords='offset points',fontsize=13)
ax.set(xlim=(0,.32),ylim=(0,105),xlabel='Estimated API cost per attempt (USD)',ylabel='Completed pass rate (%)',title='Observed frontiers with complete costs')
ax.set_xticks([0,.08,.16,.24,.32],['$0.00','$0.08','$0.16','$0.24','$0.32']);ax.legend(loc='upper left',frameon=False,fontsize=13)
models=['gpt-5.6-luna','gpt-5.6-terra','deepseek/deepseek-v4.1-flash'];x=np.arange(3);width=.24
for i,c in enumerate(colors):
 a=[next(s for s in d['panels'] if s['model']==m and s['condition']==c) for m in models];xs=x+(i-1)*width
 bx.bar(xs,[s['passes']/39*100 for s in a],width,color=colors[c])
 for xx,s in zip(xs,a):
  bx.text(xx,s['passes']/39*100+2,f"{s['passes']}/39",ha='center',fontsize=13)
  bx.text(xx,-10,str(s['errors']),ha='center',fontsize=13,color='#555555')
bx.set(xlim=(-.55,2.55),ylim=(-16,105),ylabel='Completed pass rate (%)',title='All outcomes, including execution errors')
bx.set_xticks(x,['GPT-5.6\nLuna','GPT-5.6\nTerra','DeepSeek\nV4.1 Flash']);bx.set_yticks([0,20,40,60,80,100])
for a in [ax,bx]:
 a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.25);a.set_axisbelow(True)
fig.text(.065,.19,'Frontiers use only 4 of 9 settings with complete costs. Missing or uncertain costs exclude the other 5.',fontsize=15)
fig.text(.065,.145,'Above bars: completed passes out of 39. Below zero: execution errors. Errors do not establish physics failures.',fontsize=15)
fig.text(.065,.10,'Luna uses historical controls with different CLI version, date and concurrency. Local scheduling changed from 3 to 6 trials sharing 8 CPUs.',fontsize=14)
fig.text(.065,.055,'These are descriptive comparisons. Seven Prime observations retain uncertain charges; spending reserves are not confirmed costs.',fontsize=14)
for ext in ['png','pdf','svg']:fig.savefig(W/f'hint-comparison-final.{ext}',dpi=180)
plt.close(fig)
