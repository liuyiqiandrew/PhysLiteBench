from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/Users/yp9846/Desktop/PhysLiteBench');S=R/'staging/overnight-20261011';C=S/'comparison'
def main():
    data=json.loads((C/'results.json').read_text());rows=data['summary'];complete=[r for r in rows if r['complete_panel'] and r['missing_usage_trials']==0]
    fresh=[r for r in complete if not r['development_selected']];assert len(fresh)>=2,'Need at least two complete fresh model panels'
    fresh.sort(key=lambda r:r['mean_standard_api_usd']);frontier=[];best=-1.
    for row in fresh:
        if row['pass_rate']>best:frontier.append(row);best=row['pass_rate']
    names={'gpt-6-luna':'GPT-6 Luna','gpt-6-sol':'GPT-6 Sol','gpt-6-astra':'GPT-6 Astra','gpt-5.6-luna':'GPT-5.6 Luna','gpt-5.6-terra':'GPT-5.6 Terra'}
    colors={'gpt-6-luna':'#087F8C','gpt-6-sol':'#3562C9','gpt-6-astra':'#7D3C98','gpt-5.6-luna':'#6B7280','gpt-5.6-terra':'#BF6C18'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':17,'axes.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(10.2,6.2));fig.patch.set_facecolor('white');ax.set_facecolor('#FAFBFD')
    fx=[r['mean_standard_api_usd'] for r in frontier];fy=[100*r['pass_rate'] for r in frontier]
    ax.plot(fx,fy,color='#1F2937',linestyle='--',linewidth=1.25,zorder=1,label='Observed Pareto frontier (fresh panels)')
    for i,row in enumerate(complete):
        x=row['mean_standard_api_usd'];y=100*row['pass_rate'];model=row['model'];color=colors.get(model,'#347D39');development=row['development_selected']
        if not development:
            lo,hi=row['illustrative_wilson_95'];ax.errorbar(x,y,yerr=[[y-100*lo],[100*hi-y]],fmt='none',color=color,alpha=.7,capsize=4,linewidth=1.1,zorder=2)
        ax.scatter([x],[y],s=88,marker='s' if development else 'o',facecolor='white' if development else color,edgecolor=color,linewidth=1.5,zorder=3)
        label=f"{names.get(model,model)}\n{row['passes']}/{row['attempts']} • ${x:.3f}/attempt"
        if development:label+='\nselected development baseline'
        elif row['errors']:label+=f"\n{row['errors']} runtime exception(s)"
        offset=(9,11 if y<75 else -37)
        ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=9,color=color,ha='left',va='bottom',bbox=dict(facecolor='#FAFBFD',edgecolor='none',pad=1.5,alpha=.97),zorder=4)
    xs=[r['mean_standard_api_usd'] for r in complete];ax.set_xscale('log');ax.set_xlim(min(xs)/1.55,max(xs)*3.5);ax.set_ylim(-5,110);ax.set_yticks(np.arange(0,101,20));ax.set_ylabel('Completed success rate (%)');ax.set_xlabel('Estimated Standard API cost per attempt (USD, log scale)')
    ax.set_title('Physics task benchmark',loc='left',fontweight='bold',pad=28)
    ax.text(0,1.035,'10 frozen tasks · High reasoning · 3 attempts per task and model',transform=ax.transAxes,fontsize=10.5,color='#4B5563')
    ax.grid(axis='y',color='#D8DEE8',linewidth=.7);ax.grid(axis='x',which='major',color='#E8EBF1',linewidth=.5);ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper left',frameon=False,fontsize=9)
    pending=[r['model'] for r in rows if not r['complete_panel'] and not r['development_selected']]
    status=('In progress: '+', '.join(names.get(m,m) for m in pending)+'. ') if pending else ''
    status+='Terra and DeepSeek are not included; model/access details remain unresolved.'
    fig.text(.09,.035,'Tokens priced per request, including cache rates; this is not the Pro subscription bill.\nBars: descriptive 95% Wilson intervals (small, task-clustered sample). Tasks were selected against GPT-5.6 Luna.\n'+status,fontsize=8.2,color='#4B5563',va='bottom')
    fig.subplots_adjust(left=.09,right=.98,top=.85,bottom=.2)
    output=R/'output/overnight-20261011';output.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(output/f'model-cost-score-pareto.{ext}',dpi=220,bbox_inches='tight')
    fields=['model','attempts','passes','pass_rate','scored_checkpoint_passes','errors','mean_standard_api_usd','total_standard_api_usd','development_selected','complete_panel']
    with (output/'model-cost-score.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    (output/'model-cost-score-results.json').write_text(json.dumps(data,indent=2)+'\n');(output/'pareto-frontier.json').write_text(json.dumps(dict(models=[r['model'] for r in frontier],criterion='Minimize mean estimated Standard API cost and maximize completed pass rate among complete fresh panels; target development baseline excluded.',missing= json.loads((C/'plan.json').read_text()).get('pending',{})),indent=2)+'\n')
    print(json.dumps(dict(figure=str(output/'model-cost-score-pareto.png'),frontier=[r['model'] for r in frontier],fresh_models=len(fresh))))
if __name__=='__main__':main()
