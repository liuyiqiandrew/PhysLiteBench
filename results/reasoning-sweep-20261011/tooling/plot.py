from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
E=Path(__file__).resolve().parents[1];R=E.parent.parent;O=R/'output'/E.name
names={'gpt-6-luna':'GPT-6 Luna','gpt-6-sol':'GPT-6 Sol','gpt-5.6-terra':'GPT-5.6 Terra','gpt-6-astra':'GPT-6 Astra','deepseek/deepseek-v4.1-flash':'DeepSeek V4.1 Flash'}
colors={'gpt-6-luna':'#147D92','gpt-6-sol':'#3164BE','gpt-5.6-terra':'#BD651A','gpt-6-astra':'#7B46A4','deepseek/deepseek-v4.1-flash':'#278446'}
markers={'none':'P','low':'o','high':'s','xhigh':'D','max':'^'}
effort_names={'none':'None','low':'Low','high':'High','xhigh':'Extra High','max':'Max'}
def main():
    d=json.loads((E/'results.json').read_text());rows=[r for r in d['summary'] if r['complete_panel'] and (r['fully_observed_cost'] or r.get('prime_cost_reconciled')) and (r['model']!='deepseek/deepseek-v4.1-flash' or r.get('prime_cost_reconciled'))]
    rows.sort(key=lambda r:r['mean_estimated_api_usd']);O.mkdir(parents=True,exist_ok=True)
    status_file=E/'run-status.json';run_status=json.loads(status_file.read_text()) if status_file.exists() else {}
    lines=['# Reasoning-effort comparison','',f"Updated {d['updated_utc']}. Each complete point covers 13 frozen tasks and 39 attempts. All settings use the same Codex 0.160.0 client and 600/60-second agent/verifier limits.",'','| Model | Effort | Status | Recorded attempts | Completed passes | Passing checkpoints | Errors | Mean estimated API cost |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in d['summary']:
        cost=f"${r['mean_estimated_api_usd']:.5f}" if r['mean_estimated_api_usd'] is not None and r['complete_panel'] else ('—' if run_status.get('ended_utc') else 'pending')
        status='Complete' if r['complete_panel'] else ('Partial' if r['attempts'] else 'Not run')
        scores=[r[k] if r['attempts'] else '—' for k in ['passes','scored_checkpoint_passes','errors']]
        lines.append(f"| {names[r['model']]} | {effort_names[r['reasoning_effort']]} | {status} | {r['attempts']}/39 | {scores[0]} | {scores[1]} | {scores[2]} | {cost} |")
    if run_status:
        lines+=['',run_status.get('note',''),'','[Final run status and budget decision](run-status.json).']
    lines+=['','Only complete panels with reconciled costs enter the frontier. Missing attempts are not treated as failures. Execution errors remain recorded and are reported separately. The main score counts a pass only when the agent finishes without an exception; passing saved checkpoints after a timeout are also listed, without changing that predefined score. Low and Extra High are fresh runs; OpenAI High reuses 30 matching observations per model from the original ten-task comparison and adds nine observations on the new tasks.','','Costs are API-equivalent token estimates, not Pro subscription charges. Task selection against GPT-5.6 Luna and reuse of the original High observations are disclosed. Four historical Terra High trials were affected by host sleep; see the prior comparison timing audit. The selected GPT-5.6 Luna baseline remains in the old ten-task plot and is excluded from this thirteen-task frontier.','','[Frozen plan](plan.json), [results](results.json), [CSV](summary.csv), [pricing](pricing.json), [evidence audit](evidence-audit.json), [task-source audit](task-source-audit.json), [final audit](final-audit.json), and [plot reproduction](REPRODUCE.md). [Per-task scores](per-task-scores.csv), [trial outcomes](trial-outcomes.csv), and [model inventory](model-inventory.json) provide the detailed counts.']
    budget=E/'prime/budget-decision.json'
    if budget.exists() and not (E/'continuation-20261012.json').exists():
        b=json.loads(budget.read_text());lines+=['',f"Prime Low is partial ({b['low_pilot_attempts']}/39 attempts) and excluded from the frontier. Its pilot cost ${b['low_pilot_cost_usd']:.2f}; cumulative conservative Prime spending is ${b['conservative_cumulative_spend_usd']:.2f} against the $10 cap. Further Prime calls were stopped because projected completion cost exceeds remaining funds. See [budget decision](prime/budget-decision.json)."]
    lines+=['','Provider errors are execution failures. Token estimates may omit usage from interrupted responses; inspect the [saved reviews](reviews/) before treating small cost differences as meaningful.']
    if (E/'continuation-20261012.json').exists():
        lines+=['','The October 12 continuation adds Astra Extra High and a fresh DeepSeek cohort at None, Low, High and Max. All earlier observations are preserved. The user removed the Prime spending cap and waived the earlier account ceiling for the full Astra panel. See [authorization](continuation-20261012.json). DeepSeek Max uses Codex’s xhigh transport setting, translated to provider Max by the relay; both settings are verified in the request ledger. The provider’s none setting is labeled None. The fresh DeepSeek cohort uses [cache-aware current catalog rates](prime/pricing-v2.json) and retains conservative reserves for missing usage. A reproduced 15-second response-header limit in the network sidecar affected the legacy relay; see the [transport correction](prime/transport-v2-amendment.json). All 48 [legacy DeepSeek observations](prime/legacy-transport-results.json) remain separate from the new four-effort cohort.']
    if (E/'unstarted-setup-recovery-20261012.json').exists():
        lines+=['','Six setup attempts were interrupted by host sleep and network failures before model execution, grading or any saved native model session. They remain in the [unstarted-setup recovery record](unstarted-setup-recovery-20261012.json); their six replacements were declared before launch. Primary points count 39 started model attempts. No started or scored outcome was replaced.']
    if (E/'timing-flag-reviews.json').exists():
        lines+=['','Three started DeepSeek High attempts on superconducting heat overlapped closed-lid host sleep and ended in connection failures. Their original outcomes and abnormal elapsed times remain in the primary data; they were not replaced. See the [timing reviews](timing-flag-reviews.json). These execution failures do not establish physics mistakes.']
    if len(rows)>=2:
        frontier=[r for r in rows if not any(o['mean_estimated_api_usd']<=r['mean_estimated_api_usd'] and o['pass_rate']>=r['pass_rate'] and (o['mean_estimated_api_usd']<r['mean_estimated_api_usd'] or o['pass_rate']>r['pass_rate']) for o in rows)]
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
        fig=plt.figure(figsize=(12.8,7.5));ax=fig.add_axes([.075,.22,.56,.65]);ax.set_facecolor('#FAFBFD')
        ax.plot([x['mean_estimated_api_usd'] for x in frontier],[100*x['pass_rate'] for x in frontier],'--',color='#334155',lw=1.2,zorder=1)
        right=fig.add_axes([.67,.22,.30,.65]);right.axis('off');right.text(0,1.04,'MODEL / EFFORT',fontsize=10,fontweight='bold');right.text(1,1.04,'SCORE · COST',ha='right',fontsize=10,fontweight='bold')
        for i,r in enumerate(rows,1):
            x,y=r['mean_estimated_api_usd'],100*r['pass_rate'];color=colors[r['model']];lo,hi=r['descriptive_wilson_95']
            ax.errorbar(x,y,yerr=[[y-100*lo],[100*hi-y]],color=color,fmt='none',capsize=3,alpha=.42,lw=.9)
            if r.get('mean_cost_range_usd'):
                lower,upper=r['mean_cost_range_usd'];ax.errorbar(x,y,xerr=[[x-lower],[upper-x]],color=color,fmt='none',capsize=3,lw=.9)
            ax.scatter(x,y,marker=markers[r['reasoning_effort']],s=75,c=color,edgecolor='white',linewidth=.7,zorder=3)
            # Compact numbered point labels connect the graph to an explicit key.
            dy=10 if i%2 else -15;ax.annotate(str(i),(x,y),xytext=(5,dy),textcoords='offset points',fontsize=8,fontweight='bold',color=color,bbox=dict(fc='white',ec='none',alpha=.8,pad=.8))
            yy=1-(i-1)*min(.068,.95/max(len(rows),1));right.text(0,yy,f"{i}  {names[r['model']]} / {effort_names[r['reasoning_effort']]}",fontsize=9,color=color,va='top');right.text(1,yy-.027,(f"{r['passes']}/39 · ${x:.4f}" + (f" · {r['errors']} errors" if r['errors'] else '')),ha='right',fontsize=8.8,color='#475569',va='top')
        ax.set_xscale('log');xs=[r['mean_estimated_api_usd'] for r in rows];ax.set_xlim(min(xs)*.68,max(xs)*1.5);ax.set_ylim(-6,108);ax.set_yticks(range(0,101,20));ax.set_ylabel('Completed passing attempts (%)');ax.set_xlabel('Estimated API token cost per attempt (USD, log scale)');ax.grid(axis='y',color='#E2E8F0');ax.spines[['top','right']].set_visible(False)
        legend=[Line2D([0],[0],marker=markers[e],color='#475569',linestyle='',label={'none':'None','low':'Low','high':'High','xhigh':'Extra High','max':'Max'}[e]) for e in markers if any(r['reasoning_effort']==e for r in rows)]
        legend.append(Line2D([0],[0],color='#334155',linestyle='--',label='Observed frontier'));ax.legend(handles=legend,loc='upper left',bbox_to_anchor=(0,-.13),ncol=4,frameon=False,fontsize=8.5)
        fig.text(.075,.945,'Physics benchmark: reasoning effort and cost',fontsize=20,fontweight='bold',color='#0F172A');fig.text(.075,.904,f"13 frozen tasks · 3 attempts per task · {len(rows)} complete model–effort points",fontsize=11,color='#475569')
        if run_status.get('state')=='budget_limited':fig.text(.075,.880,'Stopped at usage guard: DeepSeek None is incomplete (3/39 attempts) and excluded.',fontsize=8.5,color='#92400E')
        fig.text(.075,.025,'Per-request token estimates; excludes local compute and subscription billing. Bars: descriptive 95% Wilson intervals.\nOpenAI High includes reused results; host-sleep effects are documented in the report. Task selection used GPT-5.6 Luna.\nIncomplete panels excluded. Interrupted responses may omit usage; Prime ranges retain missing-usage reserves.',fontsize=8.5,color='#475569')
        for ext in ['png','pdf','svg']:fig.savefig(O/f'pareto-reasoning.{ext}',dpi=210,bbox_inches='tight')
        plt.close(fig);(O/'pareto-frontier.json').write_text(json.dumps({'points':[{'model':r['model'],'effort':r['reasoning_effort']} for r in frontier],'task_count':13,'attempts_per_point':39},indent=2)+'\n')
        lines+=['','![Reasoning-effort frontier](../../output/reasoning-sweep-20261011/pareto-reasoning.png)','','[PDF](../../output/reasoning-sweep-20261011/pareto-reasoning.pdf) · [SVG](../../output/reasoning-sweep-20261011/pareto-reasoning.svg)']
    (E/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'complete_points':len(rows),'report':str(E/'README.md'),'figure_ready':len(rows)>=2}))
if __name__=='__main__':main()
