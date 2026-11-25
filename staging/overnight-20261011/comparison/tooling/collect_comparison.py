from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,re
R=Path('/Users/yp9846/Desktop/PhysLiteBench');S=R/'staging/overnight-20261011';C=S/'comparison'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def trial_record(task,model,p,prices,development=False):
    result=json.loads((p/'result.json').read_text());trajectory_path=p/'agent/trajectory.json';trajectory=json.loads(trajectory_path.read_text()) if trajectory_path.exists() else {};events=[];seen=set();native=[]
    for file in sorted((p/'agent/sessions').rglob('*.jsonl')):
        native.append(dict(path=str(file.relative_to(R)),sha256=sha(file)))
        for line in file.read_text().splitlines():
            try:d=json.loads(line)
            except json.JSONDecodeError:continue
            if d.get('type')!='token_usage_record':continue
            v=d['payload'];key=v.get('response_id') or (v.get('session_id'),d.get('ordinal'),d.get('timestamp'))
            if key in seen:continue
            seen.add(key);u=v['usage'];events.append(dict(response_id=v.get('response_id'),timestamp=d.get('timestamp'),usage=u))
    rates=prices['models'][model];totals=dict(input_tokens=0,cached_input_tokens=0,cache_write_input_tokens=0,output_tokens=0,reasoning_output_tokens=0);cost=0.;long_count=0
    for event in events:
        u=event['usage'];ins=u['input_tokens'];cached=u.get('cached_input_tokens',0);writes=u.get('cache_write_input_tokens',0);outs=u['output_tokens'];assert 0<=cached+writes<=ins
        long=ins>prices['long_context_threshold_input_tokens'];im=2 if long else 1;om=1.5 if long else 1
        usd=((ins-cached-writes)*rates['input']*im+cached*rates['cached_input']*im+writes*rates['cache_write_input']*im+outs*rates['output']*om)/1e6
        cost+=usd;long_count+=long;event['estimated_standard_api_usd']=usd;event['long_context']=long
        for name in totals:totals[name]+=u.get(name,0)
    steps=trajectory.get('steps',[]);models=sorted({s['model_name'] for s in steps if s.get('model_name')});network=[]
    for s in steps:
        for call in s.get('tool_calls',[]):
            if re.search(r'\b(curl|wget|urlopen|requests\.get|web__run|web_search)\b|git\s+clone|raw\.githubusercontent\.com',json.dumps(call.get('arguments',{}))):network.append(dict(step_id=s.get('step_id'),function=call.get('function_name')))
    reward=(result.get('verifier_result') or {}).get('rewards',{}).get('reward');model_path=p/'artifacts/app/model.py'
    row=dict(task=task,model=model,trial=p.name,path=str(p.relative_to(R)),reward=reward,exception=result.get('exception_info'),development_selected=development,model_names=models,token_usage=totals,request_count=len(events),long_context_requests=long_count,estimated_standard_api_usd=cost if events else None,usage_events=events,native_sources=native,source_sha256=sha(model_path) if model_path.exists() else None,trajectory_sha256=sha(trajectory_path) if trajectory_path.exists() else None,network_command_scan=network)
    timing=result.get('agent_execution') or {}
    if timing.get('started_at') and timing.get('finished_at'):
        parse=lambda x:datetime.fromisoformat(x.replace('Z','+00:00'));row['agent_execution_seconds']=(parse(timing['finished_at'])-parse(timing['started_at'])).total_seconds()
    return row
def main():
    plan=json.loads((C/'plan.json').read_text());prices=json.loads((S/'results/comparison-prices.json').read_text());rows=[]
    for runfile in sorted((C/'runs').glob('*/run.json')):
        r=json.loads(runfile.read_text())
        for p in sorted((R/'jobs'/r['job']).glob('*/result.json')):rows.append(trial_record(r['task'],r['model'],p.parent,prices))
    manifest=json.loads((C/'qualification-manifest.json').read_text())
    for task,qual in manifest['tasks'].items():
        if 'job' in qual:job=qual['job']
        else:
            ev=json.loads((R/qual['evidence']).read_text());job=next(x['job'] for x in ev['runs'] if x['condition']=='plain')
        for p in sorted((R/'jobs'/job).glob('*/result.json')):rows.append(trial_record(task,'gpt-5.6-luna',p.parent,prices,True))
    summary=[]
    for model in sorted({r['model'] for r in rows}):
        rs=[r for r in rows if r['model']==model];passed=sum(r['reward']==1 and r['exception'] is None for r in rs);checkpoint_passes=sum(r['reward']==1 for r in rs);n=len(rs);p=passed/n;z=1.95996398454;center=(p+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        costs=[r['estimated_standard_api_usd'] for r in rs if r['estimated_standard_api_usd'] is not None];names={r['task'] for r in rs};per_task={name:dict(attempts=sum(r['task']==name for r in rs),passes=sum(r['task']==name and r['reward']==1 and r['exception'] is None for r in rs),scored_checkpoint_passes=sum(r['task']==name and r['reward']==1 for r in rs),errors=sum(r['task']==name and r['exception'] is not None for r in rs)) for name in sorted(names)}
        summary.append(dict(model=model,attempts=n,scored=sum(r['reward'] is not None for r in rs),passes=passed,pass_rate=p,scored_checkpoint_passes=checkpoint_passes,scored_checkpoint_pass_rate=checkpoint_passes/n,illustrative_wilson_95=[max(0,center-half),min(1,center+half)],errors=sum(r['exception'] is not None for r in rs),tasks=len(names),complete_panel=len(names)==10 and all(v['attempts']==3 for v in per_task.values()),mean_standard_api_usd=sum(costs)/len(costs) if costs else None,total_standard_api_usd=sum(costs) if costs else None,missing_usage_trials=n-len(costs),development_selected=model=='gpt-5.6-luna',per_task=per_task))
    data=dict(updated_at=datetime.now(timezone.utc).isoformat(),comparison_plan_sha256=sha(C/'plan.json'),comparison_codex_cli=plan.get('codex_cli_version','0.154.0'),qualification_codex_cli='0.154.0',prior_access_failures=plan.get('access_failures',[]),pending_models=plan.get('pending',{}),pricing=prices,summary=summary,trials=rows,interpretation='Development target baseline was selected on these outcomes and used CLI0.154.0. Fresh panels use CLI0.160.0 consistently, after six preserved pre-model access failures with the older client. Access-only checks are excluded from scientific scores and per-attempt costs. Trial counts are small and clustered by task; Wilson bars are descriptive only. Preserve scored checkpoints with execution exceptions separately. Pareto means apply only to complete panels with fully observed usage; no claims for unrun models.')
    (C/'results.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
