from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import json, statistics, hashlib
R=Path('/Users/yp9846/Desktop/PhysLiteBench')
O=Path(__file__).resolve().parent
rows=json.loads((R/'results/hint-comparison-20261014/results.json').read_text())['trials']
dt=lambda s:datetime.fromisoformat(s.replace('Z','+00:00')) if s else None
def union_seconds(intervals):
    out=0.;right=None
    for a,b in sorted(intervals):
        if right is None or a>right:out+=(b-a).total_seconds();right=b
        elif b>right:out+=(b-right).total_seconds();right=b
    return out
trials=[];hashes={}
for r in rows:
    if 'deepseek' not in r['model']:continue
    p=R/f"results/hint-comparison-20261014/blocks/{r['block']:03}/{r['condition']}-prime-ledger.json"
    hashes[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
    requests=json.loads(p.read_text())['requests']
    assert len({q['id'] for q in requests})==len(requests)
    result=R/r['path']/'result.json';start=end=None
    if result.exists():
        resultdata=json.loads(result.read_text());ex=resultdata.get('agent_execution') or {}
        start=dt(ex.get('started_at'));end=dt(ex.get('finished_at'))
    deadline=start+timedelta(seconds=600) if start else None
    intervals=[];active=[];out=reason=0;large=0;headers=[]
    safe=[]
    for q in requests:
        assert q['trial_tag']==f"block-{r['block']:03}-{r['condition']}"
        a,b=dt(q.get('at')),dt(q.get('finished_at'));u=q.get('usage') or {}
        ot=u.get('output_tokens',u.get('completion_tokens',0)) or 0
        rt=(u.get('output_tokens_details') or u.get('completion_tokens_details') or {}).get('reasoning_tokens',0) or 0
        out+=ot;reason+=rt;large+=rt>=32000
        if a and b and start and end:
            aa=max(a,start);bb=min(b,end)
            if bb>aa:intervals.append((aa,bb))
        if a and deadline and a<=deadline and (b is None or b>deadline):active.append(q['id'])
        h=dt(q.get('upstream_headers_received_at'))
        if h and a:headers.append((h-a).total_seconds())
        safe.append({'id':q['id'],'status':q.get('status'),'at':q.get('at'),'finished_at':q.get('finished_at'),'output_tokens':ot,'reasoning_tokens':rt,'delivered':q.get('delivered'),'duration_seconds':(b-a).total_seconds() if a and b else None})
    trials.append({'task':r['task'],'condition':r['condition'],'block':r['block'],'trial':r['trial'],'exception':(r.get('exception') or {}).get('exception_type'),'completed_pass':r['completed_pass'],'reward':r['reward'],'agent_seconds':r.get('agent_execution_seconds'),'requests':len(requests),'output_tokens':out,'reasoning_tokens':reason,'large_reasoning_requests':large,'provider_active_at_deadline':bool(active),'provider_inflight_seconds_within_agent':union_seconds(intervals),'provider_headers_median_seconds':statistics.median(headers) if headers else None,'usage_complete_requests':sum(bool(q.get('usage')) for q in requests),'request_details':safe,'source':str(p.relative_to(R))})
summary=[]
for c in ['baseline','general','physics']:
    a=[r for r in trials if r['condition']==c];t=[r for r in a if r['exception']=='AgentTimeoutError']
    summary.append({'condition':c,'attempts':len(a),'exceptions':dict(Counter(r['exception'] or 'normal_completion' for r in a)),'passes':sum(r['completed_pass'] for r in a),'output_tokens_recorded':sum(r['output_tokens'] for r in a),'reasoning_tokens_recorded':sum(r['reasoning_tokens'] for r in a),'timeouts_with_provider_request_active_at_600s':sum(r['provider_active_at_deadline'] for r in t),'timeouts_with_at_least_32k_reasoning_request':sum(r['large_reasoning_requests']>0 for r in t),'timeout_provider_inflight_median_seconds':statistics.median(r['provider_inflight_seconds_within_agent'] for r in t),'median_provider_requests_per_trial':statistics.median(r['requests'] for r in a),'normal_completions':[{'task':r['task'],'trial':r['trial'],'pass':r['completed_pass'],'agent_seconds':r['agent_seconds']} for r in a if not r['exception']]})
report={'scope':'117 DeepSeek High attempts from the fixed three-condition study; 39 attempts per condition = 13 tasks x 3 repeats. Read-only resource/error review, no new runs or regrading.','method':'Provider request intervals clipped to native agent start/end and unioned per trial. Active-at-deadline uses the fixed 600-second boundary. Token counts are provider-reported metadata, including streams that settled after timeout; reasoning content is not read or reproduced. Inflight time includes server wait, generation and network delivery, not model compute alone. Missing usage is not imputed.','summary':summary,'trials':trials,'source_hashes':hashes}
(O/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
