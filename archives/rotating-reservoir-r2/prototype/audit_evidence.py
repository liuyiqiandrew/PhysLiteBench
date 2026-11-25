"""Read-only evidence inventory for the rotating-reservoir hardening decision."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
job=ROOT/'jobs/rotating-reservoir-retained-three-r1-plain-20261004-035631'
canonical=ROOT/'tasks/rotating-reservoir';frozen=job/'frozen-task'
prior=json.loads((ROOT/'results/retained-three-brownian-trial-reviews.json').read_text())
approved=json.loads((ROOT/'results/retained-three-r1-native-input-review.json').read_text())['approved_blocks']

def function_tree(path,name):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
    return ast.dump(node,include_attributes=False)

inventory={}
def add(p):
    inventory[str(p.relative_to(ROOT))]=sha(p)
    return sha(p)
source_checks=[]
for p in sorted(canonical.rglob('*')):
    if not p.is_file() or '__pycache__' in str(p):continue
    rel=p.relative_to(canonical);f=frozen/rel
    source_checks.append({'file':str(rel),'current_sha256':add(p),'frozen_sha256':add(f),'identical':sha(p)==sha(f)})
trials=[]
for t in sorted(job.glob('rotating-reservoir__*')):
    result=json.loads((t/'result.json').read_text());traj=json.loads((t/'agent/trajectory.json').read_text())
    native=next((t/'agent/sessions').rglob('*.jsonl'));events=[json.loads(s) for s in native.read_text().splitlines()]
    user_blocks=[];developer_blocks=[];contexts=set();versions=set();calls=[]
    for event in events:
        p=event.get('payload',{});ty=event.get('type')
        if ty=='session_meta':versions.add(p.get('cli_version'))
        if ty=='turn_context':contexts.add((p.get('model'),p.get('effort')))
        if ty=='response_item' and p.get('type')=='message' and p.get('role') in ['user','developer','system']:
            for c in p.get('content',[]):
                if c.get('type') not in ['input_text','output_text']:continue
                s=c.get('text','');d=hashlib.sha256(s.encode()).hexdigest()
                block={'sha256':d,'length':len(s)}
                if p['role']=='user':block['approved']=d in approved;user_blocks.append(block)
                else:developer_blocks.append(block)
        if ty=='response_item' and p.get('type') in ['custom_tool_call','function_call']:
            s=p.get('input',p.get('arguments',''));calls.append({'name':p.get('name'),'sha256':hashlib.sha256(s.encode()).hexdigest()})
    evidence={}
    for rel in ['result.json','config.json','final.diff','agent/trajectory.json','agent/codex.txt','verifier/metrics.json','verifier/reward.txt','verifier/test-stdout.txt','artifacts/app/model.py','artifacts/app/README.md','artifacts/app/test_public.py','artifacts/app/data/calibration.json']:
        p=t/rel;evidence[str(p.relative_to(ROOT))]=add(p)
    evidence[str(native.relative_to(ROOT))]=add(native)
    output=prior[t.name]
    messages=[{'step':s['step_id'],'text':s.get('message','')} for s in traj['steps'] if s['source']=='agent' and s.get('message')]
    source=t/'artifacts/app/model.py'
    for rel in ['README.md','test_public.py','data/calibration.json']:
        assert sha(t/'artifacts/app'/rel)==sha(canonical/'environment'/rel)
    assert all(b['approved'] for b in user_blocks)
    assert contexts=={('gpt-5.6-luna','high')} and versions=={'0.154.0'}
    trials.append({'trial':t.name,'reward':result['verifier_result']['rewards']['reward'],'exception':result['exception_info'],
       'classification':output['classification'],'physics_reason':output['reason'],'metrics':json.loads((t/'verifier/metrics.json').read_text()),
       'source_sha256':sha(source),'same_covariance_function_as_starter':function_tree(source,'stationary_covariance')==function_tree(canonical/'environment/model.py','stationary_covariance'),
       'same_heat_function_as_starter':function_tree(source,'heat_rate')==function_tree(canonical/'environment/model.py','heat_rate'),
       'native_model_effort':[list(x) for x in sorted(contexts)],'native_cli_versions':sorted(versions),'native_user_blocks':user_blocks,'developer_blocks':developer_blocks,
       'emitted_action_count':len(calls),'emitted_action_hashes':calls,'public_messages':messages,'evidence_sha256':evidence,
       'causal_scope':'Final source, emitted public actions/messages and verifier results only; private reasoning is excluded.'})
refs=['results/retained-three-brownian-trial-reviews.json','results/retained-three-r1-native-input-review.json','results/candidates.json',
 'results/rotating-reservoir-memory-prototype.json','scripts/prototype_rotating_reservoir_memory.py',
 'results/rotating-reservoir-neutral-r1-fixed-parameter-repair.json',
 'archives/screened/tasks/dumbbell-stress/AUTHOR.md','archives/screened/tasks/dumbbell-stress/environment/model.py',
 'archives/thermal-bodies-r11/tasks/thermal-bodies/AUTHOR.md','archives/screened/thermal-bodies-r12/tasks/thermal-bodies/AUTHOR.md',
 'archives/fene-stress-r1/tasks/fene-stress/AUTHOR.md']
for rel in refs:add(ROOT/rel)
old=ROOT/'jobs/rotating-reservoir-neutral-r1-initial-plain-20261003-020125/rotating-reservoir__ht9aYFz'
for rel in ['artifacts/app/model.py','agent/trajectory.json','verifier/metrics.json','result.json']:add(old/rel)
results=[]
def visit(x):
    if isinstance(x,dict):
        if x.get('task') in ['rotating-reservoir','dumbbell-stress','thermal-bodies'] and 'job' in x and isinstance(x.get('trials'),list):
            if x['task'] in ['rotating-reservoir','dumbbell-stress'] or 'neutral-r4' in x['job']:
                results.append({'task':x['task'],'job':x['job'],'condition':x.get('condition'),'passes':x.get('passes'),'attempt_count':len(x['trials']),'trials':[{k:t.get(k) for k in ['trial','reward','exception']} for t in x['trials']]})
        else:
            for v in x.values():visit(v)
    elif isinstance(x,list):
        for v in x:visit(v)
visit(json.loads((ROOT/'results/candidates.json').read_text()))
out={'status':'read-only historical evidence review complete','model_runs':0,'canonical_edits':False,
 'fresh_batch':{'scored_trials':3,'passes':1,'clean_physical_failures':2,'exceptions':0,'no_outcome_discarded':True},
 'current_frozen_source_checks':source_checks,'trials':trials,'prior_family_and_nearest_outcomes':results,
 'prior_initial_trial':{'trial':'rotating-reservoir__ht9aYFz','reward':0,'physics':'Actual final source retains the laboratory-energy heat function. Public step7 explicitly accepts it; emitted edits add only fitting/API validation. This earlier0/1 remains separate from fresh1/3.'},
 'instruction_review':'All native user blocks exactly match the previously approved plugin inventory, Oct4 environment block and neutral full-replacement instruction. Developer skill/plugin blocks contain no apparatus-specific correction. Three artifact README/public test/data hashes match current canonical copies. Public actions use only /app sources/data and ordinary local calculations; no private-file access observed.',
 'passing_solver_capability':'fu84PgC step8 identifies the moving-bath internal heat and uses the full covariance of v−ΩRr, preserving exact mechanics. Simply changing drag anisotropy, rotation sign, trap axes or adding objective rigid-rotation memory does not invalidate that physical argument.',
 'evidence_sha256':inventory}
(HERE/'trial-review.json').write_text(json.dumps(out,indent=2)+'\n')
print('source files',len(source_checks),'matching',sum(c['identical'] for c in source_checks),'trials',len(trials),'evidence files',len(inventory))
