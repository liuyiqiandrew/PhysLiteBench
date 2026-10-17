from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,subprocess,os,sys
R=Path('/Users/yp9846/Desktop/PhysLiteBench');B=R/'results/hint-comparison-20261014';S=R/'staging/hint-comparison-20261014';W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench/.parallel-hints-20261015')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
blocks=[json.loads(x.read_text()) for x in (B/'blocks').glob('*/block.json')]
assert not any(x['status']!='completed' for x in blocks),'Existing trial still running or needs review'
# Verify that the original serial scheduler has exited before installing its replacement.
ps=subprocess.run(['/bin/ps','-p','90103','-o','command='],capture_output=True,text=True)
assert 'tooling/controller.py' not in ps.stdout,'Old scheduler still alive'
old=json.loads((B/'plan.json').read_text());oldhash=sha(B/'plan.json')
assert old['max_concurrent_trials']==3
hist=B/'history/before-parallel-eight-cpus-20261015';hist.mkdir(exist_ok=False)
shutil.copy2(B/'plan.json',hist/'plan.json')
shutil.copytree(S/'tooling',hist/'tooling')
shutil.copy2(B/'results.json',hist/'results.json');shutil.copy2(B/'status.json',hist/'status.json')
for name in ['controller.py','run_block.py','run_science_openai.py','run_science_prime.py']:
 shutil.copy2(W/name,S/'tooling'/name)
(S/'resources').mkdir(exist_ok=True);shutil.copy2(W/'eight-cpus.yaml',S/'resources/eight-cpus.yaml')
# Add operational provenance to reporting; no reward, instruction, grading or cost changes.
p=S/'tooling/collect.py';s=p.read_text();oldline="row['completed_pass'] = row['reward'] == 1 and row['exception'] is None"
new="row['execution_cohort'] = run.get('execution_cohort', 'serial-three-trials')\n                row['max_concurrent_trials'] = run.get('max_concurrent_trials', 3)\n                row['cpu_set'] = run.get('cpu_set')\n                "+oldline
assert oldline in s;s=s.replace(oldline,new)
s=s.replace("plan.get('reuse_caveat', 'Historical task-selection results are not reused.'),","plan.get('reuse_caveat', 'Historical task-selection results are not reused.'),\n             plan.get('execution_resource_caveat', ''),")
p.write_text(s)
p=S/'tooling/plot.py';s=p.read_text();oldline='Descriptive historical-control comparison; client version, concurrency and date differ. Failed grades are not automatically physics errors.'
new='Local schedule: 3 concurrent trials initially, up to 6 sharing 8 CPUs after the user-requested amendment.\\nHistorical controls differ in client version, concurrency and date. Execution errors are not automatically physics failures.'
assert oldline in s;s=s.replace(oldline,new).replace('figsize=(12, 7.2)','figsize=(12, 7.6)').replace('bottom=.23','bottom=.27');p.write_text(s)
newplan=json.loads(json.dumps(old));newplan.update(max_concurrent_trials=6,max_concurrent_blocks=2,parallelism_amendment='parallel-eight-cpus-amendment-20261015.json',execution_resource_caveat='User-authorized scheduling amendment after block 131: earlier local observations used one block/up to 3 trials with unrestricted Docker CPU affinity. Later observations run up to 2 blocks/up to 6 trials, with all main containers and network sidecars sharing Docker CPUs 0-7 of 12. No model, effort, task, prompt, grading, agent timeout or verifier timeout changed. Resource contention can affect completion and cost, so retain execution-cohort labels.',resource_policy={'host_physical_cores':12,'docker_vcpus':12,'shared_cpu_set':'0-7','reserved_vcpu_capacity':4,'max_blocks':2,'max_trials':6,'max_prime_blocks':1,'per_trial_cpu_quota':None,'memory_limits_changed':False,'host_core_affinity_guaranteed':False})
newplan['reuse_caveat']='Luna baseline combines 38 historical Codex CLI 0.154.0/concurrency-4 trials with one local CLI 0.160.0 trial. Local hints used up to 3 concurrent trials before the user scheduling amendment and up to 6 sharing 8 Docker CPUs afterward. Task, instruction and grading hashes match, but client version, run date, concurrency and resource policy differ: this is a descriptive historical-control comparison, not an isolated causal hint effect.'
newplan['protocol'].append('User authorized parallelism on 2026-10-15: two concurrent blocks, up to six trials, sharing eight Docker virtual CPUs and leaving four-vCPU capacity. Keep only one Prime block active to preserve ledger attribution. Launch earliest eligible pending blocks, skipping a blocked Prime block only while another Prime block is active. Retain all started outcomes and annotate the execution cohort.')
for rel in list(newplan['tooling_sha256']):newplan['tooling_sha256'][rel]=sha(S/rel)
newplan['tooling_sha256']['resources/eight-cpus.yaml']=sha(S/'resources/eight-cpus.yaml')
assert newplan['tasks']==old['tasks'] and newplan['blocks']==old['blocks']
(B/'plan.json').write_text(json.dumps(newplan,indent=2)+'\n')
amendment={'created_utc':datetime.now(timezone.utc).isoformat(),'user_request':'Parallelize processes, leave four cores for user work, use remaining cores for Docker experiment.','old_plan_sha256':oldhash,'new_plan_sha256':sha(B/'plan.json'),'old_plan_path':str((hist/'plan.json').relative_to(R)),'resource_policy':newplan['resource_policy'],'execution_resource_caveat':newplan['execution_resource_caveat'],'last_serial_block':max(x['index'] for x in blocks),'next_pending_blocks':[x['index'] for x in newplan['blocks'] if not (B/'blocks'/f"{x['index']:03d}"/'block.json').exists()],'task_prompt_grading_and_schedule_slots_unchanged':True,'model_and_effort_unchanged':True,'agent_timeout_seconds':600,'verifier_timeout_seconds':60,'old_trials_drained_without_interruption':True,'cancelled_baselines_not_relaunched':38,'parallel_controller':'staging/hint-comparison-20261014/tooling/controller.py','prime_accounting':'At most one Prime block active. Its trial-tagged ledger slice remains isolated.','tests':'Python syntax, max-two-block admission, no concurrent Prime blocks, duplicate-slot refusal, compose CPU-set merge.','original_outcomes_snapshot':str((hist/'results.json').relative_to(R)),'changed_tooling_sha256':{rel:newplan['tooling_sha256'][rel] for rel in newplan['tooling_sha256'] if rel not in old['tooling_sha256'] or newplan['tooling_sha256'][rel]!=old['tooling_sha256'][rel]}}
(B/'parallel-eight-cpus-amendment-20261015.json').write_text(json.dumps(amendment,indent=2)+'\n')
# Keep presentation work pointed to the existing reviewed deck; only update future completion requirements.
cp=R/'results/presentation-20261014/update-on-completion.json';c=json.loads(cp.read_text());shutil.copy2(cp,hist/'presentation-completion-contract.json');c['local_study_resource_amendment']=str((B/'parallel-eight-cpus-amendment-20261015.json').relative_to(R));c['final_update_requirements'].append('Label the local execution resource cohorts: up to 3 trials before the parallelism amendment, then up to 6 trials sharing 8 Docker CPUs. Preserve this caveat in notes and results; do not describe the whole local study as concurrency 3. Frozen tasks, hints and grading remain unchanged.');cp.write_text(json.dumps(c,indent=2)+'\n')
# No credentials are printed or copied. Inherit the already-authorized private session location only as a path.
env=os.environ.copy();env.update(PATH=str(R/'.cache/bin')+':'+env['PATH'],UV_CACHE_DIR='/private/tmp/physlitebench-inspection-uv',UV_PYTHON_INSTALL_DIR=str(R/'.cache/python'))
subprocess.run([sys.executable,str(S/'tooling/collect.py')],cwd=R,env=env,check=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S');log=B/'monitoring'/f'controller-parallel-{stamp}.log'
with log.open('a') as stream:
 proc=subprocess.Popen([sys.executable,str(S/'tooling/controller.py')],cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
restart={'started_utc':datetime.now(timezone.utc).isoformat(),'controller_pid':proc.pid,'relay_pid':7905,'controller_log':str(log.relative_to(R)),'relay_log':'results/hint-comparison-20261014/monitoring/prime-relay-20261015-042015.log','scheduling':'two concurrent blocks / up to six trials / shared cpuset 0-7','review':'parallel-eight-cpus-amendment-20261015.json','relay_action':'Existing relay retained; only one Prime block active at a time.'}
(B/'monitoring'/f'restart-{stamp}.json').write_text(json.dumps(restart,indent=2)+'\n')
(W/'deployment.json').write_text(json.dumps(restart,indent=2)+'\n')
print(json.dumps(restart))
