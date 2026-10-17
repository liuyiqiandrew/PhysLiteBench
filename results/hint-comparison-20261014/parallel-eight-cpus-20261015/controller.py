"""Run at most two frozen blocks concurrently, with one Prime block at a time."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl,json,os,subprocess,sys,time
ROOT=Path('/Users/yp9846/Desktop/PhysLiteBench')
STAGING=ROOT/'staging/hint-comparison-20261014'
RESULTS=ROOT/'results/hint-comparison-20261014'
def emit(**data):
 print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),**data}),flush=True)
def block_state(block):
 p=RESULTS/'blocks'/f"{block['index']:03d}"/'block.json'
 return json.loads(p.read_text()) if p.exists() else None
def select_next(blocks,started,active,limit):
 if len(active)>=limit:return None
 prime_busy=any(b['model'].startswith('deepseek/') for b in active)
 return next((b for b in blocks if b['index'] not in started and not(prime_busy and b['model'].startswith('deepseek/'))),None)
def main():
 with (STAGING/'controller.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit('Another parallel scheduler owns the lock')
  plan=json.loads((RESULTS/'plan.json').read_text())
  assert plan['max_concurrent_blocks']==2 and plan['max_concurrent_trials']==6
  blocks=plan['blocks'];started=set();active={};stop=False
  for b in blocks:
   state=block_state(b)
   if state:
    if state['status']!='completed':raise SystemExit(f"Existing block {b['index']} requires review; refusing duplicate execution")
    started.add(b['index'])
  env=os.environ.copy();env.update(UV_CACHE_DIR='/private/tmp/physlitebench-inspection-uv',UV_PYTHON_INSTALL_DIR=str(ROOT/'.cache/python'),PATH=str(ROOT/'.cache/bin')+':'+env['PATH'])
  def collect():
   subprocess.run([sys.executable,str(STAGING/'tooling/collect.py')],cwd=ROOT,env=env,check=True)
   subprocess.run([str(ROOT/'.cache/bin/uv'),'run','--offline','--python','3.13','--with','matplotlib==3.10.7','--with','numpy==2.5.4','python',str(STAGING/'tooling/plot.py')],cwd=ROOT,env=env,check=True)
  while True:
   completed=[]
   for idx,(b,proc,log) in list(active.items()):
    code=proc.poll()
    if code is None:continue
    log.close();state=block_state(b);del active[idx];completed.append(idx)
    if code or not state or state['status']!='completed':stop=True
    emit(finished_block=idx,returncode=code,state=state.get('status') if state else None)
   if completed:collect()
   if (RESULTS/'monitoring/STOP_PARALLEL_AFTER_DRAIN').exists():stop=True
   if stop:
    if not active:emit(stopped='Review or stop marker requires attention; all started outcomes retained');return
   else:
    while True:
     b=select_next(blocks,started,[entry[0] for entry in active.values()],2)
     if b is None:break
     assert block_state(b) is None,'New external block appeared; refusing duplicate'
     log=(RESULTS/'monitoring'/f"parallel-block-{b['index']:03d}.log").open('a')
     proc=subprocess.Popen([sys.executable,str(STAGING/'tooling/run_block.py'),str(b['index'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
     active[b['index']]=(b,proc,log);started.add(b['index'])
     emit(launch_block=b['index'],pid=proc.pid,model=b['model'],task=b['task'],active_blocks=list(active),cpu_set='0-7')
   if not active and len(started)==len(blocks):collect();emit(completed=True,expected_observations=351);return
   time.sleep(2)
if __name__=='__main__':main()
