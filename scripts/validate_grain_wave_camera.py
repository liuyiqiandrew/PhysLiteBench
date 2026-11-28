from pathlib import Path
import json,sys
import numpy as np
task=Path(__file__).resolve().parents[1]/"tasks/grain-wave-camera"
for d in ("environment","solution","tests"):sys.path.insert(0,str(task/d))
from reference import finite_joint,finite_mass,literal
from model import Model
from waves import signal,masses
base=[[0,1,1],[0,2,3],[0,3,2],[1,2,1],[2,3,2],[1,4,2],[3,4,1],[4,-1,2],[2,-1,1]]
rate=.1579
cal=[dict(input=dict(test_round=j/2),value=round(np.exp(-rate*j/2),9)) for j in range(1,11)]
m=Model().fit(cal)
rng=np.random.default_rng(28295)
groups={}
for kind in ("one_grain","competing","multiple"):
    rs=[];errors=[];delta=[]
    for i in range(7):
        H=1 if kind=="one_grain" else 2 if kind=="competing" else 3
        # A domain with one local two-grain return competes with domains
        # requiring two unit returns. Every multiple case needs two or more.
        noses=([[1,2,1],[2,1,1]] if H==1 else [[1,1,1],[2,1,1]]
               if H==2 else [[1,2,2],[2,1,1]])
        ds=[]
        for n,head in enumerate(noses):
            t=[]
            for k,(a,b,l) in enumerate(base):
                c=int(rng.integers(1,4)) if k>=3 else head[k]
                t.append([a,b,int(rng.integers(1,5)),c,int(rng.integers(1,4))])
            ds.append(dict(weight=(.37,.63)[n],tracks=t))
        while True:
            picks=[[int(rng.integers(0,9)),float(rng.choice([.25,.5,.75]))] for j in range(2)]
            if picks[0]==picks[1]:continue
            q=dict(sites=5,drive=0,deficit=H,pickup=picks,rounds=[0,9.125],
                   delay=[-1.125,3.125],domains=ds)
            if any(not np.any(masses(q,d,rate)[:,0]) for d in ds):continue
            v=signal(q,rate)
            if .045<v<.94:break
        f,g=(finite_joint(q,k,rate) for k in (1024,2048))
        delta.append(abs(2*g-f-v))
        errors.append(abs(signal(q,rate,True)-v))
        rs.append(dict(input=q,value=round(v,9)))
    print(kind,"finite-channel extrapolation",max(delta),"per-domain normalization",max(errors),flush=True)
    assert max(delta)<.00005 and max(errors)>.02
    groups[kind]=rs
A=[[0,1,1,1,1],[1,-1,1,1,1],[0,2,1,1,1],[2,-1,1,1,1],[0,3,1,1,1],[3,-1,1,1,1]]
B=[[0,1,1,2,1],*A[1:]]
q=dict(sites=4,drive=0,domains=[dict(weight=.4,tracks=A),dict(weight=.6,tracks=B)],deficit=2,
       pickup=[[2,.5],[4,.75]],rounds=[0,3],delay=[-1,1])
assert abs(signal(q,.15)-.625*np.exp(-.15/4))<1e-12
assert signal(dict(q,domains=[q["domains"][0]]),.15)==0
# Literal microscopic stationary heights/parallel grain transfer with nonuniform bond multiplicities.
for N in [2,4]:
    tr=[[0,1,1,1,2],[1,2,1,1,1],[2,0,1,2,1],[1,-1,1,1,1]]
    n=3;edges=[];picks=[];local=[]
    for a,b,l,c,d in tr:
        chain=[a,*range(n,n+N*l-1),b];n+=N*l-1
        picks.append(chain[N*l//2])
        for j,(s,t) in enumerate(zip(chain,chain[1:])):
            copies=c if j<N*l//2 else d
            edges.extend([(s,t)]*copies)
            if s==0:local.append((t,copies))
            if t==0:local.append((s,copies))
    tab,count=literal(n,tuple(edges),0)
    q=dict(sites=3,drive=0,domains=[dict(tracks=tr,weight=1.)],pickup=[[0,.5],[1,.5]],
           rounds=[0,8],delay=[-3.125,4.125],deficit=1)
    a=b=0
    for (depth,last),mult in tab.items():
        missing=sum(links for site,links in local if depth[site]<0)
        g,p=depth[picks[0]]/N,depth[picks[1]]/N
        if missing!=1 or g<0:continue
        a+=mult*np.exp(-rate*g)
        if p>=0 and -3.125<=p-g<=4.125:b+=mult*np.exp(-rate*max(p,g))
    assert a and abs(b/a-finite_joint(q,N,rate))<1e-10
    print("literal weighted grain additions",N,count,sum(tab.values()))
for p,obj in [(task/"environment/data/calibration.json",cal),(task/"tests/data/calibration.json",cal),
              (task/"tests/data/predictions.json",groups)]:
    if "--write" in sys.argv:p.write_text(json.dumps(obj,indent=2)+"\n")
    assert json.loads(p.read_text())==obj
