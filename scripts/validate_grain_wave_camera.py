from pathlib import Path
import json,sys
import numpy as np
task=Path(__file__).resolve().parents[1]/"tasks/grain-wave-camera"
for d in ("environment","solution","tests"):sys.path.insert(0,str(task/d))
from reference import finite,literal
from model import Model
from waves import signal
p0=dict(sites=5,drive=0,tracks=[[0,1,1],[0,2,3],[0,3,2],[1,2,1],[2,3,2],[1,4,2],[3,4,1],[4,-1,2],[2,-1,1]])
p1=dict(sites=6,drive=1,tracks=[[1,0,3],[1,2,1],[3,1,2],[0,4,1],[4,2,2],[2,3,1],[3,5,2],[4,5,3],[5,-1,1],[0,-1,2]])
p2=dict(sites=5,drive=2,tracks=[[0,2,2],[2,1,1],[2,3,4],[0,4,2],[1,4,3],[3,4,1],[3,0,2],[4,-1,1],[1,-1,2]])
graphs=(p0,p1,p2)
rate=.1579
cal=[dict(input=dict(test_round=j/2),value=round(np.exp(-rate*j/2),9)) for j in range(1,11)]
m=Model().fit(cal)
rng=np.random.default_rng(2835)
groups={}
for kind in ("all","last","port"):
    rs=[];errors=[];delta=[]
    for i in range(8):
        graph=graphs[i%3]
        while True:
            tr=list(map(int,rng.choice(len(graph["tracks"]),2,replace=True)))
            sites=[[e,float(rng.choice([.25,.5,.75]))] for e in tr]
            if sites[0]==sites[1]:continue
            q=dict(graph,pickup=sites,rounds=[0,int(rng.integers(4,10))+.125],
               delay=[int(rng.integers(-3,0))-.125,int(rng.integers(1,5))+.125],**{"class":kind})
            if kind=="port":
                q['ports']=[i%3,(i+1)%3]
            try:v=signal(q,rate)
            except ZeroDivisionError:continue
            if .04<v<.93 and q not in [r["input"] for r in rs]:break
        f,g=(finite(q,rate,N) for N in (128,256))
        err=abs(2*g-f-v)
        delta.append(err)
        errors.append(abs(signal(q,rate,True)-v))
        rs.append(dict(input=q,value=round(v,9)))
    print(kind,"finite-channel extrapolation",max(delta),"junction trigger",max(errors),flush=True)
    assert max(delta)<.00004 and (kind=="all" or max(errors)>.004)
    groups[kind]=rs
toy=dict(p0,pickup=[[0,.5],[1,.5]],rounds=[0,8],delay=[-1.125,2.125])
for k in groups:
    q=dict(toy,ports=[0,2],**{"class":k})
    print("worked",k,signal(q,.15),signal(q,.15,True),flush=True)
# Positive microscopic check with parallel topplings, no tree model.
for N in [2,4]:
    tracks=[[0,1,1],[1,2,1],[2,0,1],[1,-1,1]];n=3;edges=[];gates=[];near=[]
    for a,b,l in tracks:
        path=[a,*range(n,n+N*l-1),b];n+=N*l-1
        gates.append(path[N*l//2]);edges.extend(zip(path,path[1:]))
        near.append(path[1] if a==0 else path[-2] if b==0 else None)
    tab,count=literal(n,tuple(edges),0)
    for kind in ("all","last","port"):
        q=dict(sites=3,drive=0,tracks=tracks,pickup=[[0,.5],[1,.5]],rounds=[0,7.125],
               delay=[-2.125,2.125],ports=[2],**{"class":kind})
        num=den=0.
        for (d,last),c in tab.items():
            if (kind=="last" and not last) or (kind=="port" and d[near[2]]>=0):continue
            g,p=d[gates[0]]/N,d[gates[1]]/N
            if d[gates[0]]<0:continue
            den+=c*np.exp(-rate*g)
            if d[gates[1]]>=0 and -2.125<=p-g<=2.125:num+=c*np.exp(-rate*max(g,p))
        assert den>0 and abs(num/den-finite(q,rate,N))<1e-11
    print("literal heights",N,n,count,sum(tab.values()),flush=True)
for p,obj in [(task/"environment/data/calibration.json",cal),(task/"tests/data/calibration.json",cal),
              (task/"tests/data/predictions.json",groups)]:
    if "--write" in sys.argv:p.write_text(json.dumps(obj,indent=2)+"\n")
    assert json.loads(p.read_text())==obj
