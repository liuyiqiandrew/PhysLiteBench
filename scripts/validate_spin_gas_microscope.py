from pathlib import Path
import numpy as np
import json
import sys
from importlib.util import spec_from_file_location, module_from_spec

task=Path(__file__).resolve().parents[1]/"tasks/spin-gas-microscope"
for sub in ("environment","solution","tests"):
    sys.path.insert(0,str(task/sub))
from model import Model
from reference import solve
from gas import Gas,traces


def load_baseline():
    path=Path(__file__).with_name("spin_gas_microscope_baseline.py")
    spec=spec_from_file_location("independent",path)
    m=module_from_spec(spec);spec.loader.exec_module(m)
    return m.Model()


vis=.8437
cal=[dict(input=dict(blank=True,phase=float(p)),value=round(vis*np.cos(p),9))
     for p in [-1.7,-1.1,-.7,-.3,.2,.65,1.,1.4,1.95]]
rng=np.random.default_rng(28863)
groups={}
for kind in ("disjoint","nested","crossed"):
    rs=[]
    for i in range(8):
        x=list(np.cumsum([int(rng.integers(1,4)),*map(int,rng.integers(1,4,size=3))]))
        a,b,c,d=map(int,x)
        moves=([[b,a],[d,c]] if kind=="disjoint" else
               [[d,a],[c,b]] if kind=="nested" else [[c,a],[d,b]])
        rotations=rng.uniform(-2.8,2.8,size=(3,3)).round(3).tolist()
        flip=[.09,.12] if i%3==0 else ([.83,.92] if i%3==1 else [.22,.65])
        q=dict(length=16,beta=round(rng.uniform(.9,2.),3),mu=round(rng.uniform(-.7,.7),3),
            flip=flip,moves=moves,rotations=rotations[:2],plate=rotations[2],
            phase=round(rng.uniform(-1.4,1.4),3))
        rs.append(dict(input=q,value=round(vis*float(solve(q)),9)))
    groups[kind]=rs
m=Model().fit(cal); short=load_baseline().fit(cal)
assert abs(m.visibility-vis)<1e-8
for name,rs in groups.items():
    qs=[r["input"] for r in rs];true=np.array([r["value"] for r in rs])
    correct=m.predict(qs);approx=short.predict(qs)
    print(name,"oracle",max(abs(correct-true)),"independent cycles",max(abs(approx-true)))
    assert max(abs(correct-true))<1e-7 and max(abs(approx-true))>.005
    for q in qs:
        assert min(w for _,_,w in traces(q))>=-1e-12
        rev=dict(q,moves=q["moves"][::-1],rotations=q["rotations"][::-1])
        assert abs(m.coherence(rev)-m.coherence(q))<1e-8
base=dict(length=10,beta=1.4,mu=.2,flip=[.15,.22],moves=[[5,1],[6,3]],
          rotations=[[.8,-.3,1.1],[-1.1,.9,-.7]],plate=[.65,.4,.3],phase=.3)
for moves in ([[5,1],[6,3]],[[6,1],[5,3]],[[3,1],[6,5]]):
    q=dict(base,moves=moves)
    print("worked",moves,"correct",solve(q),"fixed labels",np.real(np.exp(1j*q["phase"])*Gas().coherence(q)))
# All four addresses consecutive, identity pulses, a crossed or nested move.
# Either the ordered labels align or the two transferred labels must coincide.
test=dict(base,moves=[[3,1],[4,2]],rotations=[[0,0,0],[0,0,0]],plate=[0,0,0],phase=0)
assert abs(solve(test)-1)<1e-12
test["moves"]=[[4,1],[3,2]]
a,b=test["flip"]
assert abs(solve(test)-(b*(1-a)+a*(1-b))/(a+b))<1e-12
for p,data in [(task/"environment/data/calibration.json",cal),
               (task/"tests/data/calibration.json",cal),(task/"tests/data/predictions.json",groups)]:
    if "--write" in sys.argv: p.write_text(json.dumps(data,indent=2)+"\n")
    assert json.loads(p.read_text())==data
