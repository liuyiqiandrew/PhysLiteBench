"""Validate fixed-reach bubble measurements; --write regenerates the frozen data."""
from pathlib import Path
import json
import sys
import numpy as np

task = Path(__file__).resolve().parents[1] / "tasks/polymer-wall-passage"
for sub in ("environment","solution","tests"):
    sys.path.insert(0,str(task/sub))
from reference import value, estimate
from passage import cumulative, response
from model import Model

aspect=.9163
rng=np.random.default_rng(28137)


def experiment(j,walls,gate):
    r=round(rng.uniform(2.,3.2),3)
    while True:
        pts=rng.uniform([-.55,.2],[.55,.56],size=(2,2)).round(3)*r
        if np.linalg.norm(pts[0]-pts[1])>.28*r:
            return dict(radius=r,dyes=pts.round(4).tolist(),walls=walls,gate=gate)


def row(q):
    return {"input":q,"value":round(float(value(q,aspect)),8)}


cal=[row(experiment(j,[],"off")) for j in range(9)]
groups={}
for n,name in enumerate(("bare","one","two")):
    qs=[]
    for j in range(8):
        a,b=rng.uniform(.62,1.13,size=2).round(2)
        walls=[] if n==0 else [(-1)**j*a] if n==1 else [(-1)**j*a,(-1)**(j+1)*b]
        qs.append(experiment(j,walls,"inside"))
    groups[name]=list(map(row,qs))
m=Model().fit(cal)
assert abs(m.aspect-aspect)<1e-6
for name,rs in groups.items():
    delta,shortcut,lower=[],[],[]
    for row_ in rs:
        q=row_["input"]
        z,w=[complex(x,aspect*y) for x,y in q["dyes"]]
        r,walls=q["radius"],q["walls"]
        true=response(r,z,w,walls)
        one,both=cumulative(r,z,w,walls)
        delta.append(abs(true-row_["value"]))
        shortcut.append(abs(both/one-true))
        lower.append(min(true,1-true))
        assert abs(true-response(1.5*r,1.5*z,1.5*w,[1.5*x for x in walls]))<.0001
        assert abs(true-response(r,-z.conjugate(),-w.conjugate(),[-x for x in walls]))<.0001
        assert abs(true-response(r,z,w,walls[::-1]))<.0001
    print(name,"oracle",max(delta),"radius cap",max(shortcut),"minimum probability",min(lower))
    assert max(delta)<.0002 and max(shortcut)>.005 and min(lower)>0
for walls in ([],[-.74],[-.74,1.04]):
    r,z,w=2.25,-.42+.84j,.94+.66j
    one,both=cumulative(r,z,w,walls)
    print("worked",walls,response(r,z,w,walls),both/one)
for path,data in [(task/"environment/data/calibration.json",cal),(task/"tests/data/calibration.json",cal),
                  (task/"tests/data/predictions.json",groups)]:
    if "--write" in sys.argv:
        path.write_text(json.dumps(data,indent=2)+"\n")
    assert json.loads(path.read_text())==data
