from pathlib import Path
import json,sys
import numpy as np
from importlib.util import spec_from_file_location,module_from_spec

task=Path(__file__).resolve().parents[1]/"tasks/barkhausen-pulse-camera"
for d in ("environment","solution","tests"):sys.path.insert(0,str(task/d))
from reference import readout
from pulse import measure
from model import Model
spec=spec_from_file_location("independent",Path(__file__).with_name("barkhausen_pulse_camera_baseline.py"))
mod=module_from_spec(spec);spec.loader.exec_module(mod)
rng=np.random.default_rng(28334)
D=.5391
cal=[]
for j in range(9):
    L=round(rng.uniform(2,4),2); x=round(L*rng.uniform(.12,.83),3)
    cal.append(dict(input=dict(length=L,x=x),value=round(2*D*x*(1-x/L),9)))
groups={}
for name in ("forward","reverse","mixed"):
    rs=[]
    for j in range(7):
        L=round(rng.uniform(2.5,3.8),3)
        a,b=np.round(L*np.array([rng.uniform(.19,.33),rng.uniform(.67,.81)]),3)
        u,v=rng.uniform(1.,1.52,2).round(3)
        k,h=rng.uniform(.18,.53,2).round(3)
        q=dict(length=L,walls=[[u,k],[v,h]],sizes=[b,round(L-a,3)],gates=[])
        relative=(.41,.62) if j%2==0 else (.60,.36)
        for n,r in enumerate(relative):
            x=round(a+r*(b-a),4)
            side=name if name!="mixed" else ["forward","reverse"][(n+j)%2]
            W=v-h*(L-x)-(k*x-u)
            lo,hi= ((.11,.42),(.54,.85))[int((j+n%2)%3==0)]
            q["gates"].append([x,side,round(lo*W,5),round(hi*W,5)])
        rs.append(dict(input=q,value=round(measure(q,D),9)))
    groups[name]=rs
m=Model().fit(cal);short=mod.Model().fit(cal)
for name,rs in groups.items():
    qs=[r["input"] for r in rs];y=np.array([r["value"] for r in rs])
    coarse=np.array([readout(q,D,80) for q in qs])
    fine=np.array([readout(q,D,160) for q in qs])
    err=max(abs(fine-y));deviation=max(abs(short.predict(qs)-y))
    print(name,"PDE 80 & 160",max(abs(coarse-y)),err,"independent gates",deviation,flush=True)
    assert err<.0007 and deviation>.02 and max(abs(m.predict(qs)-y))<2e-7
    for q in qs:
        # Reverse the physical sample and swap the two scans/opening gates.
        L=q["length"]
        other=dict(q,walls=q["walls"][::-1],sizes=q["sizes"][::-1],gates=[
            [L-x,{"forward":"reverse","reverse":"forward"}[side],lo,hi] for x,side,lo,hi in q["gates"]])
        assert abs(measure(other,D)-measure(q,D))<1e-8
base=dict(length=3.,walls=[[1.25,.25],[1.18,.48]],sizes=[2.24,2.2],
          gates=[[1.35,"forward",.25,.65],[1.74,"reverse",.30,.78]])
print("worked",measure(base,.55),readout(base,.55,240),flush=True)
alt=dict(base,gates=[base["gates"][1],base["gates"][0]])
print("reverse conditioning",measure(alt,.55),flush=True)
# Narrow legal strip: after enough diffusion the gate law is the stationary
# squared ground mode. Compare the laboratory solver against that limit.
edge=dict(length=3.,walls=[[1.,.5],[1.,.5]],sizes=[2.4,2.4],
          gates=[[1.05,"reverse",.04,.13],[1.95,"forward",.29,.38]])
assert abs(measure(edge,.70)-readout(edge,.70,180))<.00015
for p,obj in [(task/"environment/data/calibration.json",cal),(task/"tests/data/calibration.json",cal),
              (task/"tests/data/predictions.json",groups)]:
    if "--write" in sys.argv:p.write_text(json.dumps(obj,indent=2)+"\n")
    assert json.loads(p.read_text())==obj
