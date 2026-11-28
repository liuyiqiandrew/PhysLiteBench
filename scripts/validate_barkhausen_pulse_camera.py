"""Compare branching and fixed-duration bridge formulations; --write regenerates records."""
from pathlib import Path
import json,sys
import numpy as np
from importlib.util import spec_from_file_location,module_from_spec
task=Path(__file__).resolve().parents[1]/"tasks/barkhausen-pulse-camera"
for d in ("environment","solution","tests"):sys.path.insert(0,str(task/d))
from model import Model
from pulse import measure,single,transform
from reference import measure as other
from meter import Meter
spec=spec_from_file_location("independent",Path(__file__).with_name("barkhausen_pulse_camera_baseline.py"))
module=module_from_spec(spec);spec.loader.exec_module(module)
rng=np.random.default_rng(28117)
gain=.8821
cal=[]
for j in range(9):
    q=dict(kick=round(rng.uniform(.3,1.),3),spring=round(rng.uniform(.65,1.7),3),
           time=round(rng.uniform(.2,1.3),3))
    cal.append(dict(input=q,value=round(gain*Meter.calibration(q),9)))
groups={}
m=Model().fit(cal); approx=module.Model().fit(cal)
assert abs(m.diffusivity-gain)<1e-7
for name in ("first","last","resolved"):
    rs=[]
    for j in range(7):
        T=round(rng.uniform(.82,1.56),3)
        q=dict(time=T,size=round(rng.uniform(.68,1.78),3),trigger=name if name!="resolved" else ("first","last")[j%2],
               identity="blind" if name!="resolved" else (j//2)%2,
               walls=rng.uniform([.3,.65,.7],[.96,1.6,1.3],(2,3)).round(3).tolist(),
               flashes=[[round(T*f,3),*rng.uniform(.4,2.5,2).round(3)] for f in (.34,.61)])
        rs.append(dict(input=q,value=round(other(q,gain),8)))
    groups[name]=rs
    qs=[r["input"] for r in rs]
    targets=np.array([r["value"] for r in rs])
    error=max(abs(m.predict(qs)-targets))
    fake=max(abs(approx.predict(qs)-targets))
    print(name,error,fake, [float(min(targets)),float(max(targets))],flush=True)
    assert error<.00001 and fake>.005
    q=qs[-1]
    assert abs(other(q,gain,n=20,frequency_step=.35,cutoff=810)-targets[-1])<1e-6
    for q in qs:
        flip=dict(q,walls=q["walls"][::-1],flashes=[[t,b,a] for t,a,b in q["flashes"]],
                identity=q["identity"] if q["identity"]=="blind" else 1-q["identity"])
        assert abs(measure(q,gain)-measure(flip,gain))<1e-8
        # At zero source the extinction CDF, density and total weight are elementary.
        for v,k,h in q['walls']:
            D=gain*h;T=q['time']; exp=np.exp(k*T)
            p=np.exp(-v*k/(D*(exp-1)))
            vals=single([v,k,h],gain,0.,T,[T/3],[0.])
            np.testing.assert_allclose(vals,[p,p*v*k*k*exp/(D*(exp-1)**2),1.],atol=1e-12)
base=dict(time=1.25,size=1.13,walls=[[.48,.95,1.14],[.77,1.32,.84]],flashes=[[.42,1.2,.5],[.83,.46,1.8]],identity="blind")
for trig in ("first","last"):
    q=dict(base,trigger=trig)
    print("worked",trig,measure(q,.85),other(q,.85),flush=True)
for path,payload in [(task/"environment/data/calibration.json",cal),(task/"tests/data/calibration.json",cal),
                     (task/"tests/data/predictions.json",groups)]:
    if "--write" in sys.argv:path.write_text(json.dumps(payload,indent=2)+"\n")
    assert json.loads(path.read_text())==payload
