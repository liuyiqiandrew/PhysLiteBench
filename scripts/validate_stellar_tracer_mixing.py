"""Check signed stellar echoes by direct propagation of retarded fields."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/"tasks/stellar-tracer-mixing"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref=load("stream_reference",TASK/"tests/reference.py")
oracle=load("stream_oracle",TASK/"solution/model.py")
control=load("ballistic_control",ROOT/"scripts/stellar_tracer_mixing_baseline.py")


def preparations():
    rng=np.random.default_rng(48712)
    groups={g:[] for g in ("calibration","joint_density","transferred_echo","spectral_band")}
    for group,rows in groups.items():
        for i in range(16):
            s=i%2
            modes=([1,2],[2,3],[1,3],[2,4])[i%4]
            first=rng.uniform(-1,1,2)
            second=rng.uniform(-1,1,2)
            if group=="calibration":
                first[s],second[s]=1.,1.
            if group=="transferred_echo":
                first[:],second[:]=([1.,0.],[0.,1.]) if i%3 else ([0.,1.],[1.,0.])
            band=None
            if group=="spectral_band" or (group=="calibration" and i%3==0):
                band=[rng.uniform(-1.15,1.15),rng.uniform(.3,.95)]
            q=dict(k=modes,first=first.tolist(),second=second.tolist(),drift=rng.uniform(-.85,.85,2).tolist(),
                   strength=0. if group=="calibration" else rng.uniform(.34,.5),
                   delay=rng.uniform(-1.35,1.35),angle=rng.uniform(-2.8,2.8),
                   population="AB"[s],band=band)
            rows.append(q)
    groups["calibration"][0] = dict(k=[1,2],first=[1,0],second=[1,0],drift=[0,0],
                                    strength=0.,delay=.75,angle=0.,population="A",band=None)
    return groups


def encode(path,obj):
    path.write_text(json.dumps(obj,indent=2)+"\n")


def verify(generate):
    groups=preparations()
    folder=TASK/"tests/data"
    if generate:
        # Free streaming is analytic. The time-domain convolution checks that limit below.
        cal=[{"input":q,"value":round(oracle.ballistic(q,ref.TRUE_PARAMETER),8)}
             for q in groups["calibration"]]
        encode(folder/"calibration.json",cal)
        encode(TASK/"environment/data/calibration.json",cal)
        encode(folder/"predictions.json",{g:[{"input":q,"value":ref.signal(q,spacing=.0125)}
                                             for q in qs] for g,qs in groups.items() if g!="calibration"})
    assert (folder/"calibration.json").read_bytes()==(TASK/"environment/data/calibration.json").read_bytes()
    rows=json.loads((folder/"calibration.json").read_text())
    truth=json.loads((folder/"predictions.json").read_text())
    assert [r["input"] for r in rows]==groups["calibration"]
    models=[cls.Model().fit(rows) for cls in (oracle,control)]
    assert all(abs(m.dispersion-ref.TRUE_PARAMETER)<1e-6 for m in models)
    for group,measurements in [("calibration",rows),*truth.items()]:
        inputs=[r["input"] for r in measurements]
        assert inputs==groups[group]
        y=np.array([r["value"] for r in measurements])
        check=[ref.signal(q,spacing=.025) for q in inputs]
        delta=np.max(abs(np.array(check)-y))
        errors=[np.max(abs(m.predict(inputs)-y)) for m in models]
        print(group,"grid delta",delta,"oracle/shortcut errors",errors)
        assert delta<4e-5 and errors[0] < 2e-5
        assert errors[1]<2e-5 if group=="calibration" else errors[1]>.00045
    # Shifting both mean velocities and a receiver band translates the echo in phase.
    for q in groups["spectral_band"][:3]:
        u=.04
        shifted=dict(q,drift=[x+u for x in q["drift"]],band=[q["band"][0]+u,q["band"][1]],
                     angle=q["angle"]-(q["k"][1]-q["k"][0])*u*q["delay"])
        vals=models[0].predict([q,shifted])
        assert abs(vals[0]-vals[1])<1e-10


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    verify(parser.parse_args().generate)
