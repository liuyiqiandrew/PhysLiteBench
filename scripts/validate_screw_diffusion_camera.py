"""Check scattering from a tagged chiral rigid body against finite screw advances."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/"tasks/screw-diffusion-camera"
sys.path[:0] = [str(TASK/"solution"),str(TASK/"environment")]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle=load("oracle",TASK/"solution/model.py")
ref=load("ref",TASK/"tests/reference.py")
starter=load("starter",TASK/"environment/model.py")
baseline=load("baseline",ROOT/"scripts/screw_diffusion_camera_baseline.py")
from angular import fringe


def experiments():
    rng=np.random.default_rng(305791)
    def axis(length):
        v=rng.normal(size=3)
        return np.round(length*v/np.linalg.norm(v),5).tolist()
    groups={}
    for family in ["origin","tagged","quadrature","reversal"]:
        qs=[]
        for i in range(7):
            k=np.round(rng.uniform(.7,2.35,2),4)
            if family=="reversal" or i%2:
                k[1] = -k[1]
            q=dict(mode="fringe",times=np.round(rng.uniform(.16,1.25,2),4).tolist(),
                   wave=k.tolist(),hand=(-1)**i,tag=axis(rng.uniform(.4,1.26)),
                   gates=[axis(rng.uniform(.7,.98)),axis(rng.uniform(.7,.98))],
                   phase=round(rng.uniform(-.9,.9),5))
            if family=="origin":
                q["tag"]=[0.,0.,0.]
            if family=="quadrature":
                q["phase"]=1.57079633
            qs.append(q)
        groups[family]=qs
    for family in ("current","phase_lock"):
        qs = []
        for i in range(8):
            q = dict(mode="fringe", times=np.round(rng.uniform(.18,1.,2),4).tolist(),
                     wave=np.round(rng.uniform(-2.,2.,2),4).tolist(), hand=(-1)**(i//2),
                     tag=axis(rng.uniform(.35,1.2)), gates=[axis(.93),axis(.88)],
                     phase=float([0.,np.pi/2,-.4,np.pi/2][i%4]))
            q["motor"] = {"bias":round(float(rng.uniform(-.78,.78)),3)}
            if family=="current":
                q["motor"]["flux"] = [round(float(rng.uniform(1.5,2.95))*(-1)**i,3),
                                      round(float(rng.uniform(1.1,2.9))*(-1)**(i//2),3)]
            else:
                q["pair"] = [i%3,(i+1+(i%2))%3]
            qs.append(q)
        groups[family]=qs
    return groups


def validate(generate):
    rho=ref.TRUE_PARAMETER
    cal=[]
    for j,r in enumerate([.55,.82,1.12]):
        for t in [.15,.41,.76,1.23]:
            value=np.exp(-rho*(2.49-r)*t)
            cal.append(dict(input=dict(mode="rotation",axis=j,t=t),value=round(value,8)))
    for file in [TASK/"tests/data/calibration.json",TASK/"environment/data/calibration.json"]:
        if generate:
            file.write_text(json.dumps(cal,separators=(",",":"))+"\n")
        assert json.loads(file.read_text())==cal
    groups=experiments()
    truth={}
    for family, qs in groups.items():
        truth[family]=[dict(input=q,value=round(ref.measurement(q),8)) for q in qs]
    path=TASK/"tests/data/hidden.json"
    if generate:
        path.write_text(json.dumps(truth,separators=(",",":"))+"\n")
    frozen=json.loads(path.read_text())
    assert frozen.keys()==truth.keys()
    for family,rows in truth.items():
        assert [r["input"] for r in rows]==[r["input"] for r in frozen[family]]
        np.testing.assert_allclose([r["value"] for r in rows],[r["value"] for r in frozen[family]],atol=1e-7,rtol=0)
    for module in (oracle,baseline,starter):
        model=module.Model().fit(cal)
        assert abs(model.mobility-rho)<1e-7
        errors={name:float(max(abs(model.predict(qs)-[r["value"] for r in truth[name]])))
                for name,qs in groups.items()}
        print(module.__name__,model.mobility,errors)
        if module is oracle:
            assert max(errors.values()) < .00005
        else:
            assert max(errors[k] for k in ("origin","tagged","quadrature","reversal")) < .00005
            assert min(errors[k] for k in ("current","phase_lock")) > .003
    for qs in groups.values():
        q=qs[0]
        fine=ref.measurement(q,order=11,step=.0015)
        assert abs(ref.measurement(q)-fine)<5e-5
        assert abs(fringe(q,rho)-fringe(q,rho,order=9))<5e-5
        mirror=dict(q,tag=(-np.asarray(q["tag"])).tolist(),hand=-q["hand"],phase=-q["phase"])
        assert abs(fringe(q,rho)-fringe(mirror,rho))<1e-10
        zero = 0. if "pair" in q else np.cos(q["phase"])
        assert abs(fringe(dict(q,wave=[0.,0.]),rho)-zero)<1e-11
    example=dict(mode="fringe",hand=1,times=[.45,.65],wave=[1.5,-1.6],
                 tag=[.65,-.4,.5],gates=[[.93,0,0],[0,.88,0]],phase=np.pi/2)
    assert round(ref.measurement(example),5)==round(fringe(example,rho),5)==-.05087
    lock = dict(mode="fringe", times=[.27,.56], wave=[.9,1.4], hand=1,
                tag=[.7,-.4,.5], gates=[[.34,-.81,.3],[.38,-.7,.53]],
                phase=np.pi/2, motor={"bias":.75}, pair=[0,1])
    assert abs(fringe(lock,rho)+.10982)<4e-6
    for q in (lock,groups["phase_lock"][0]):
        coarse,fine=[ref.finite_motor(q,eps,rho) for eps in (.025,.0125)]
        assert abs(2*fine-coarse-fringe(q,rho))<.0002


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    validate(parser.parse_args().generate)
