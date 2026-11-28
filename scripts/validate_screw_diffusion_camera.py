"""Check scattering from a tagged chiral rigid body against finite screw advances."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/"tasks/screw-diffusion-camera"
sys.path[:0] = [str(TASK/"solution"),str(TASK/"environment"),str(TASK/"tests")]


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
from rotor import loop
from director import director
from director_reference import measurement as director_reference
loops=load("loops",TASK/"tests/loop_reference.py")


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
    frames = []
    for i in range(6):
        k=np.round(rng.uniform(.65,1.85,2),4)
        k[1]*=(-1)**i
        frames.append(dict(mode="fringe",times=np.round(rng.uniform(1.04,1.38,2),4).tolist(),
                           wave=k.tolist(),tag=axis(rng.uniform(.55,1.15)),
                           gates=[axis(.91),axis(.96)],hand=(-1)**(i//2),
                           phase=float([0.,.88,-.4,.35,1.52,-.68][i])))
    for sign in ("plus","minus","contrast"):
        groups["return_"+sign]=[dict(q,loop=sign) for q in frames]
    rng=np.random.default_rng(419041)
    directors=[]
    for i in range(12):
        axes=rng.choice(3,2,replace=False).tolist()
        q=dict(mode="fringe", hand=(-1)**i, times=np.round(rng.uniform(1,1.4,2),5).tolist(),
               wave=np.round(rng.uniform(.7,1.85,2)*rng.choice([-1,1],2),5).tolist(),
               tag=axis(rng.uniform(.5,1.12)), gates=[axis(.92),axis(.93)],
               phase=[0.,1.55,-.55,.7][i%4])
        directors.append((q,axes))
    for family,read in [("commuting","commute"),("crossed","crossed"),
                        ("exchange","exchange"),("lab_echo","exchange")]:
        qs=[]
        for i in (0,3,5,6,7,8):
            q,axes=directors[i]
            setting={"read":read}
            if read=="exchange":
                setting["axes"]=axes
                if family=="lab_echo" or i not in (0,8):
                    setting["first_sign"]=(-1)**i
            if family=="lab_echo":
                setting["pulse"]=[.30+.08*(i%3),2.45+.14*(i%4)]
            qs.append(dict(q,director=setting))
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
        truth[family]=[dict(input=q,value=round(director_reference(q,rho) if "director" in q else
                                                 loops.loop_reference(q,rho) if "loop" in q else ref.measurement(q),8)) for q in qs]
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
            added=("commuting","crossed","exchange","lab_echo")
            assert max(errors[k] for k in groups if k not in added) < .00005
            assert min(errors[k] for k in added) > .003
    for qs in groups.values():
        q=qs[0]
        if "director" in q:
            fine=director_reference(q,rho,order=8,step=.001)
            assert abs(fine-director(q,rho))<.0003
            continue
        if "loop" in q:
            fine=loops.loop_reference(q,rho,order=9,step=.001)
            assert abs(fine-loop(q,rho)) < .00005
            continue
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

    blank=dict(mode="fringe",hand=1,times=[1.1,1.2],wave=[0.,0.],gates=[[0.,0.,0.]]*2,
               tag=[0.,0.,0.],phase=0.,loop="contrast")
    def heat_sum(half):
        result=0.
        for j in np.arange(0,10)+half:
            m=np.arange(-j,j+1.)
            raise_m=np.diag(np.sqrt((j-m[:-1])*(j+m[:-1]+1.)),-1)
            # Asymmetric top in its principal axes.
            jx=(raise_m+raise_m.T)/2
            jy=-.5j*(raise_m-raise_m.T)
            levels=np.linalg.eigvalsh(.55*(jx@jx)+.82*(jy@jy)+1.12*np.diag(m*m))
            result+=(2*j+1)*np.exp(-rho*sum(blank["times"])*levels).sum()
        return result
    control=heat_sum(.5)/heat_sum(0.)
    assert abs(loop(blank,rho)-control)<1e-10 and abs(control-.85249)<5e-6
    check=dict(mode="fringe",hand=-1,times=[1.07,1.18],wave=[1.24,-1.35],tag=[.61,.44,-.52],
               gates=[[.68,.27,-.39],[.53,-.64,.33]],phase=.41)
    for label,target in [("plus",.09298),("minus",.01280)]:
        q=dict(check,loop=label)
        assert abs(loop(q,rho)-target)<5e-6 and abs(loops.loop_reference(q,rho)-target)<5e-6
    closure=dict(check,loop="minus",wave=[1.24,1.24])
    assert abs(loop(closure,rho)-loop(dict(closure,tag=[0.,0.,0.]),rho))<2e-6

    edge=dict(mode="fringe",loop="minus",hand=1,times=[1.4,1.4],wave=[2.4,2.4],
              tag=[-.2693,1.16962,-.49944],gates=[[.6063,-.78738,.11144],[-.20612,-.90504,-.37202]],
              phase=-2.094)
    exact=loops.loop_reference(dict(edge,tag=[0.,0.,0.]),.7)
    assert abs(loop(edge,.7)-exact)<.0002

    biax=dict(mode="fringe",hand=1,times=[1.12,1.25],wave=[1.45,-1.33],tag=[.72,.30,-.53],
              gates=[[.71,-.54,.30],[.15,.77,.53]],phase=.54)
    settings=[({"read":"commute"},.10651),({"read":"crossed"},-.01527),
              ({"read":"exchange","axes":[0,2],"first_sign":-1},.00765),
              ({"read":"exchange","axes":[0,2],"first_sign":-1,"pulse":[.4,2.7]},-.00335)]
    for setting,target in settings:
        q=dict(biax,director=setting)
        assert abs(director(q,rho)-target)<5e-6 and abs(director_reference(q,rho)-target)<5e-6
        blank=dict(q,wave=[0.,0.])
        wanted=0. if setting["read"]=="exchange" else np.cos(q["phase"])
        assert abs(director(blank,rho)-wanted)<2e-8
    flipped=dict(q,director=dict(q["director"],axes=[2,0]))
    assert abs(director(flipped,rho)+director(q,rho))<1e-10
    pos=dict(q,director=dict(q["director"],first_sign=1))
    raw={k:v for k,v in q["director"].items() if k!="first_sign"}
    assert abs((director(q,rho)+director(pos,rho))/2-director(dict(q,director=raw),rho))<1e-10


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    validate(parser.parse_args().generate)
