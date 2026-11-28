"""Validate the frozen Majorana island predictions and physical controls."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/majorana-island-fringe"
sys.path[:0] = [str(TASK/"solution"), str(TASK/"environment"), str(TASK/"tests")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


oracle = load("oracle", TASK/"solution/model.py")
ref = load("ref", TASK/"tests/reference.py")
baseline = load("baseline", ROOT/"scripts/majorana_island_fringe_baseline.py")
starter = load("starter", TASK/"environment/model.py")


def runs():
    rng = np.random.default_rng(718305)
    def setting(n):
        return dict(n=n, beta=round(rng.uniform(1.,4.),4), ripple=round(rng.uniform(-.4,.4),4),
                    parity=int(rng.choice([-1,1])), frame=str(rng.choice(["site","wave"])),
                    meter_phase=round(rng.uniform(-2.,2.),4), outcome=int(rng.choice([-1,1])),
                    phase=round(rng.uniform(-2.,2.),4))
    def pulse(indices):
        a,b = rng.choice(indices,2,replace=False)
        return [int(a),int(b),round(rng.uniform(-1.8,1.8),4),round(rng.uniform(-1.6,1.6),4)]
    calibration = []
    for i in range(12):
        q = setting(3+i%5)
        arm = [pulse(range(2*q["n"])) for _ in range(3)]
        q.update(arms=[arm,arm], outcome=1, meter_phase=0.)
        calibration.append(dict(input=q,value=round(ref.TRUE_PARAMETER*np.cos(q["phase"]),8)))

    groups = {}
    for group in ("winding","meter","charge","extended"):
        rows = []
        for i in range(8):
            n = [3,4,5,6,8,11,17,21][i] if group!="extended" else [12,16,13,22,24,28,30,32][i]
            q = setting(n)
            indices = rng.choice(2*n,min(8,2*n),replace=False)
            arm = [pulse(indices) for _ in range(5)]
            if group == "winding":
                a,b = map(int,indices[:2])
                turn = [[a,b,round(np.pi*.75,6),0.]]*2+[[a,b,round(np.pi*.5,6),0.]]
                second = turn+arm
                second = second+[pulse(indices)]
            elif group == "meter":
                second = [p[:] for p in arm]
                for k in (0,3):
                    second[k][3] = -arm[k][3]
                q.update(meter_phase=.68+.13*i,outcome=-1 if i%2 else 1)
            elif group == "charge":
                q.update(frame="site",beta=5.5,parity=-1,phase=1.12+i*.025,
                         outcome=1,meter_phase=.35)
                # Rotate several occupied-level detectors before reading one parity.
                indices = list(map(int,rng.choice(n,min(4,n),replace=False)))
                arm = [[2*j,2*j+1,.0,.45] for j in indices]
                second = [[2*j,2*j+1,1.6,.78] for j in indices]
            else:
                q.update(frame="wave",outcome=1)
                arm = [pulse(indices) for _ in range(12)]
                second = [p[:] for p in arm]
                for k in (1,4,7,11):
                    second[k] = pulse(indices)
            q["arms"] = [arm, second]
            acceptance, signal = ref.measurement(q)
            assert acceptance>=.05 and abs(signal)<=1+1e-12
            rows.append(dict(input=q,value=round(ref.TRUE_PARAMETER*signal,9)))
        groups[group] = rows
    return calibration, groups


def cavity_experiments():
    rng=np.random.default_rng(20273431)
    groups={}
    for family in ("wavepacket","conditioned"):
        rows=[]
        for j,n in enumerate([3,4,5,10,21,32]):
            num=2 if j in (0,2,4) else 3
            used=list(map(int,rng.choice(2*n,size=min(6,2*n),replace=False)))
            def pair():return list(map(int,rng.choice(used,size=2,replace=False)))
            def pulse():return pair()+np.round(rng.uniform(-1.1,1.1,2),4).tolist()
            arms=[[pulse() for _ in range(num+1)] for a in range(2)]
            # Recombine nearby arms with a measurable spin fringe.
            arms[1]=[p.copy() for p in arms[0]]
            arms[1][0][2]+=.65
            arms[1][-1][3]*=-.6
            q=dict(n=n,beta=round(rng.uniform(.7,4.5),3),ripple=.25,parity=(-1)**j,
                   frame="wave" if j%3 else "site",phase=round(rng.uniform(-1.5,1.5),3),
                   meter_phase=.68,outcome=1,arms=arms)
            bins=[[k+1 if k<num-1 else num+1,*pair(),*np.round(rng.uniform(-.54,.54,2),4)]for k in range(num)]
            other=[r.copy() for r in bins]
            for k in range(num):
                other[k][-2]*=-1 if k%2==0 else 1
                if k==num-1:other[k][1:3]=pair()
            q['cavity']=dict(thermal=round(rng.uniform(.12,.6),3),angle=round(rng.uniform(-1.8,1.8),3),
                transmission=np.round(rng.uniform(.18,.77,num),3).tolist(),
                mode=np.round(rng.uniform(-1,1,(num,2)),3).tolist(),
                window=[-.3,.9] if family=="wavepacket" else [.25,1.25],segments=[bins,other])
            rows.append(q)
        groups[family]=rows
    return groups


def injected():
    rng=np.random.default_rng(4163957)
    groups={}
    for label,rows in cavity_experiments().items():
        for q in rows:
            c=q['cavity']
            bins=len(c['transmission'])
            def packet():
                a=rng.normal(size=(2*bins,2))
                a[bins:]*=.5
                return a.round(3).tolist()
            first=[packet(),packet()]
            if label=='wavepacket':
                first[1]=(.65*np.array(first[0])+.55*np.array(first[1])).round(3).tolist()
                second=(.45*np.array([packet(),packet()])+.7*np.array(first)).round(3).tolist()
                marker=np.exp(1j*rng.uniform(-2,2))*rng.uniform(.5,.95)
                pump=np.exp(1j*rng.uniform(np.pi*.62,np.pi*1.38))*rng.uniform(.5,1.15)
            else:
                second=(.55*np.array([packet(),packet()])+.7*np.array(first)).round(3).tolist()
                phi=rng.uniform(-2,2)
                marker=np.exp(1j*phi)*rng.uniform(.86,.98)
                pump=-np.exp(1j*(rng.uniform(-.35,.35)-phi))*rng.uniform(.82,1.16)
            def parts(x):return [round(x.real,3),round(x.imag,3)]
            c['pair_source']=dict(preparations=[first,second],pump=parts(pump),marker=parts(marker))
        groups['pair_exchange' if label=='wavepacket' else 'source_marker']=rows
    return groups


def dispersive_experiments():
    rng=np.random.default_rng(38617432)
    result={}
    for tag,rows in [('thermal_echo',cavity_experiments()['conditioned']),
                     ('incident_echo',injected()['pair_exchange'])]:
        for i,q in enumerate(rows):
            c=q['cavity'];bins=len(c['transmission'])
            shifts=rng.uniform(-np.pi,np.pi,(2,bins)).round(4)
            if i%2==0:shifts[1]=-shifts[0]
            c['dispersive']=shifts.tolist()
            c['thermal']=round(float(rng.uniform(.32,.64)),3)
            for arm in c['segments']:
                for segment in arm:segment[-2:]=[0.,0.]
        result[tag]=rows
    return result


def dot_experiments():
    rng=np.random.default_rng(5551731)
    def rounded(a,b):return round(float(rng.uniform(a,b)),4)
    groups={}
    for tag in ('coherent_tunnel','charging_echo'):
        rows=[]
        for i,n in enumerate((3,4,6,9,17,32)):
            count=3+i%3
            used=rng.choice(2*n,6,replace=False)
            def inds():return list(map(int,rng.choice(used,2,replace=False)))
            charge=tag=='charging_echo'
            regular=[[*inds(),rounded(-1.2,1.2),rounded(-.7,.7)] for j in range(count)]
            arms=[regular,[p.copy() for p in regular]]
            arms[1][1][2]+=.15
            seq=[[j+1,j%2,*inds(),rounded(.6,1.8),rounded(-2.5,2.5),rounded(-2.9,2.9) if charge else 0.]
                 for j in range(count)]
            other=[p.copy() for p in seq]
            if charge:
                for seg in other:seg[-1]*=-1
            else:
                for j in range(1,count):other[j][1]=1-other[j][1]
                other[-1][5]*=-1
            initial=[[1.,0.],[rounded(-.85,.85),rounded(.45,1.05)]]
            analyzer=[[rounded(.65,1.),rounded(-.35,.35)],[rounded(-.8,.9),rounded(.55,.9)]]
            q=dict(n=n,frame='site' if i in (0,3) else 'wave',beta=rounded(1.,3.8),ripple=rounded(-.4,.4),
                   parity=(-1)**i,phase=rounded(-1.5,1.5),meter_phase=rounded(-1.,1.),outcome=1,arms=arms,
                   dots=dict(prepare=initial,analyze=analyzer,segments=[seq,other]))
            rows.append(q)
        groups[tag]=rows
    return groups


def validate(generate):
    calibration, groups = runs()
    cavity_ref = load("cavity_ref", TASK/"tests/cavity_reference.py")
    for family, inputs in cavity_experiments().items():
        groups[family] = []
        for q in inputs:
            accept, value = cavity_ref.reference(q)
            assert accept > .025 and abs(value) < 1+1e-12
            groups[family].append(dict(input=q, value=round(ref.TRUE_PARAMETER*value,9)))
    pair_ref=load("pair_ref",TASK/"tests/pair_reference.py")
    for family,inputs in injected().items():
        groups[family]=[]
        for q in inputs:
            accept, signal = pair_ref.reference(q)
            assert accept>.025 and abs(signal)<1+1e-10
            groups[family].append(dict(input=q,value=round(ref.TRUE_PARAMETER*signal,9)))
    dsp_ref=load('dsp_ref',TASK/'tests/dispersive_reference.py')
    for family,inputs in dispersive_experiments().items():
        groups[family]=[]
        for q in inputs:
            accept,signal=dsp_ref.direct(q,19,26)
            assert accept>.025 and abs(signal)<1+1e-10
            groups[family].append(dict(input=q,value=round(ref.TRUE_PARAMETER*signal,9)))
    dot_ref=load('dot_ref',TASK/'tests/dot_reference.py')
    for family,inputs in dot_experiments().items():
        groups[family]=[]
        for q in inputs:
            accept,signal=dot_ref.reference(q)
            assert accept>.025 and abs(signal)<1+1e-10,(family,accept)
            groups[family].append(dict(input=q,value=round(ref.TRUE_PARAMETER*signal,9)))
    for path, data in [(TASK/"environment/data/calibration.json",calibration),
                       (TASK/"tests/data/calibration.json",calibration),
                       (TASK/"tests/data/hidden.json",groups)]:
        if generate:
            path.write_text(json.dumps(data, separators=(",",":")) + "\n")
        saved = json.loads(path.read_text())
        if isinstance(data, list):
            assert saved == data
        else:
            for group, rows in data.items():
                assert [r["input"] for r in rows] == [r["input"] for r in saved[group]]
                np.testing.assert_allclose([r["value"] for r in rows],
                                           [r["value"] for r in saved[group]],atol=2e-9,rtol=0)
    for group, rows in groups.items():
        for row in rows:
            q = row["input"]
            if q["n"] <= 6 and "cavity" not in q:
                np.testing.assert_allclose(ref.dense(q),ref.measurement(q),atol=1e-12,rtol=0)
    for model in (oracle, baseline, starter):
        fitted = model.Model().fit(calibration)
        assert abs(fitted.visibility-ref.TRUE_PARAMETER)<1e-8
        errors = {name:float(max(abs(fitted.predict([r["input"] for r in rows])
                                    - [r["value"] for r in rows]))) for name,rows in groups.items()}
        assert (max(errors.values())<1e-7) == (model is oracle)
        print(model.__name__,fitted.visibility,errors)
        if model is not oracle:
            assert max(errors[k] for k in ("winding","meter","charge","extended","wavepacket","conditioned","pair_exchange","source_marker","thermal_echo","incident_echo")) < 2e-8
            assert min(errors[k] for k in ("coherent_tunnel","charging_echo")) > .003
    example = dict(n=3,beta=1.5,ripple=0.,parity=1,frame="site",meter_phase=0.,outcome=1,
                   phase=np.pi/2,arms=[[],[]],
                   cavity=dict(thermal=0.,angle=0.,mode=[[1.,0.]],transmission=[.5],
                               window=[.3,1.2],segments=[[[0,0,1,0.,0.]],[[0,0,1,0.,.6]]]))
    for compare in (cavity_ref.reference(example)[1],oracle.Model().fringe(example)):
        assert abs(compare + .286820) < 5e-7
    h=[[1.,0.],[0.,0.]]
    v=[[0.,0.],[1.,0.]]
    diagonal=[[1.,0.],[1.,0.]]
    src=dict(preparations=[[h,h],[h,diagonal]],pump=[.7,.2],marker=[.5,.2])
    example['cavity']['pair_source']=src
    a=pair_ref.reference(example)
    b=oracle.Model().fringe(example)
    np.testing.assert_allclose(a[1],b,atol=1e-12)
    np.testing.assert_allclose(a,[.23777452477913,.06046239672673],atol=2e-12)
    print('two-photon voltage example',a)
    src['preparations']=[[v,v],[v,v]]
    np.testing.assert_allclose(oracle.Model().fringe(example),cavity_ref.reference(example)[1],atol=1e-12)
    src['preparations']=[[h,diagonal],[h,h]]
    src['marker']=[0.,0.]
    np.testing.assert_allclose(oracle.Model().fringe(example),baseline.Model().fringe(example),atol=1e-12)
    from scipy.special import ndtr
    del example['cavity']['pair_source']
    example['phase']=0.
    c=example['cavity']
    c.update(thermal=.5,dispersive=[[0.],[np.pi]])
    c['segments'][1][0][-1]=0.
    def interval(v):
        lo,hi=c['window']
        return ndtr(hi/np.sqrt(v))-ndtr(lo/np.sqrt(v))
    expected=.5*interval(.375)/interval(.75)
    np.testing.assert_allclose(oracle.Model().fringe(example),expected,atol=1e-11)
    print('zero mean, one parity rotation',expected)
    c['thermal']=0
    np.testing.assert_allclose(oracle.Model().fringe(example),1.,atol=1e-12)
    tunnel=dict(n=3,frame='site',beta=1.5,ripple=0.,parity=1,phase=0.,meter_phase=0.,outcome=1,arms=[[],[]])
    a=[0,1,0,1,np.pi/2,0.,0.]; b=[0,1,0,1,-np.pi/2,0.,0.]
    tunnel['dots']=dict(prepare=[[1,0],[1,0]],analyze=[[1,0],[1,0]],segments=[[a,b],[a[:-1]+[np.pi],b]])
    t=np.tanh(.75);p0=(1+(t+t*t)/(1+t**3))/2
    np.testing.assert_allclose(dot_ref.reference(tunnel),[(1+p0)/2,2*p0/(1+p0)],atol=1e-12)
    np.testing.assert_allclose(oracle.Model().fringe(tunnel),2*p0/(1+p0),atol=1e-12)
    q = dict(calibration[0]["input"],phase=0.,arms=[[],[[0,1,2*np.pi,0.]]])
    np.testing.assert_allclose(oracle.Model().fit(calibration).fringe(q),-1.,atol=1e-12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    validate(parser.parse_args().generate)
