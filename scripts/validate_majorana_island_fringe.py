"""Regenerate and compare the coherent charge-counting fixtures."""
import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/majorana-island-fringe'
sys.path[:0]=[str(TASK/name) for name in ('environment','solution','tests')]
from counter import measurement
from reference import reference

parameter=.8364
oracle=runpy.run_path(str(TASK/'solution/model.py'))['Model']
starter=runpy.run_path(str(TASK/'environment/model.py'))['Model']
control=runpy.run_path(str(ROOT/'scripts/majorana_island_fringe_baseline.py'))['Model']


def public_example():
    h=[[1.,0.],[0.,0.]]
    plus=[[1.,0.],[1.,0.]]
    def encounter(hold):
        return dict(pulses=[[1,0,1,np.pi/2,0.,hold],[1,0,1,-np.pi/2,0.,0.]],
                    prepare=plus,tag=h,erase=plus,feedback=0.,dwell=0.,marker=[1.,0.])
    return dict(n=3,frame='site',beta=1.5,ripple=0.,parity=1,phase=-np.pi/2,
                probe=[[],[[0,1,np.pi]]],
                counter=dict(prepare=[[1,0],[0,0],[0,0]],analyze=[[0,0],[1,0],[0,0]]),
                encounters=[encounter(0.),encounter(np.pi)])


def inputs():
    rng=np.random.default_rng(2184939)
    def val(a,b):return round(float(rng.uniform(a,b)),4)
    def fixed(D,index):
        return [[int(i==index),0] for i in range(D)]
    def complex_vector(D):
        return np.round(rng.normal(size=(D,2)),3).tolist()
    def example(family,i):
        n=[3,4,5,8,13,18,24,32][i]; D=3+i%2
        used=rng.choice(2*n,6,replace=False)
        def pair():return list(map(int,rng.choice(used,2,replace=False)))
        prep=fixed(D,0);analyze=fixed(D,1+i%(D-1))
        if family=='clock_feedback':
            prep=complex_vector(D);analyze=complex_vector(D)
        qs=[]
        L=2+i%3
        for j in range(L):
            raw=complex_vector(2)
            erase=(np.array(raw)*.65+np.array(complex_vector(2))*.65).round(3).tolist()
            tag=complex_vector(2)
            m=0j if family=='resolved_counts' else rng.uniform(.74,.99)*np.exp(1j*rng.uniform(-2,2))
            pulses=[[k%2,*pair(),val(.4,1.7),val(-2.5,2.5),val(-2.5,2.5)] for k in range(2+j%2)]
            qs.append(dict(prepare=raw,tag=tag,erase=erase,pulses=pulses,
                           marker=[round(m.real,4),round(m.imag,4)],
                           feedback=val(-1.8,1.8) if family=='clock_feedback' else 0.,
                           dwell=val(-2.,2.) if family=='clock_feedback' else 0.))
        probes=[[[*pair(),val(-2.,2.)] for _ in range(3)],[[*pair(),val(-2.,2.)] for _ in range(3)]]
        return dict(n=n,beta=val(.6,4.7),ripple=val(-.4,.4),parity=(-1)**i,
                    frame='site' if i in (0,3) else 'wave',phase=val(-2.5,2.5),
                    counter=dict(prepare=prep,analyze=analyze),encounters=qs,probe=probes)
    calibration=[]
    for i in range(12):
        q=public_example()
        q['encounters']=[]
        q['counter']['analyze']=q['counter']['prepare']
        q['probe'][1]=q['probe'][0]
        q['phase']=val(-2.5,2.5)
        calibration.append(dict(input=q,value=round(parameter*np.cos(q['phase']),8)))
    groups={}; rates=[]
    for name in ('resolved_counts','coherent_tally','clock_feedback'):
        rows=[]
        for i in range(8):
            for _ in range(100):
                q=example(name,i)
                rate,fringe=reference(q)
                if rate>.001:break
            assert rate>.001 and abs(fringe)<1+1e-12
            rates.append(rate)
            np.testing.assert_allclose(measurement(q),[rate,fringe],atol=1e-11,rtol=0)
            rows.append(dict(input=q,value=round(parameter*fringe,9)))
        groups[name]=rows
    print('retained rates',min(rates),max(rates))
    return calibration,groups


def main(write):
    cal,groups=inputs()
    for dest,value in [(TASK/'environment/data/calibration.json',cal),(TASK/'tests/data/calibration.json',cal),
                       (TASK/'tests/data/hidden.json',groups)]:
        if write:dest.write_text(json.dumps(value,separators=(',',':'))+'\n')
        assert json.loads(dest.read_text())==value
    for name,model in [('oracle',oracle()),('starter',starter()),('control',control())]:
        model.fit(cal)
        assert abs(model.visibility-parameter)<2e-8
        errors={n:float(np.max(abs(model.predict([r['input'] for r in rows])-[r['value'] for r in rows])))
                for n,rows in groups.items()}
        print(name,model.visibility,errors)
        if name=='oracle':assert max(errors.values())<2e-8
        if name=='starter':assert min(errors.values())>.003
        if name=='control':
            assert errors['resolved_counts']<2e-8
            assert min(errors[n] for n in ('coherent_tally','clock_feedback'))>.003
    q=public_example()
    t=np.tanh(.75)
    p0=(1+(t+t*t)/(1+t**3))/2
    for fn in (measurement,reference):
        np.testing.assert_allclose(fn(q),[p0/4,1],atol=1e-12)
    np.testing.assert_allclose(measurement(q,record_tags=True),[1/8,2*p0-1],atol=1e-12)


if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--generate',action='store_true')
    main(args.parse_args().generate)
