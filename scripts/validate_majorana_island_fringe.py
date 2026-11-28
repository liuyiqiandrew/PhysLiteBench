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
from histories import measurement
from lead import lead_gram
from lead_reference import lead_trace
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
    for name in ('closed_ring','unchanged_charge','transferred_charge'):
        rows=[]
        for i in range(8):
            for _ in range(100):
                q=example('coherent_tally' if i%3==0 else 'clock_feedback',i)
                for enc in q['encounters']:
                    phase=val(-1.5,1.5)
                    enc['marker']=[round(.97*np.cos(phase),5),round(.97*np.sin(phase),5)]
                m=4+i%4; particles=1+i%(m-1)
                region=sorted(map(int,rng.choice(m,2+(i%2),replace=False)))
                sites=sorted(map(int,rng.choice(m,min(3,m),replace=False)))
                delta=0 if name=='unchanged_charge' else (-1)**i
                q['lead']=dict(onsite=[val(-1.2,1.2) for _ in range(m)],hop=val(.55,1.3),
                    flux=val(-np.pi,np.pi),beta=val(1.1,3.1),particles=particles,
                    quenches=[[int(rng.choice(sites)),val(-4.8,4.8),val(.45,1.65)] for _ in q['encounters']],
                    readout=None if name=='closed_ring' else dict(sites=region,delta=delta))
                rate,fringe=reference(q)
                if rate>.001:break
            assert rate>.001 and abs(fringe)<1+1e-12
            rates.append(rate)
            np.testing.assert_allclose(measurement(q,lead_gram),[rate,fringe],atol=3e-11,rtol=0)
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
            assert errors['closed_ring']<2e-8
            assert min(errors[n] for n in ('unchanged_charge','transferred_charge'))>.003
    q=public_example()
    q['lead']=dict(onsite=[0.,0.,0.,0.],hop=0.,flux=0.,beta=1.,particles=2,
                   quenches=[[0,np.pi,1.],[1,np.pi,1.]],readout=None)
    t=np.tanh(.75)
    contrast=(t+t*t)/(1+t**3)
    C=-1/3
    expected=((1+contrast*C)/8,(contrast+C)/(1+contrast*C))
    print('two-site witness',expected)
    np.testing.assert_allclose(measurement(q,lead_gram),expected,atol=1e-12)
    np.testing.assert_allclose(reference(q),expected,atol=1e-12)
    paths=[(0,1),(1,0)]
    np.testing.assert_allclose(lead_gram(q['lead'],paths),[[1,C],[C,1]],atol=1e-12)
    # The probabilities of all possible transfer outcomes sum to one on every history.
    q['lead'].update(hop=.87,flux=.53,onsite=[.2,-.8,.4,.1],beta=1.7)
    closed=lead_gram(q['lead'],paths)
    total=np.zeros_like(closed)
    for delta in range(-2,3):
        q['lead']['readout']=dict(sites=[0,1],delta=delta)
        exact=lead_gram(q['lead'],paths)
        np.testing.assert_allclose(exact,lead_trace(q['lead'],paths),atol=1e-12)
        assert np.linalg.eigvalsh(exact).min()>-1e-12
        total+=exact
    np.testing.assert_allclose(np.diag(total),1.,atol=1e-12)
    assert abs(total[0,1]-closed[0,1])>.01


if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--generate',action='store_true')
    main(args.parse_args().generate)
