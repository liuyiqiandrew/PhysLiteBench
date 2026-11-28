"""Regenerate the conditional spectral measurements and check both independent calculations."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np
from scipy.integrate import quad
from scipy.special import betainc

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/aging-line-memory'
sys.path[:0]=[str(TASK/p) for p in ('environment','solution','tests')]
from reference import reading
from renewal import line

def preparations():
    rng=np.random.default_rng(47282)
    def draw(lo,hi,n=None):
        x=np.round(rng.uniform(lo,hi,n),5)
        return x.tolist() if n is not None else float(x)
    output={}
    for n,family in enumerate(('temperature_jump','return_cycle','successive_pulses'),1):
        rows=[]
        for i in range(8):
            ts=draw(.50,.70,n+1)
            if n==2 and i%2==0:ts[-1]=ts[0]
            if n==3 and i%3==0:ts[-1]=ts[1]
            q=dict(temperatures=ts,write=[draw(.55,1.3),draw(.35,.9)],
                   pulses=[[draw(.4,1.5),draw(.1,.4)] for _ in range(n)],
                   read=[0 if i%2==0 else i%n,round(.1+(i%6)*.11,2)],recovery=draw(.2,1.2))
            rows.append(q)
        output[family]=rows
    return output

def isothermal(q,tg):
    a=q['temperatures'][0]/tg
    A,D=q['write'];(B,G),=q['pulses'];E=D+B+G+q['recovery']
    def joint(f):
        L=D+f*B
        def integrand(v):
            density=np.sin(np.pi*a)/np.pi*A**a/(v**a*(A+v))
            return density*betainc(a,1-a,(L-v)/(E-v))
        return betainc(a,1-a,A/(A+E))+quad(integrand,D,L,epsabs=2e-12)[0]
    denom=joint(1)
    return joint(q['read'][1])/denom,denom

def main(generate):
    tg=.9863
    cal=[dict(input=dict(threshold=x),value=round(float(np.exp(-x/tg)),8))
         for x in (.22,.35,.52,.66,.80,1.05,1.24,1.46,1.68,1.92)]
    data={}
    for family,qs in preparations().items():
        data[family]=[]
        for q in qs:
            val,p=reading(q,tg)
            np.testing.assert_allclose(line(q,tg),(val,p),atol=3e-7,rtol=0)
            assert p>.05 and 0<val<1
            data[family].append(dict(input=q,value=round(val,9)))
    for name,obj in [('tests/data/calibration.json',cal),('environment/data/calibration.json',cal),
                     ('tests/data/predictions.json',data)]:
        f=TASK/name
        if generate:f.write_text(json.dumps(obj,indent=1)+'\n')
        saved=json.loads(f.read_text())
        if isinstance(obj,list):assert saved==obj
        else:
            assert saved.keys()==obj.keys()
            for k,rs in obj.items():
                assert [r['input'] for r in saved[k]]==[r['input'] for r in rs]
                np.testing.assert_allclose([r['value'] for r in saved[k]],[r['value'] for r in rs],atol=1.1e-9,rtol=0)
    for role,p in [('oracle',TASK/'solution/model.py'),('starter',TASK/'environment/model.py'),
                   ('shortcut',ROOT/'scripts/aging_line_memory_baseline.py')]:
        m=runpy.run_path(str(p))['Model']().fit(cal)
        assert abs(m.depth_scale-tg)<1e-8
        errors={}
        for k,rs in data.items():
            y=m.predict([r['input'] for r in rs])
            assert np.isfinite(y).all() and np.all((y>=0)&(y<=1))
            errors[k]=max(abs(y-[r['value'] for r in rs]))
        assert max(errors.values())<2e-7 if role=='oracle' else min(errors.values())>.004
        print(role,errors)
    for temp in (.50,.62,.70):
        for f in (0,.5,1.):
            q=dict(temperatures=[temp,temp],write=[1.,.3],pulses=[[.8,.2]],recovery=.3,read=[0,f])
            ex=isothermal(q,1.)
            for method in (reading,line):
                np.testing.assert_allclose(method(q,1.),ex,atol=2e-8,rtol=0)
            if temp==.5 and f==.5:print('worked example',ex)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    main(parser.parse_args().generate)
