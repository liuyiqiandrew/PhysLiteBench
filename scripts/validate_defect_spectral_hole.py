"""Compare Fourier inversion to fixed-occupancy conditional Gaussian spectra."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'tasks/defect-spectral-hole'
sys.path[:0]=[str(T/'environment'),str(T/'solution'),str(T/'tests')]
from echo import reading
from reference import truth
V=.83642


def witness():
    return dict(phase=.27,mix=[0.,0.],gate=[.85,-.85],unsigned_b=False,kicks=[[.8,1.2,-.3]],
                domains=[dict(weight=1.,ratio=-1.,pockets=[[.6,.8,.3],[.9,1.,.4],[.75,.9,.7]])])


def split_check():
    # Thin the Poisson pocket into independent flipped and unflipped populations at t=.8.
    from math import factorial
    q=witness();x=.85;h=1.5;lam=.75;sigma=.9;phase=.27
    p=(1-np.exp(-2*.7*.8))/2
    total=0.;amp=0j
    for n in range(1,20):
        g=np.exp(-lam-x*x/(2*n*sigma*sigma))*lam**n/(factorial(n)*np.sqrt(2*np.pi*n*sigma*sigma))
        for k in range(n+1):
            pk=factorial(n)/(factorial(k)*factorial(n-k))*p**k*(1-p)**(n-k)
            mean=x*(1-2*k/n);variance=4*sigma*sigma*k*(1-k/n)
            total+=g*pk; amp+=g*pk*np.exp(-1j*h*mean-.5*h*h*variance)
    return total,float((np.exp(-1j*phase)*amp).real/total)


def cases():
    rng=np.random.default_rng(490814)
    def sample(a,b):return round(rng.uniform(a,b),4)
    def pocket():return [sample(.35,1.13),sample(.62,1.),sample(.22,1.2)]
    cal=[dict(input=dict(blank=True,phase=p),value=round(V*np.cos(p),8))
         for p in np.linspace(-2.4,2.5,12).round(4)]
    groups={}
    for group in ('sparse_resonances','overlap_shifts','unresolved_polarity'):
        rows=[]
        for i in range(7):
            r=sample(.63,1.33)*(-1)**i;r2=-r if i%2 else -r*sample(.68,.96)
            w=sample(.23,.77);u,v=sample(-.24,.25),sample(-.23,.28)
            dom=[dict(weight=w,ratio=r,pockets=[pocket() for _ in range(3)]),
                 dict(weight=1-w,ratio=r2,pockets=[pocket() for _ in range(3)])]
            if group=='sparse_resonances' and i%3==1:dom=dom[:1];dom[0]['weight']=1.
            kicks=[[sample(.09,.72),sample(1.3,2.8)*(-1)**i,sample(.85,2.2)*(-1)**(i//2)]]
            if i%3!=1:kicks.append([sample(.81,1.7),sample(.9,2.4)*(-1)**(i//2+1),sample(.8,2.)*(-1)**i])
            phase=sample(-2.6,2.6);x=sample(.47,1.48)*(-1)**i;y=sample(.33,1.1)*(-1)**(i//2)
            if group=='sparse_resonances':
                if i%3==0:y=r*x
                elif i%3==1:y=0.
                else:x=0.
                if i==6:x=y=0.
            if group=='overlap_shifts' and i%3==0:y=r*x+.045
            if group=='unresolved_polarity':
                if i in (0,1,3,5):u=v=0.;y=r*x
                elif i in (2,4):y=0.
            a=x+u*y;b=v*x+y
            q=dict(phase=phase,mix=[u,v],gate=[a,abs(b) if group=='unresolved_polarity' else b],
                   unsigned_b=group=='unresolved_polarity',kicks=kicks,domains=dom)
            value=truth(q);direct=reading(q)
            np.testing.assert_allclose(direct,value,rtol=0,atol=2e-7)
            rows.append(dict(input=q,value=round(V*value,9)))
        groups[group]=rows
    return cal,groups


def main(write):
    print('single shared pocket',split_check(),flush=True)
    for fun in (reading,truth):np.testing.assert_allclose(fun(witness()),split_check()[1],rtol=0,atol=2e-8)
    cal,groups=cases()
    for file,data in [(T/'environment/data/calibration.json',cal),(T/'tests/data/calibration.json',cal),(T/'tests/data/hidden.json',groups)]:
        if write:file.write_text(json.dumps(data,separators=(',',':'))+'\n')
        assert json.loads(file.read_text())==data
    for label,path in [('oracle',T/'solution/model.py'),('starter',T/'environment/model.py'),('reset',ROOT/'scripts/defect_spectral_hole_baseline.py')]:
        model=runpy.run_path(str(path))['Model']().fit(cal)
        errors=[max(abs(model.predict([r['input'] for r in rows])-[r['value'] for r in rows])) for rows in groups.values()]
        print(label,errors,flush=True)
        if label=='oracle':assert max(errors)<2e-7
        else:assert min(errors)>.003

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--generate',action='store_true');main(cli.parse_args().generate)
