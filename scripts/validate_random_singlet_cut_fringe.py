"""Regenerate the two-cut disorder ensemble and reproduce the competing-exchange witnesses."""
import argparse,json,runpy,sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad

base=Path(__file__).resolve().parents[1]
task=base/'tasks/random-singlet-cut-fringe'
sys.path[:0]=[str(task/s) for s in ('environment','solution','tests')]
from reference import reference
from renewal import response

visibility=.8217


def pick(arms=None,herald=None):
    return dict(arms=[[],[]] if arms is None else arms,herald=[0.,0.,0.] if herald is None else herald)


def make():
    rng=np.random.default_rng(431582)
    def draw(a,b):return round(float(rng.uniform(a,b)),4)
    def q(n,count):return dict(central=n,left_sign=[-1,1,None][int(rng.integers(3))],count=count,
                                phase=draw(-.98,.98),bands=[])
    cal=[]
    for _ in range(12):
        query=q(1,None)
        cal.append(dict(input=query,value=round(visibility*np.cos(query['phase']),8)))
    groups={}
    for name in ('one_center','two_centers','paired_herald'):
        rows=[]
        for i in range(8):
            for _ in range(80):
                count=None if i%3==0 else i%2
                if name=='paired_herald':count=2 if i%3 else 1
                query=q(1 if name=='one_center' else 2,count)
                for j in range(1+i%3):
                    width=draw(.75,1.6)
                    band=dict(width=width)
                    def pulse():return [str(rng.choice(['L','R'])),str(rng.choice(['x','y','z'])),draw(-3.,3.),draw(-2.8,2.8)]
                    for spot in ('left','right','both'):
                        arms=[[pulse() for _ in range(int(rng.integers(3)))] for _ in range(2)]
                        herald=[draw(.3,.97),draw(-2.9,2.9),draw(-2.8,2.8)]
                        if name=='paired_herald':
                            herald=[draw(.88,.99),draw(2.4,3.12),draw(-.25,.25)] if spot!='both' else [draw(0,.12),1.1,.12]
                            if spot!='both':arms=[[],[]]
                            else:arms=[[],[['L','x',draw(2.1,3.1),draw(-.2,.2)]]]
                        band[spot]=pick(arms,herald)
                    query['bands'].append(band)
                prob,fringe=response(query)
                if prob>.025:break
            assert prob>.025
            bench=reference(query)
            np.testing.assert_allclose(bench,[prob,fringe],rtol=0,atol=4e-9)
            rows.append(dict(input=query,value=round(visibility*bench[1],9)))
        groups[name]=rows
    return cal,groups


def main(write):
    cal,groups=make()
    for p,x in [(task/'environment/data/calibration.json',cal),(task/'tests/data/calibration.json',cal),
                (task/'tests/data/hidden.json',groups)]:
        if write:p.write_text(json.dumps(x,separators=(',',':'))+'\n')
        assert json.loads(p.read_text())==x
    for name,path in [('oracle',task/'solution/model.py'),('starter',task/'environment/model.py'),
                      ('control',base/'scripts/random_singlet_cut_fringe_baseline.py')]:
        m=runpy.run_path(str(path))['Model']().fit(cal)
        errs={g:float(np.max(abs(m.predict([x['input'] for x in rows])-[x['value'] for x in rows])))
              for g,rows in groups.items()}
        print(name,errs,flush=True)
        if name=='oracle':assert max(errs.values())<2e-8
        elif name=='starter':assert min(errs.values())>.003
        else:assert errs['one_center']<2e-8 and errs['paired_herald']>.003
    t=1.
    dark=pick();off=pick([[],[['L','z',np.pi,0.]]])
    b=dict(width=t,left=off,right=off,both=dark)
    query=dict(central=2,left_sign=None,count=None,phase=0.,bands=[b])
    witness=1-quad(lambda u:2*np.exp(-2*u+1-np.exp(u)),0,t)[0]
    print('no one-sided crossing, two/one center',witness,np.exp(-2))
    for fn in (reference,response):
        np.testing.assert_allclose(fn(query),[1.,witness],atol=2e-10,rtol=0)
        np.testing.assert_allclose(fn(dict(query,central=1)),[1.,np.exp(-2)],atol=2e-10,rtol=0)
        split=dict(query,bands=[dict(b,width=.3),dict(b,width=.7)])
        np.testing.assert_allclose(fn(split),fn(query),atol=2e-10,rtol=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generate',action='store_true')
    main(p.parse_args().generate)
