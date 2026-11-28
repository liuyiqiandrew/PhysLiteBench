"""Regenerate the random-singlet cut spectrometer and its controls."""
import argparse,json,runpy,sys
from pathlib import Path
import numpy as np

base=Path(__file__).resolve().parents[1]
task=base/'tasks/random-singlet-cut-fringe'
sys.path[:0]=[str(task/s) for s in ('environment','solution','tests')]
from reference import reference
from renewal import response
from apparatus import poisson

visibility=.8217


def make():
    rng=np.random.default_rng(294378)
    def sample(lo,hi):return round(float(rng.uniform(lo,hi)),4)
    def query(start,last,count,phase):return dict(start=start,last=last,count=count,phase=phase,bands=[])
    cal=[]
    for _ in range(12):
        q=query('crossing',1,None,sample(-np.pi,np.pi))
        cal.append(dict(input=q,value=round(visibility*np.cos(q['phase']),8)))
    held={}
    for family in ('cut_origin','quiet_spectrum','heralded_spectrum'):
        rows=[]
        for i in range(8):
            for _ in range(80):
                last=[-1,1,None][i%3]
                count=None if family=='cut_origin' else 0 if family=='quiet_spectrum' else 1+i%2
                q=query('crossing' if i%2==0 else 'bulk',last,count,sample(-1.1,1.1))
                for _ in range(1+i%3 if family=='cut_origin' else 2+i%3):
                    def pulse():
                        return [str(rng.choice(['L','R'])),str(rng.choice(['x','y','z'])),
                                sample(-2.8,2.8),sample(-2.8,2.8)]
                    arms=[[pulse() for _ in range(1+int(rng.integers(3)))] for _ in range(2)]
                    q['bands'].append(dict(width=sample(.6,3.8),arms=arms,
                                           herald=[sample(.35,.99),sample(-2.9,2.9),sample(-2.9,2.9)]))
                acceptance,fringe=response(q)
                if acceptance>.025:break
            assert acceptance>.025
            benchmark=reference(q)
            np.testing.assert_allclose(benchmark,[acceptance,fringe],atol=6e-10,rtol=0)
            rows.append(dict(input=q,value=round(visibility*benchmark[1],9)))
        held[family]=rows
    return cal,held


def main(write):
    cal,groups=make()
    for dest,val in [(task/'environment/data/calibration.json',cal),(task/'tests/data/calibration.json',cal),
                     (task/'tests/data/hidden.json',groups)]:
        if write:dest.write_text(json.dumps(val,separators=(',',':'))+'\n')
        assert json.loads(dest.read_text())==val
    for name,script in [('oracle',task/'solution/model.py'),('starter',task/'environment/model.py'),
                        ('control',base/'scripts/random_singlet_cut_fringe_baseline.py')]:
        model=runpy.run_path(str(script))['Model']().fit(cal)
        assert abs(model.visibility-visibility)<2e-8
        err={g:float(np.max(abs(model.predict([x['input'] for x in rows])-[x['value'] for x in rows])))
             for g,rows in groups.items()}
        print(name,err)
        if name=='oracle': assert max(err.values())<2e-8
        elif name=='starter':assert min(err.values())>.003
        else:
            assert err['cut_origin']<2e-8
            assert min(err[n] for n in ('quiet_spectrum','heralded_spectrum'))>.003
    band=dict(width=1.,arms=[[],[['L','z',np.pi,0.]]],herald=[1.,np.pi,0.])
    q=dict(start='crossing',last=1,count=None,phase=0.,bands=[band])
    for start,expected in [('crossing',.7866455993033683),('bulk',.6957759534158587)]:
        q['start']=start
        for calc in (response,reference):np.testing.assert_allclose(calc(q),[1.,expected],atol=1e-10)
        q['count']=0
        for calc in (response,reference):np.testing.assert_allclose(calc(q),[expected,1.],atol=1e-10)
        q['count']=None

    # Rotating both members identically leaves the singlet unchanged.
    invariant=dict(width=2.,arms=[[],[['L','y',1.2,0.],['R','y',1.2,0.]]],herald=[.9,1.2,.1])
    q.update(start='bulk',last=None,count=1,bands=[invariant],phase=.43)
    for calc in (response,reference):
        full=calc(q)
        np.testing.assert_allclose(full[1],np.cos(.43),atol=1e-11)
        split=dict(q,bands=[dict(invariant,width=.7),dict(invariant,width=1.3)])
        np.testing.assert_allclose(calc(split),full,atol=1e-11)
    # The first crossing after left sign + is on the minus sublattice. Only
    # the second crossing suppresses this Ramsey signal; integrate its waiting law.
    from scipy.integrate import quad
    slow=(3-np.sqrt(5))/2
    fast=3-slow
    wait=lambda t:(np.exp(-slow*t)-np.exp(-fast*t))/np.sqrt(5)
    stay=lambda t:(fast*np.exp(-slow*t)-slow*np.exp(-fast*t))/np.sqrt(5)
    q.update(start='crossing',last=1,count=None,phase=0.,
             bands=[dict(width=2.,arms=[[],[['L','z',np.pi/2,np.pi/2]]],herald=[1,0,0])])
    second=stay(2.)+quad(lambda t:wait(t)*stay(2.-t),0.,2.)[0]
    for calc in (response,reference):
        np.testing.assert_allclose(calc(q),[1.,second],atol=1e-11)
        np.testing.assert_allclose(calc(dict(q,last=-1)),[1.,stay(2.)],atol=1e-11)


if __name__=='__main__':
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--generate',action='store_true')
    main(arg.parse_args().generate)
