"""Validate spatially consecutive carrier exchanges on nested and disjoint singlet arcs."""
import argparse,json,runpy,sys
from pathlib import Path
import numpy as np
base=Path(__file__).resolve().parents[1];t=base/'tasks/random-singlet-cut-fringe'
sys.path[:0]=[str(t/s) for s in ('environment','solution','tests')]
from apparatus import fringe,all_nested
from forest import records
from reference import truth
V=.8217

def example():
    h=np.pi/2;field=[['z',0.],['z',0.]]
    def mark(arms):return dict(arms=arms,field=field,herald=[1.,np.pi,0.])
    left=mark([[0.,0.,0.,0.],[h,0.,h,0.]])
    right=mark([[0.,0.,h,0.],[0.,0.,h,0.]])
    both=mark([[0.,0.,0.,0.],[0.,0.,0.,0.]])
    return dict(central=2,left_sign=None,counts=[1,1,None],phase=0.,launch=field,prepare=[[1,0],[0,0]],analyze=None,
                bands=[dict(width=1.,left=left,right=right,both=both)])

def inputs():
    rng=np.random.default_rng(654819)
    def u(a,b):return round(float(rng.uniform(a,b)),4)
    def rotation():return [str(rng.choice(['x','y','z'])),u(-3.,3.)]
    cal=[]
    for _ in range(12):
        q=example();q.update(bands=[],counts=[None,None,None],phase=u(-3.1,3.1))
        cal.append(dict(input=q,value=round(V*np.cos(q['phase']),8)))
    groups={}
    for name in ('one_center','mixed_arcs','opposite_arcs'):
        rows=[]
        for i in range(6):
            for _ in range(60):
                q=example();q.update(central=1 if name=='one_center' else 2,bands=[],phase=u(-2.7,2.7),
                    launch=[rotation(),rotation()],left_sign=[-1,1,None][i%3],prepare=np.round(rng.normal(size=(2,2)),3).tolist(),
                    analyze=None if i in (0,3) else np.round(rng.normal(size=(2,2)),3).tolist())
                q['counts']=([1,1,None] if name=='opposite_arcs' else
                             ([None,None,None] if i%3==0 else [None,None,1] if i%3==1 else [1,None,None]))
                for j in range(1+i%2):
                    band={'width':u(.8,1.7)}
                    for spot in ('left','right','both'):
                        arms=[[u(-1.6,1.6),u(-.7,.7),u(-1.6,1.6),u(-.7,.7)] for _ in range(2)]
                        herald=([.98,u(2.6,3.1),u(-.2,.2)] if name=='opposite_arcs' else
                                [u(.65,1.),u(1.,3.1),u(-.8,.8)])
                        band[spot]=dict(arms=arms,field=[rotation(),rotation()],herald=herald)
                    q['bands'].append(band)
                p,s=fringe(q,records)
                if p>.025:break
            assert p>.025
            bench=truth(q);np.testing.assert_allclose([p,s],bench,atol=3e-8,rtol=0)
            rows.append(dict(input=q,value=round(V*bench[1],9)))
        groups[name]=rows
    return cal,groups

def main(write):
    cal,groups=inputs()
    for dest,v in [(t/'environment/data/calibration.json',cal),(t/'tests/data/calibration.json',cal),(t/'tests/data/hidden.json',groups)]:
        if write:dest.write_text(json.dumps(v,separators=(',',':'))+'\n')
        assert json.loads(dest.read_text())==v
    for name,path in [('oracle',t/'solution/model.py'),('starter',t/'environment/model.py'),('control',base/'scripts/random_singlet_cut_fringe_baseline.py')]:
        model=runpy.run_path(str(path))['Model']().fit(cal)
        errors={g:float(np.max(abs(model.predict([r['input'] for r in rows])-[r['value'] for r in rows]))) for g,rows in groups.items()}
        print(name,errors,flush=True);assert errors['one_center']<3e-8
        if name=='oracle':assert max(errors.values())<3e-8
        else:assert min(errors[n] for n in ('mixed_arcs','opposite_arcs'))>.003
    q=example()
    for method in (lambda q:fringe(q,records),truth):
        np.testing.assert_allclose(method(q),[.2775250920,.5],atol=4e-9)
    np.testing.assert_allclose(fringe(q,all_nested)[1],.125,atol=4e-9)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generate',action='store_true')
    main(p.parse_args().generate)
