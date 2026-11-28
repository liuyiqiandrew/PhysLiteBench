"""Calibrate the carrier interferometer and compare backward nested channels with forward tomography."""
import argparse,json,runpy,sys
from pathlib import Path
import numpy as np

base=Path(__file__).resolve().parents[1];t=base/'tasks/random-singlet-cut-fringe'
sys.path[:0]=[str(t/s) for s in ('environment','solution','tests')]
from apparatus import interferometer
from partners import partner_map
from reference import truth

V=.8217

def example():
    ports=dict(arms=[[0.,0.,0.,0.],[np.pi/2,0.,np.pi/2,0.]],herald=[1.,np.pi,0.])
    return dict(central=1,left_sign=None,count=1,phase=0.,prepare=[[1,0],[0,0]],analyze=None,
                turnaround=[['z',0.],['z',0.]],bands=[dict(width=1.,left=ports,right=ports,both=ports)])

def make():
    rng=np.random.default_rng(553962)
    def u(a,b):return round(float(rng.uniform(a,b)),4)
    cal=[]
    for _ in range(12):
        q=example();q.update(bands=[],count=None,phase=u(-3.1,3.1))
        cal.append(dict(input=q,value=round(V*np.cos(q['phase']),8)))
    groups={}
    for name in ('outbound_only','singlet_echo','selected_carrier'):
        rows=[]
        for i in range(6):
            for _ in range(35):
                q=example();q.update(central=1+i%2,left_sign=[-1,1,None][i%3],phase=u(-2.3,2.3),bands=[],
                    count=None if i%3==0 else i%2,
                    turnaround=[[str(rng.choice(['x','y','z'])),u(-1.8,1.8)] for _ in range(2)],
                    prepare=np.round(rng.normal(size=(2,2)),3).tolist())
                if name=='selected_carrier':
                    q['analyze']=np.round(rng.normal(size=(2,2)),3).tolist()
                    q['count']=i%3
                for j in range(1+i%3):
                    band=dict(width=u(.65,1.4))
                    for spot in ('left','right','both'):
                        arms=[[u(-1.6,1.6),u(-.9,.9),u(-1.6,1.6),u(-.9,.9)] for _ in range(2)]
                        if name=='outbound_only':
                            for arm in arms:arm[2:]=[0.,0.]
                        band[spot]=dict(arms=arms,herald=[u(.55,.98),u(1.1,3.1),u(-1.1,1.1)])
                    q['bands'].append(band)
                p,fringe=interferometer(q,partner_map)
                if p>.025:break
            assert p>.025
            direct=truth(q);np.testing.assert_allclose([p,fringe],direct,atol=2e-8,rtol=0)
            rows.append(dict(input=q,value=round(V*direct[1],9)))
        groups[name]=rows
    return cal,groups

def main(write):
    cal,groups=make()
    for dest,v in [(t/'environment/data/calibration.json',cal),(t/'tests/data/calibration.json',cal),(t/'tests/data/hidden.json',groups)]:
        if write:dest.write_text(json.dumps(v,separators=(',',':'))+'\n')
        assert json.loads(dest.read_text())==v
    for name,path in [('oracle',t/'solution/model.py'),('starter',t/'environment/model.py'),('control',base/'scripts/random_singlet_cut_fringe_baseline.py')]:
        model=runpy.run_path(str(path))['Model']().fit(cal)
        errors={g:float(np.max(abs(model.predict([r['input'] for r in rows])-[r['value'] for r in rows]))) for g,rows in groups.items()}
        print(name,errors,flush=True)
        assert errors['outbound_only']<2e-8
        if name=='oracle':assert max(errors.values())<2e-8
        else:assert min(errors[g] for g in ('singlet_echo','selected_carrier'))>.003
    q=example()
    for calc in (lambda q:interferometer(q,partner_map),truth):
        np.testing.assert_allclose(calc(q),[.7574027568135,.5],atol=2e-10,rtol=0)
        np.testing.assert_allclose(calc(dict(q,analyze=q['prepare']))[1],2/3,atol=2e-10)
    np.testing.assert_allclose(interferometer(q)[1],-.25,atol=2e-10)
    classical=lambda e,l:partner_map(e,l,resolve=True)
    np.testing.assert_allclose(interferometer(q,classical)[1],0.,atol=2e-10)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generate',action='store_true')
    main(p.parse_args().generate)
