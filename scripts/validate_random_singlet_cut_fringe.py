"""Cross-check tagged jump densities against independently resolved optical windows."""
import argparse,json,runpy,sys
from pathlib import Path
import numpy as np
base=Path(__file__).resolve().parents[1];t=base/'tasks/random-singlet-cut-fringe'
sys.path[:0]=[str(t/s) for s in ('environment','solution','tests')]
from lines import measurement
from reference import truth,window
V=.8217

def example():
    spec=dict(arms=[[],[['L','z',np.pi,0.],['L','z',np.pi,0.]]],detectors=[[1.,'y',0.,0.],[1.,'y',np.pi,0.]])
    return dict(central=0,left_sign=None,lines=[1.,1.],phase=0.,bands=[dict(width=1.,left=spec,right=spec,both=spec)])

def make():
    rng=np.random.default_rng(956907)
    def u(a,b):return round(float(rng.uniform(a,b)),4)
    def pick(x):return str(rng.choice(x))
    cal=[]
    for _ in range(12):
        q=example();q.update(central=1,bands=[],lines=[None,None],phase=u(-3.1,3.1))
        cal.append(dict(input=q,value=round(V*np.cos(q['phase']),8)))
    groups={}
    for name in ('silent_channel','separated_lines','coincident_line'):
        rows=[]
        for i in range(8):
            for _ in range(80):
                q=dict(central=i%3,left_sign=[-1,1,None][i%3],phase=u(-1.5,1.5),bands=[])
                for j in range(1+i%2):
                    band=dict(width=u(1.35,2.2))
                    for g in ('left','right','both'):
                        def pulse():return [pick(['L','R']),pick(['x','y','z']),u(-3.1,3.1),u(-2.9,2.9)]
                        arms=[[pulse() for _ in range(1+int(rng.integers(2)))] for _ in (0,1)]
                        detectors=[[u(.67,.98),pick(['x','y']),u(-2.8,2.8),u(-1.9,1.9)] for _ in (0,1)]
                        band[g]=dict(arms=arms,detectors=detectors)
                    q['bands'].append(band)
                width=sum(b['width'] for b in q['bands'])
                first=u(.22*width,.42*width);second=u(.62*width,.88*width)
                if name=='silent_channel':
                    q['lines']=[None,None] if i in (1,4) else [None,second] if i%2 else [first,None]
                elif name=='separated_lines':q['lines']=[first,second] if i%2 else [second,first]
                else:q['lines']=[second,second]
                marks=[x for x in q['lines'] if x is not None]
                if any(abs(x-q['bands'][0]['width'])<.035 for x in marks):continue
                density,signal=measurement(q)
                if density>.006:break
            assert density>.006 and abs(signal)<1.+1e-10
            ref=truth(q)
            np.testing.assert_allclose([density,signal],ref,rtol=0,atol=8e-8)
            rows.append(dict(input=q,value=round(V*ref[1],9)))
        groups[name]=rows
    return cal,groups

def main(write):
    cal,groups=make()
    for dest,v in [(t/'environment/data/calibration.json',cal),(t/'tests/data/calibration.json',cal),(t/'tests/data/hidden.json',groups)]:
        if write:dest.write_text(json.dumps(v,separators=(',',':'))+'\n')
        assert json.loads(dest.read_text())==v
    for name,path in [('oracle',t/'solution/model.py'),('starter',t/'environment/model.py'),('control',base/'scripts/random_singlet_cut_fringe_baseline.py')]:
        m=runpy.run_path(str(path))['Model']().fit(cal)
        errors={g:float(np.max(abs(m.predict([r['input'] for r in rows])-[r['value'] for r in rows]))) for g,rows in groups.items()}
        print(name,errors,flush=True)
        if name=='oracle':assert max(errors.values())<1e-7
        else:assert min(errors.values())>.003
    q=example();a=np.sqrt(3);b=np.sqrt(7)
    exact=[np.exp(-1.5)*np.sinh(b/2)/b,-np.sinh(a/2)*b/(a*np.sinh(b/2))]
    np.testing.assert_allclose(measurement(q),exact,rtol=0,atol=1e-10)
    np.testing.assert_allclose(truth(q),exact,rtol=0,atol=1e-8)
    print('analytic shared record',exact,flush=True)
    # Coincident hits scale with one power of energy resolution, separated ones with two.
    for family,power in [('coincident_line',1),('separated_lines',2)]:
        test=groups[family][0]['input']
        estimates=[h**power*window(test,h)[0].real for h in (.004,.002)]
        assert abs(estimates[0]/estimates[1]-2**power)<.04
        print(family,'raw two-width count ratio',estimates[0]/estimates[1],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generate',action='store_true')
    main(p.parse_args().generate)
