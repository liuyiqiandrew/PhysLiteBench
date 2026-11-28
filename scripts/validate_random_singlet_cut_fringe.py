"""Compare mixed spectral measures to independent finite-aperture integrals."""
import argparse,json,runpy,sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
base=Path(__file__).resolve().parents[1];t=base/'tasks/random-singlet-cut-fringe'
sys.path[:0]=[str(t/s) for s in ('environment','solution','tests')]
from resolution import reading
from reference import truth
V=.8217

def example():
    b=dict(width=2.,arms=[[],[['L','z',np.pi],['L','z',np.pi]]],detectors=[[1.,'z',0.],[1.,'y',np.pi/2]])
    return dict(start='renewed',bands=[b],phase=0.,kind='signed',center=1.,slit=[.15,.15],lock=[0.,0.],dither=[-.3,.3],lag=0.)

def inputs():
    rng=np.random.default_rng(880412)
    def u(a,b):return round(float(rng.uniform(a,b)),4)
    def pick(x):return str(rng.choice(x))
    cal=[]
    for _ in range(12):
        angle=u(-3.,3.);cal.append(dict(input=dict(calibration=True,phase=angle),value=round(V*np.cos(angle),8)))
    groups={}
    for name in ('resolved_splits','reference_lock','beat_aperture'):
        rows=[]
        for i in range(6):
            for _ in range(120):
                q=example();q['phase']=u(-2.7,2.7);q['bands']=[]
                q['start']='renewed' if i%2==0 else 'bulk'
                for j in range(1+i%2):
                    def pulse():return [pick(['L','R']),pick(['x','y','z']),u(-3.1,3.1)]
                    arms=[[pulse() for _ in range(1+int(rng.integers(3)))] for _ in (0,1)]
                    det=[[u(.7,.98),pick(['x','y']),u(-2.8,2.8)] for _ in (0,1)]
                    q['bands'].append(dict(width=u(1.1,2.1),arms=arms,detectors=det))
                T=sum(b['width'] for b in q['bands']);q['center']=u(.36*T,.65*T)
                pos=u(-.26,.26);q['lock']=[u(.25,.68),pos]
                q['dither']=[u(-.52,-.28),u(.31,.6)]
                q['lag']=u(2.5,6.8)*(-1)**i
                if name=='resolved_splits':
                    d=u(-.22,.22);q['slit']=[d,d];q['lag']=0.;q['lock'][0]=0.
                elif name=='reference_lock':
                    q['kind']='beat' if i%2 else 'signed'
                    val=abs(pos) if i%2 else pos
                    if abs(val)<.025:continue
                    q['slit']=[val,val]
                    if i==5:
                        for band in q['bands']:band['detectors']=[[1.,'z',0.],[1.,'z',0.]]
                else:
                    q['kind']='beat';q['slit']=[u(.04,.14),u(.24,.43)]
                if min(abs(q['center']-e) for e in np.cumsum([b['width'] for b in q['bands']]))<.05:continue
                if name=='beat_aperture' and min(abs(abs(pos)-s) for s in q['slit'])<.03:continue
                prob,fringe=reading(q)
                if prob>.007:break
            assert prob>.007 and abs(fringe)<1.+1e-10
            ref=truth(q,width=.003)
            np.testing.assert_allclose([prob,fringe],ref,atol=2e-7,rtol=0)
            rows.append(dict(input=q,value=round(V*ref[1],9)))
        groups[name]=rows
    return cal,groups

def main(write):
    cal,groups=inputs()
    for dest,v in [(t/'environment/data/calibration.json',cal),(t/'tests/data/calibration.json',cal),(t/'tests/data/hidden.json',groups)]:
        if write:dest.write_text(json.dumps(v,separators=(',',':'))+'\n')
        assert json.loads(dest.read_text())==v
    for name,path in [('oracle',t/'solution/model.py'),('starter',t/'environment/model.py'),('control',base/'scripts/random_singlet_cut_fringe_baseline.py')]:
        m=runpy.run_path(str(path))['Model']().fit(cal)
        errors={g:float(np.max(abs(m.predict([r['input'] for r in rows])-[r['value'] for r in rows]))) for g,rows in groups.items()}
        print(name,errors,flush=True)
        if name=='oracle':assert max(errors.values())<2e-7
        elif name=='starter':assert min(errors.values())>.003
        else:
            assert errors['resolved_splits']<2e-7
            assert min(errors[g] for g in ('reference_lock','beat_aperture'))>.003
    q=example()
    def flux(z,t):
        d=np.sqrt(5+4*z);return 2*np.exp(-1.5*t)*np.sinh(d*t/2)/d
    def count(z,t):
        d=np.sqrt(5+4*z);return np.exp(-1.5*t)*(np.cosh(d*t/2)+3*np.sinh(d*t/2)/d)
    def target(m):
        shared=.25*m*flux(.25*m,1)*count(m,1)/.6
        def area(d):return .125*m*m*flux(.25*m,1-abs(d))*flux(.5*m,2*abs(d))*count(m,1-abs(d))
        smooth=2*(quad(area,-.15,0.)[0]+quad(area,0.,.45)[0])/.6
        return shared,smooth
    a,b=target(1.);u,v=target(-1.)
    expected=[a+b,(u+v)/(a+b)]
    print('shared and ordinary densities',a,b,'fringe',expected[1],flush=True)
    for method in (reading,truth):np.testing.assert_allclose(method(q),expected,atol=1e-9,rtol=0)
    np.testing.assert_allclose(reading(q,drop_shared=True)[1],v/b,atol=1e-10,rtol=0)
    q.update(lock=[.4,.12],slit=[.12,.12],lag=4.1)
    q['bands'][0]['detectors']=[[1.,'z',0.],[1.,'z',0.]]
    np.testing.assert_allclose(reading(q),truth(q,width=.002),rtol=0,atol=3e-8)
    assert reading(q)[0]>0.01

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generate',action='store_true')
    main(p.parse_args().generate)
