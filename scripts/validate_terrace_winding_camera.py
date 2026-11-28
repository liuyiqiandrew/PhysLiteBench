"""Check common-nucleation terrace probabilities and responses to opening a slit."""
import argparse
import importlib.util
import json
import sys
from scipy.integrate import quad
from scipy.special import ndtr
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/terrace-winding-camera"


sys.path.insert(0, str(TASK / "solution"))


def load(relative):
    spec = importlib.util.spec_from_file_location("camera", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load(TASK / "tests/reference.py")
oracle = load(TASK / "solution/model.py")
shortcut = load("scripts/terrace_winding_camera_baseline.py")
starter = load(TASK / "environment/model.py")



def preparations():
    rng = np.random.default_rng(411269)
    groups = {}
    sizes = [2,3,4,5,3,4,5,2]
    lower = [[2],[2,1],[1,2,1],[2,1,2],[1,2],[2,2],[2,3],[2]]
    upper = [[2],[1,2],[3,1],[1,2,2],[3],[2,2],[3,2],[1,1]]
    def pins(multiplicities, L):
        centers = (np.arange(len(multiplicities)) + rng.uniform(.25,.75,len(multiplicities)))*L/len(multiplicities)
        return [round(float(x),3) for x,m in zip(centers,multiplicities) for _ in range(m)]
    for group in ("separate","nucleated","joint","slit"):
        rows = []
        for i,n in enumerate(sizes):
            L = [7.,8.,9.][(i+1)%3]
            bot = pins([1]*n if group=="separate" else lower[i], L)
            top = pins([1]*n if group=="separate" else upper[i], L)
            winding = [0,1,-1,0,-1,1,1,-1][i]
            tag = i % n
            a,b = (round(float(v),3) for v in [rng.uniform(.25,.4),rng.uniform(.6,.73)])
            lo = round(float(rng.uniform(.05,.48)*L),3)
            hi = round(min(L,lo+float(rng.uniform(.22,.50))*L),3)
            cl = round(max(0.,lo + float(rng.uniform(-.18,.12))*L),3)
            ch = round(min(L,hi + float(rng.uniform(-.06,.18))*L),3)
            count = min(n,max(0,round(n*(ch-cl)/L)+(1 if i%4==1 else 0)))
            if group=="nucleated" and i%2==0:
                cl,ch,count=0.,L,n
            doublet = next((j for j,x in enumerate(bot[:-1]) if bot.count(x)==2 and bot[j+1]==x),None)
            if group=="slit":
                assert doublet is not None
                if i%3==0:
                    tag=doublet+i%2
            rows.append(dict(mode="pinned",circumference=L,bottom=bot,top=top,
                             height=round(float(rng.uniform(3.2,4.5)),3),winding=winding,
                             levels=[a,b],tag=tag,window=[lo,hi],count_window=[cl,ch],count=count,
                             open_pair=doublet if group=="slit" else None))
        groups[group]=rows
    return groups

def tagged(q,rate,opening=0.):
    a=q['levels'][0];L=q['circumference'];T=q['height'];bottom=np.array(q['bottom'],float)
    if opening:
        bottom+=np.array([-.5,.5])*opening*L
    top=[q['top'][(i+q['winding'])%2]+((i+q['winding'])//2)*L for i in range(2)]
    g0=bottom[1]-bottom[0];g1=top[1]-top[0]
    k=np.arange(1,70)*np.pi/L
    def edge(v):
        if v==0: return k
        if v==L: return -k*np.cos(k*L)
        return np.sin(k*v)
    x0=edge(g0);x1=edge(g1); normal=(2/L)*sum(x0*x1*np.exp(-2*rate*T*k*k))
    f0=2*x0*np.exp(-2*rate*T*a*k*k)/L;f1=2*x1*np.exp(-2*rate*T*(1-a)*k*k)/L
    midpoint=(1-a)*sum(bottom)/2+a*sum(top)/2;sigma=np.sqrt(rate*T*a*(1-a))
    shifts=np.arange(-6,7)*L;lo,hi=q['window']
    def integrate(g):
        sp=np.sin(k*g);shifted=midpoint+(-1 if q['tag']==0 else 1)*g/2
        interval=sum(ndtr((hi+shifts-shifted)/sigma)-ndtr((lo+shifts-shifted)/sigma))
        return (sp@f0)*(sp@f1)*interval/normal
    return quad(integrate,0,L,epsabs=1e-11)[0]


def example():
    return dict(mode="pinned",circumference=8.,bottom=[2.,2.],top=[3.,3.],height=3.8,winding=1,
                levels=[.3,.69],tag=1,window=[.4,4.4],count_window=[0.,8.],count=2,open_pair=None)


def main(generate):
    cal = [dict(input=dict(mode="wandering",height=h),value=round(2*ref.TRUE_PARAMETER*h,8))
           for h in [.35,.707,.92,1.643,2.21,3.735,4.996,6.207]]
    root = TASK / "tests/data"
    groups = preparations()
    expected = {}
    refinement = 0.
    for group, qs in groups.items():
        rows = []
        for q in qs:
            if q['open_pair'] is None:
                y = ref.image(q,ref.TRUE_PARAMETER)
            else:
                coarse = ref.image(q,ref.TRUE_PARAMETER,step=.006)
                y = ref.image(q,ref.TRUE_PARAMETER,step=.003)
                refinement=max(refinement,abs(coarse-y))
            rows.append(dict(input=q,value=round(float(y),7)))
        expected[group]=rows
    for path, data in [(root/"calibration.json",cal),(TASK/"environment/data/calibration.json",cal),
                       (root/"predictions.json",expected)]:
        if generate:
            path.write_text(json.dumps(data,indent=1)+"\n")
        stored=json.loads(path.read_text())
        if isinstance(data,list):
            assert stored==data
        else:
            assert data.keys()==stored.keys()
            for key,rows in stored.items():
                assert [r['input'] for r in rows]==[r['input'] for r in data[key]]
                np.testing.assert_allclose([r['value'] for r in rows],[r['value'] for r in data[key]],atol=5e-5,rtol=0)
    assert refinement < 0.0002
    print('slit extrapolation difference',refinement)
    for name,mod in [('oracle',oracle),('shortcut',shortcut),('starter',starter)]:
        model=mod.Model().fit(cal)
        assert abs(model.rate-ref.TRUE_PARAMETER)<1e-8
        err={}
        for k, rows in expected.items():
            val=model.predict([r['input'] for r in rows])
            err[k]=float(max(abs(val-[r['value'] for r in rows])))
        print(name,err)
        if name=='oracle':
            assert max(err.values())<2e-4
        else:
            assert err['joint']>.003 and err['slit']>.015
    # Two line paths decompose into an independent center and a gap killed on (0,L).
    for w in (-1,0,1):
        q=dict(example(),winding=w)
        x0=tagged(q,ref.TRUE_PARAMETER)
        np.testing.assert_allclose(oracle.image(q,ref.TRUE_PARAMETER),x0,atol=2e-10,rtol=0)
        s=.001
        p1,p2,p4=[tagged(q,ref.TRUE_PARAMETER,c*s) for c in (1,2,4)]
        curvature=(4*(p2-p1)/3-(p4-p2)/12)/(3*s*s)
        np.testing.assert_allclose(oracle.image(dict(q,open_pair=0),ref.TRUE_PARAMETER),curvature,atol=1e-6,rtol=0)
    q=example()
    assert abs(oracle.image(q,ref.TRUE_PARAMETER)-.26814)<5e-6
    assert abs(oracle.image(dict(q,open_pair=0),ref.TRUE_PARAMETER)+.83267)<5e-6
    # Summing over every possible count removes the second camera.
    q=dict(groups['slit'][2])
    total=sum(oracle.image(dict(q,count=j),ref.TRUE_PARAMETER) for j in range(len(q['top'])+1))
    actual=oracle.image(dict(q,count_window=[0.,q['circumference']],count=len(q['top'])),ref.TRUE_PARAMETER)
    assert abs(total-actual)<2e-7


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    main(parser.parse_args().generate)
