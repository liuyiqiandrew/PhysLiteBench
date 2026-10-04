"""Navier-slip phoresis: curved force kernel versus fixed-thrust drag correction.

Prototype only. No calibration files, packaged task or agent evaluations.
"""
from pathlib import Path
import importlib.util, json, hashlib, itertools, time
from functools import lru_cache
import numpy as np
from scipy.integrate import quad,solve_bvp
ROOT=Path(__file__).resolve().parents[2]
REF=ROOT/'archives/finite-layer-phoresis-r1/tasks/finite-layer-phoresis/tests/reference.py'
spec=importlib.util.spec_from_file_location('r1_fem',REF)
fem=importlib.util.module_from_spec(spec);spec.loader.exec_module(fem)

@lru_cache(256)
def fields(radius,width,strength):
    end=radius+width
    U=lambda r:strength*((end-r)/width)**2
    derivative=lambda r:-2*strength*(end-r)/width**2
    def rhs(r,y):return np.array([y[1],-(2/r-derivative(r))*y[1]+2*y[0]/r**2])
    def bc(l,r):return np.array([l[1],end*r[1]+2*r[0]-3*end])
    grid=np.linspace(radius,end,129)
    guess=np.array([grid+radius**3/(2*grid**2),1-radius**3/grid**3])
    solution=solve_bvp(rhs,bc,grid,guess,tol=2e-10,max_nodes=10000)
    assert solution.success,solution.message
    return lambda r:np.exp(-U(r))*solution.sol(r)[0],derivative

def velocity(radius,width,strength,slip,full=True):
    if strength==0:return 0.
    c,up=fields(radius,width,strength)
    beta=slip/radius;drag=(1+2*beta)/(1+3*beta)
    A=1.5*drag if full else 1.5
    B=.5/(1+3*beta) if full else .5
    kernel=lambda r:r*r-A*radius*r+B*radius**3/r
    return 2/(9*radius*drag)*quad(lambda r:kernel(r)*c(r)*up(r),radius,radius+width,epsabs=2e-11,epsrel=2e-11)[0]

def direct(radius,width,strength,slip,cells=256):
    if strength==0:return dict(velocity=0.,force_balance=0.,boundary=0.,slip_dissipation=0.)
    end=radius+width;c=fem.concentration(radius,width,strength,cells)
    up=lambda r:-2*strength*(end-r)/width**2
    def rhs(r,y):
        return np.array([y[1],y[2],y[3],-c(r)*up(r)+4*y[2]/r**2-8*y[1]/r**3+8*y[0]/r**4])
    def bc(l,r):
        return np.array([l[0],(1+2*slip/radius)*l[1]-slip*l[2],r[2]-2*r[0]/end**2,r[3]-2*r[1]/end**2+4*r[0]/end**3])
    grid=np.linspace(radius,end,cells+1)
    solution=solve_bvp(rhs,bc,grid,np.zeros((4,cells+1)),tol=2e-10,max_nodes=20000)
    assert solution.success,solution.message
    wall=solution.sol(radius);outer=solution.sol(end)
    speed=-2/(3*end)*(outer[1]+outer[0]/end)
    # Integrate total Newtonian solution traction plus smooth-U reaction.
    # Navier slip leaves psi' nonzero, unlike the earlier no-slip formula.
    fluid=4*np.pi*radius**2/3*(-wall[3]+2*wall[2]/radius+2*wall[1]/radius**2)
    reaction=4*np.pi/3*quad(lambda r:r*r*c(r)*up(r),radius,end,epsabs=2e-11,epsrel=2e-11)[0]
    slip_dissipation=8*np.pi/3*wall[1]**2/slip if slip else 0.
    return dict(velocity=float(speed),force_balance=abs(fluid+reaction)/max(1.,abs(fluid),abs(reaction)),boundary=float(max(abs(bc(wall,outer)))),slip_dissipation=float(slip_dissipation))

def main():
    start=time.perf_counter();rows=[]
    for a,wratio,s,bratio in itertools.product([.7,1.3],[.3,.7,1.4,2.4],[-1.5,1.5],[.3,1.0]):
        w=a*wratio;b=a*bratio
        full=velocity(a,w,s,b);bad=velocity(a,w,s,b,False)
        coarse=direct(a,w,s,b,128);fine=direct(a,w,s,b,256)
        reference=(4*fine['velocity']-coarse['velocity'])/3
        rows.append(dict(radius=a,width=w,strength=s,slip=b,oracle=full,shortcut=bad,reference=reference,
                         relative_error=abs(bad/full-1),reference_error=abs(full-reference),
                         force_balance=fine['force_balance'],boundary_residual=fine['boundary'],slip_dissipation=fine['slip_dissipation']))
    calibration=[]
    for a,w,s in itertools.product([.75,1.2],[.35,1.,2.],[-1.5,-.6,.6,1.5]):
        calibration.append(dict(radius=a,width=w,strength=s,slip=0.,oracle=velocity(a,w,s,0),shortcut=velocity(a,w,s,0,False)))
    scaling=[]
    for full in [False,True]:
        one=velocity(.8,.56,1.1,.64,full);larger=velocity(1.2,.84,1.1,.96,full)
        scaling.append(abs(larger/one-2.25))
    # Fixed b/width in a thin layer: opposite of planar Navier-slip wall flow.
    width=.0003;slip=.0003;strength=1.2
    wall=-quad(lambda z:(z+slip)*np.expm1(-strength*(1-z/width)**2),0,width,epsabs=1e-17,epsrel=1e-11)[0]
    thin=velocity(1.,width,strength,slip)
    report=dict(status='prototype_only',family='finite-layer-phoresis',proposed_revision=2,
                r1_outcome='1/3 passes, two reviewed physical failures; archived unchanged',
                approximation='Exact curved concentration and no-slip force kernel with exact Navier slip drag, but fixed phoretic thrust.',
                direct_reference='Independent conservative concentration FEM plus force-free Stokes streamfunction with Navier tangential-stress boundary; 128/256 Richardson.',
                candidate_domain=dict(radius=[.7,1.3],width=[.05,3.5],strength=[-2,2],slip=[0,2]),
                calibration=calibration,cases=rows,
                checks=dict(max_reference_error=max(r['reference_error'] for r in rows),minimum_relative_gap=min(r['relative_error'] for r in rows),maximum_relative_gap=max(r['relative_error'] for r in rows),minimum_abs_signal=min(abs(r['oracle'])/1.2 for r in rows),
                            force_balance_max=max(r['force_balance'] for r in rows),boundary_residual_max=max(r['boundary_residual'] for r in rows),
                            nonnegative_interfacial_dissipation=all(r['slip_dissipation']>=0 for r in rows),
                            calibration_equivalence=max(abs(r['oracle']-r['shortcut']) for r in calibration),
                            geometric_scaling_error=max(scaling),thin_sphere_to_opposite_wall_ratio=thin/(-wall),
                            zero_strength_exact=velocity(1,.7,0,.5)==0,
                            positive_drag_range=[2/3,1]),
                seconds=time.perf_counter()-start,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),REF]})
    assert report['checks']['max_reference_error']<1e-7
    assert report['checks']['force_balance_max']<1e-8
    assert report['checks']['boundary_residual_max']<1e-8
    assert report['checks']['calibration_equivalence']==0
    assert max(scaling)<1e-8
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['calibration','cases']},indent=2))
if __name__=='__main__':main()
