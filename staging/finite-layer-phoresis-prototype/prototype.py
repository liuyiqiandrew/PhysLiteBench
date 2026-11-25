"""Finite interaction-layer phoresis: reciprocal force kernel vs direct Stokes BVP."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_bvp, quad


def fields(a, width, strength, tol=2e-9):
    end=a+width
    def potential(r):
        z=np.clip(1-(r-a)/width,0,1)
        return strength*z*z
    def derivative(r):return -2*strength*np.clip(1-(r-a)/width,0,1)/width
    def rhs(r,y):return np.array([y[1],-(2/r-derivative(r))*y[1]+2*y[0]/r**2])
    def bc(left,right):return np.array([left[1],end*right[1]+2*right[0]-3*end])
    r=np.linspace(a,end,161);guess=np.array([r+a**3/(2*r*r),1-a**3/r**3])
    solution=solve_bvp(rhs,bc,r,guess,tol=tol,max_nodes=10000)
    assert solution.success,solution.message
    concentration=lambda r:np.exp(-potential(r))*solution.sol(r)[0]
    return concentration,derivative


def velocities(a,width,strength,tol=2e-9):
    c,up=fields(a,width,strength,tol)
    end=a+width
    # eta=kBT=far gradient=1. Translation reciprocal field and the direct
    # solute reaction give this combined kernel; the source keeps its planar
    # near-wall expansion while retaining the same full concentration response.
    exact=2/(9*a)*quad(lambda r:(r*r-1.5*a*r+.5*a**3/r)*c(r)*up(r),a,end,epsabs=1e-11,epsrel=1e-11)[0]
    local=1/(3*a)*quad(lambda r:(r-a)**2*c(r)*up(r),a,end,epsabs=1e-11,epsrel=1e-11)[0]
    def rhs(r,y):
        force=-c(r)*up(r)
        return np.array([y[1],y[2],y[3],force+4*y[2]/r**2-8*y[1]/r**3+8*y[0]/r**4])
    def bc(left,right):
        return np.array([left[0],left[1],right[2]-2*right[0]/end**2,right[3]-2*right[1]/end**2+4*right[0]/end**3])
    r=np.linspace(a,end,201)
    stokes=solve_bvp(rhs,bc,r,np.zeros((4,len(r))),tol=tol,max_nodes=10000)
    assert stokes.success,stokes.message
    y=stokes.sol(end)
    direct=-2/(3*end)*(y[1]+y[0]/end)
    flat=-quad(lambda z:z*np.expm1(-strength*(1-z/width)**2),0,width,epsabs=1e-12)[0]
    return dict(radius=a,width=width,strength=strength,reciprocal_velocity=exact,direct_stokes_velocity=direct,planar_kernel_velocity=local,shortcut_relative_error=abs(local/exact-1),reference_absolute_error=abs(exact-direct),flat_wall_fluid_mobility=flat,particle_to_opposite_flat_mobility_ratio=exact/(-flat))


if __name__=='__main__':
    cases=[velocities(a,a*ratio,strength) for a in [.7,1.,1.3] for ratio in [.3,.8,1.5,2.5] for strength in [-2.,-.6,.6,2.]]
    thin=[velocities(1.,ratio,1.2) for ratio in [.1,.03,.01,.003]]
    report={'status':'prototype_only_not_evaluated','physical_model':'Ideal dilute neutral solute with compact quadratic radial interaction, no advection; full l=1 concentration, no-slip force-free particle. Direct Stokes far conditions exclude the Stokeslet and growing r^4 mode.','cases':cases,'thin_layer_limit':thin,'max_reference_error':max(c['reference_absolute_error'] for c in cases),'hidden_screen_min_relative_error':min(c['shortcut_relative_error'] for c in cases if c['width']/c['radius']>=1.5),'notes':['All velocities per unit imposed solute concentration gradient; eta=kBT=1.','Planar calibration reads far fluid velocity relative to fixed wall; spherical readout is particle velocity, with the opposite leading sign.','This prototype shares the concentration solve between the two hydrodynamic routes; a final reference should also discretize concentration independently.']}
    out=Path(__file__).with_name('report.json');out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','notes')}))
