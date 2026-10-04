import numpy as np
from scipy.integrate import solve_bvp

PARAMETER = 'mobility'
TRUE_PARAMETER = 0.65
TAU2 = 0.4


def experiment(width, ex, b0=0., b1=0., b2=0., observable='current', contacts=(-1.,1.)):
    e = dict(width=width,electric_field=ex,magnetic_field=b0,field_gradient=b1,
             field_modulation=b2,observable=observable)
    if observable == 'transverse_voltage': e['contacts']=list(contacts)
    return e


def field(y,e):
    return e['magnetic_field']+e['field_gradient']*2*y/e['width']+e['field_modulation']*np.cos(2*np.pi*y/e['width'])


def solution(e,mobility,tol=1e-9):
    w=e['width']; grid=np.linspace(-w/2,w/2,81)
    # Unknowns: velocity, kinetic xy momentum flux, integrated current, integrated Lorentz force.
    def ode(y,z):
        b=field(y,e)
        # Solve the xx, xy, yy tensor relaxation/rotation equations locally.
        shear=-(1+4*b*b*TAU2*TAU2)*z[1]/TAU2
        return np.vstack((shear,e['electric_field']-z[0]/mobility,z[0],b*z[0]))
    def boundary(left,right):
        return np.array([left[0],right[0],left[2],left[3]])
    result=solve_bvp(ode,boundary,grid,np.zeros((4,len(grid))),tol=tol,max_nodes=10000)
    assert result.success,result.message
    return result


def predict(experiments,mobility):
    values=[]
    for e in experiments:
        result=solution(e,mobility); w=e['width']
        if e['observable']=='current':
            value=result.sol(w/2)[2]/w
        else:
            a,b=np.array(e['contacts'])*w/2
            za,zb=result.sol(a),result.sol(b)
            # yy stress follows from the independently solved kinetic moment equations.
            value=zb[3]-za[3]-2*TAU2*(field(b,e)*zb[1]-field(a,e)*za[1])
        values.append(float(value))
    return np.array(values)


def calibration_inputs():
    return [experiment(w,ex) for _ in range(2) for w in [.6,1.,1.4,2.,2.5,3.]
            for ex in [-.05,-.04,-.03,-.02,-.01,.01,.02,.03,.04,.05]]


def hidden_inputs():
    return {
      'uniform_field': [experiment(w,ex,b,observable='transverse_voltage')
                        for w in [.55,.9,1.5,2.4] for ex in [-.045,.035] for b in [-2.2,.7,1.6]],
      'gradient_field': [experiment(w,.04,b0,b1,.0,'transverse_voltage',contacts)
                         for w in [.7,1.3,2.5] for b0,b1 in [(.4,1.8),(-1.1,2.3),(1.7,-1.5)]
                         for contacts in [(-1.,.3),(-.4,1.),(-.7,.6)]],
      'modulated_field': [experiment(w,-.035,b0,b1,b2,'transverse_voltage',contacts)
                         for w in [.8,1.7,2.8] for b0,b1,b2 in [(.3,-.8,1.9),(-1.2,.5,-1.6),(1.5,1.1,-2.3)]
                         for contacts in [(-1.,1.),(-.8,.4),(-.3,.9)]]}
