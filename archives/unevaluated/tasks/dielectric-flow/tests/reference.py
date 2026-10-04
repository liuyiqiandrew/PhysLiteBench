"""Face-current finite volumes, Maxwell-stress divergence and staggered Stokes flow."""
from functools import lru_cache
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

PARAMETER='viscosity'
TRUE_PARAMETER=1.1


def mode(wave,amplitude,phase=0.):return dict(wave=list(wave),amplitude=float(amplitude),phase=float(phase))


def experiment(epsilon,sigma=None,field=(.3,.1),component=0,wave=(1,1),phase=-np.pi/2):
    return dict(permittivity_modes=epsilon,conductivity_modes=epsilon if sigma is None else sigma,field=list(field),component=int(component),detector_wave=list(wave),detector_phase=float(phase))


def calibration_inputs():
    patterns=[[mode((1,0),.35),mode((1,1),.3,.4)],[mode((1,1),.32,.3),mode((0,1),-.3,-.6)]]
    return [experiment(pattern,field=f,component=c,wave=q,phase=p) for pattern in patterns for f in [(.3,.1),(-.1,.3)] for c,q in [(0,(0,1)),(1,(1,0)),(0,(1,1))] for p in [0.,-np.pi/2]]*6


def hidden_inputs():
    a=[mode((1,0),.35),mode((1,1),.3,.4)]
    b=[mode((0,1),.45,.2),mode((1,-1),.2,-.3)]
    c=[mode((1,0),-.3,.1),mode((1,1),.3,.6)]
    d=[mode((1,1),.3,-.5),mode((2,-1),.32,.7)]
    e=[mode((0,1),-.28,.8),mode((1,0),.35,-.2)]
    return {'crossed_profiles':[experiment(a,b,field=f,component=i,wave=q,phase=p) for f in [(.3,.1),(.1,.3)] for i,q in [(0,(0,1)),(1,(1,0)),(0,(1,1)),(0,(1,-1))] for p in [0.,-np.pi/2]],
            'phase_mismatch':[experiment(a,c,field=f,component=i,wave=q,phase=p) for f in [(.3,.1),(-.15,.3)] for i,q in [(0,(0,1)),(1,(1,0)),(0,(1,1)),(0,(1,-1))] for p in [0.,-np.pi/2]],
            'oblique_profiles':[experiment(d,e,field=f,component=i,wave=q,phase=p) for f in [(.3,.1),(-.1,.3)] for i,q in [(0,(0,1)),(1,(1,0)),(0,(1,1)),(0,(2,-1))] for p in [0.,-np.pi/2]]}


@lru_cache(128)
def fields(key,points):
    e=json.loads(key);h=2*np.pi/points
    x,y=np.meshgrid(np.arange(points)*h,np.arange(points)*h,indexing='ij')
    def profile(modes):return 1+sum(m['amplitude']*np.cos(m['wave'][0]*x+m['wave'][1]*y+m['phase']) for m in modes)+np.zeros_like(x)
    epsilon=profile(e['permittivity_modes']);sigma=100*profile(e['conductivity_modes'])
    indices=np.arange(points**2).reshape(points,points);rows=[];cols=[];values=[];rhs=np.zeros_like(x)
    for axis in [0,1]:
        conductance=(sigma+np.roll(sigma,-1,axis))/2
        face=conductance/h**2;neighbor=np.roll(indices,-1,axis)
        for i,j,v in [(indices,indices,face),(indices,neighbor,-face),(neighbor,indices,-face),(neighbor,neighbor,face)]:
            rows.extend(i.ravel());cols.extend(j.ravel());values.extend(v.ravel())
        rhs-=e['field'][axis]*(conductance-np.roll(conductance,1,axis))/h
    matrix=coo_matrix((values,(rows,cols)),shape=(points**2,points**2)).tocsc()
    potential=np.zeros(points**2);potential[1:]=spsolve(matrix[1:,1:],rhs.ravel()[1:]);potential=potential.reshape(points,points)
    electric=[];current=[]
    for axis in [0,1]:
        face=e['field'][axis]-(np.roll(potential,-1,axis)-potential)/h
        current.append((sigma+np.roll(sigma,-1,axis))/2*face)
        electric.append((face+np.roll(face,1,axis))/2)
    ex,ey=electric
    stress_xx=.5*epsilon*(ex*ex-ey*ey);stress_yy=-stress_xx
    stress_xy=epsilon*ex*ey
    corner=(stress_xy+np.roll(stress_xy,-1,0)+np.roll(stress_xy,-1,1)+np.roll(np.roll(stress_xy,-1,0),-1,1))/4
    fx=(np.roll(stress_xx,-1,0)-stress_xx+corner-np.roll(corner,1,1))/h
    fy=(np.roll(stress_yy,-1,1)-stress_yy+corner-np.roll(corner,1,0))/h
    modes=np.fft.fftfreq(points,1/points);g=(np.exp(1j*modes*h)-1)/h
    gx,gy=np.meshgrid(g,g,indexing='ij');square=abs(gx)**2+abs(gy)**2
    inverse=np.zeros_like(square);inverse[square>0]=1/square[square>0]
    force=np.fft.fft2(np.array([fx,fy]),axes=(-2,-1))
    longitudinal=(gx.conj()*force[0]+gy.conj()*force[1])*inverse
    velocity=np.fft.ifft2((force-np.array([gx,gy])*longitudinal)*inverse,axes=(-2,-1)).real
    return x,y,velocity,np.array(current),np.array([fx,fy])


def raw_predict(experiments,viscosity,points):
    out=[]
    for e in experiments:
        prep={k:e[k] for k in ['permittivity_modes','conductivity_modes','field']}
        x,y,velocity,_,_=fields(json.dumps(prep,sort_keys=True),points)
        q=e['detector_wave'];phase=q[0]*x+q[1]*y+e['detector_phase']+q[e['component']]*np.pi/points
        out.append(2*np.mean(velocity[e['component']]*np.cos(phase))/viscosity)
    return np.array(out)


def predict(experiments,viscosity,points=96):
    return (4*raw_predict(experiments,viscosity,points)-raw_predict(experiments,viscosity,points//2))/3
