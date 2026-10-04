"""Conservative face-flux composition evolution and staggered Stokes projection."""
from functools import lru_cache
import numpy as np

TRUE_PARAMETER = .04


def mode(wave,amplitude,phase=0.):
    return {'wave':list(wave),'amplitude':float(amplitude),'phase':float(phase)}


def experiment(modes,time,detector_wave,quadrature='cosine'):
    return {'modes':modes,'time':float(time),'detector_wave':list(detector_wave),'quadrature':quadrature}


def calibration_inputs():
    return [experiment([mode([k,0],amplitude)],t,[k,0])
            for k in [1,2,3] for amplitude in [.15,-.25] for t in np.linspace(.1,5.5,20)]


def hidden_inputs():
    a=[mode([1,0],.2),mode([0,2],.14)]
    b=[mode([1,1],.19,.4),mode([2,-1],.15,-.7)]
    c=[mode([1,0],.22,.3),mode([0,2],.12,-.5)]
    return {'mode_redistribution':[experiment(a,t,q) for t in np.linspace(.25,4.,10) for q in [[1,0],[0,2]]],
            'oblique_patterns':[experiment(b,t,q,z) for t in np.linspace(.2,3.,8) for q in [[1,1],[2,-1]] for z in ['cosine','sine']],
            'generated_modes':[experiment(c,t,[2,2],z) for t in np.linspace(.3,2.8,10) for z in ['cosine','sine']]}


@lru_cache(None)
def geometry(points):
    dx=2*np.pi/points;x=np.arange(points)*dx
    xx,yy=np.meshgrid(x,x,indexing='ij')
    k=np.fft.fftfreq(points,1/points)
    derivative=(np.exp(1j*k*dx)-1)/dx
    gx,gy=np.meshgrid(derivative,derivative,indexing='ij')
    square=abs(gx)**2+abs(gy)**2
    inv=np.zeros_like(square);inv[square>0]=1/square[square>0]
    return dx,xx,yy,gx,gy,square,inv


def nonlinear(c,points):
    dx,_,_,gx,gy,square,inv=geometry(points)
    # Chemical potential is the variation of the cell free energy with
    # nearest-neighbor gradient energy. Force and transport use the same
    # cell-to-face averages, giving an exact discrete work identity.
    lap=(np.roll(c,-1,0)+np.roll(c,1,0)+np.roll(c,-1,1)+np.roll(c,1,1)-4*c)/dx**2
    mu=c-.2*lap
    cx=(c+np.roll(c,-1,0))/2;cy=(c+np.roll(c,-1,1))/2
    fx=-cx*(np.roll(mu,-1,0)-mu)/dx
    fy=-cy*(np.roll(mu,-1,1)-mu)/dx
    fxh=np.fft.fft2(fx);fyh=np.fft.fft2(fy)
    pressure_component=(np.conj(gx)*fxh+np.conj(gy)*fyh)*inv
    ux=(fxh-gx*pressure_component)*inv/.004
    uy=(fyh-gy*pressure_component)*inv/.004
    vx=np.fft.ifft2(ux).real;vy=np.fft.ifft2(uy).real
    jx=cx*vx;jy=cy*vy
    advection=-((jx-np.roll(jx,1,0))+(jy-np.roll(jy,1,1)))/dx
    return np.fft.fft2(advection),vx,vy


def profile(e,times,mobility,points=128,max_step=.005):
    _,x,y,_,_,square,_=geometry(points)
    c=sum(m['amplitude']*np.cos(m['wave'][0]*x+m['wave'][1]*y+m['phase']) for m in e['modes'])
    state=np.fft.fft2(c)
    linear=-mobility*square*(1+.2*square)
    time=0.;out=[]
    for target in times:
        if target>time:
            count=max(1,int(np.ceil((target-time)/max_step)));step=(target-time)/count
            # Exponential midpoint: independent from the oracle's ETD RK4.
            E=np.exp(step*linear);E2=np.exp(.5*step*linear)
            Q=np.full_like(linear,step);Q2=np.full_like(linear,.5*step)
            use=linear!=0
            Q[use]=np.expm1(step*linear[use])/linear[use]
            Q2[use]=np.expm1(.5*step*linear[use])/linear[use]
            for _ in range(count):
                n=nonlinear(np.fft.ifft2(state).real,points)[0]
                midpoint=E2*state+Q2*n
                nmid=nonlinear(np.fft.ifft2(midpoint).real,points)[0]
                state=E*state+Q*nmid
            time=target
        out.append(np.fft.ifft2(state).real)
    return np.array(out)


def predict(experiments,mobility,points=128,max_step=.005):
    out=np.empty(len(experiments));groups={}
    for i,e in enumerate(experiments):
        if len(e['modes'])==1:
            m=e['modes'][0];k=np.array(m['wave']);q=np.array(e['detector_wave'])
            sign=1 if np.array_equal(k,q) else (-1 if np.array_equal(k,-q) else 0)
            k2=float(k@k);amplitude=m['amplitude']*np.exp(-mobility*k2*(1+.2*k2)*e['time'])
            out[i]=sign*0. if not sign else amplitude*(np.cos(m['phase']) if e['quadrature']=='cosine' else -sign*np.sin(m['phase']))
            continue
        key=tuple((tuple(m['wave']),m['amplitude'],m['phase']) for m in e['modes'])
        groups.setdefault(key,[]).append((i,e))
    for rows in groups.values():
        times=sorted({e['time'] for _,e in rows});states=profile(rows[0][1],times,mobility,points,max_step)
        _,x,y,*_=geometry(points)
        for i,e in rows:
            phase=e['detector_wave'][0]*x+e['detector_wave'][1]*y
            detector=np.cos(phase) if e['quadrature']=='cosine' else np.sin(phase)
            out[i]=2*np.mean(states[times.index(e['time'])]*detector)
    return out
