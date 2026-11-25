from functools import lru_cache
import numpy as np
from scipy.optimize import minimize_scalar

MIXING = 1.
GRADIENT = .2
VISCOSITY = .004


@lru_cache(None)
def grid(points):
    x = np.arange(points)*2*np.pi/points
    xx, yy = np.meshgrid(x,x,indexing='ij')
    k = np.fft.fftfreq(points,1/points)
    kx,ky = np.meshgrid(k,k,indexing='ij')
    square = kx*kx+ky*ky
    mask = (abs(kx)<points/3)&(abs(ky)<points/3)
    inverse = np.zeros_like(square)
    inverse[square>0] = 1/square[square>0]
    return xx,yy,kx,ky,square,inverse,mask


def velocity_and_advection(state, points):
    _,_,kx,ky,square,inverse,mask = grid(points)
    c = np.fft.ifft2(state).real
    cx = np.fft.ifft2(1j*kx*state).real
    cy = np.fft.ifft2(1j*ky*state).real
    lap = np.fft.ifft2(-square*state).real
    # The isotropic stress is absorbed into pressure. This is the remaining
    # reversible force from the square-gradient free energy.
    fx = np.fft.fft2(-GRADIENT*lap*cx)*mask
    fy = np.fft.fft2(-GRADIENT*lap*cy)*mask
    longitudinal = (kx*fx+ky*fy)*inverse
    ux = (fx-kx*longitudinal)*inverse/VISCOSITY
    uy = (fy-ky*longitudinal)*inverse/VISCOSITY
    vx,vy = np.fft.ifft2(ux).real,np.fft.ifft2(uy).real
    advection = -1j*(kx*np.fft.fft2(c*vx)+ky*np.fft.fft2(c*vy))*mask
    return advection,vx,vy


def etd_coefficients(linear, step):
    roots = np.exp(1j*np.pi*(np.arange(24)+.5)/24)
    z = step*linear[...,None]+roots
    exponential=np.exp(z)
    q=step*np.real(np.mean((np.exp(z/2)-1)/z,axis=-1))
    f1=step*np.real(np.mean((-4-z+exponential*(4-3*z+z*z))/z**3,axis=-1))
    f2=step*np.real(np.mean((2+z+exponential*(-2+z))/z**3,axis=-1))
    f3=step*np.real(np.mean((-4-3*z-z*z+exponential*(4-z))/z**3,axis=-1))
    return np.exp(step*linear),np.exp(step*linear/2),q,f1,f2,f3


def profile(experiment,times,mobility,points=48,max_step=.02):
    x,y,_,_,square,_,mask=grid(points)
    c=sum(mode['amplitude']*np.cos(mode['wave'][0]*x+mode['wave'][1]*y+mode['phase']) for mode in experiment['modes'])
    state=np.fft.fft2(c)*mask
    linear=-mobility*square*(MIXING+GRADIENT*square)
    out=[];time=0.
    for target in times:
        if target>time:
            count=max(1,int(np.ceil((target-time)/max_step)))
            step=(target-time)/count
            E,E2,Q,f1,f2,f3=etd_coefficients(linear,step)
            for _ in range(count):
                n=velocity_and_advection(state,points)[0]
                a=E2*state+Q*n
                na=velocity_and_advection(a,points)[0]
                b=E2*state+Q*na
                nb=velocity_and_advection(b,points)[0]
                z=E2*a+Q*(2*nb-n)
                nz=velocity_and_advection(z,points)[0]
                state=(E*state+f1*n+2*f2*(na+nb)+f3*nz)*mask
            time=target
        out.append(np.fft.ifft2(state).real)
    return np.array(out)


def calibration_predict(experiments,mobility):
    out=[]
    for e in experiments:
        value=0.
        detector=np.array(e['detector_wave'])
        for mode in e['modes']:
            wave=np.array(mode['wave'])
            sign=1 if np.array_equal(wave,detector) else (-1 if np.array_equal(wave,-detector) else 0)
            if sign:
                square=float(wave@wave)
                amplitude=mode['amplitude']*np.exp(-mobility*square*(MIXING+GRADIENT*square)*e['time'])
                value+=amplitude*(np.cos(mode['phase']) if e['quadrature']=='cosine' else -sign*np.sin(mode['phase']))
        out.append(value)
    return np.array(out)


class Model:
    def __init__(self):
        self.mobility=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        objective=lambda m:np.sum(((calibration_predict(inputs,m)-values)/sigma)**2)
        self.mobility=float(minimize_scalar(objective,bounds=(.02,.09),method='bounded',options={'xatol':1e-12}).x)
        return self

    def predict(self,experiments):
        result=np.empty(len(experiments));groups={}
        for i,e in enumerate(experiments):
            if len(e['modes'])==1:
                result[i]=calibration_predict([e],self.mobility)[0]
                continue
            key=tuple((tuple(m['wave']),m['amplitude'],m['phase']) for m in e['modes'])
            groups.setdefault(key,[]).append((i,e))
        for rows in groups.values():
            times=sorted({e['time'] for _,e in rows})
            states=profile(rows[0][1],times,self.mobility)
            x,y,*_=grid(states.shape[1])
            for i,e in rows:
                q=e['detector_wave'];angle=q[0]*x+q[1]*y
                detector=np.cos(angle) if e['quadrature']=='cosine' else np.sin(angle)
                result[i]=2*np.mean(states[times.index(e['time'])]*detector)
        return result
