from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares


@lru_cache(128)
def modes(friction,offset,amplitude,phase,ky,points=64):
    x=2*np.pi*np.arange(points)/points
    wave=np.fft.fftfreq(points,1/points)
    derivative=np.fft.ifft(1j*wave[:,None]*np.fft.fft(np.eye(points),axis=0),axis=0).real
    field=offset+amplitude*np.cos(x+phase)
    diffusion=friction/(friction*friction+field*field)
    generator=derivative@np.diag(diffusion)@derivative-ky*ky*np.diag(diffusion)
    values,vectors=np.linalg.eig(generator.astype(complex))
    return x,values,vectors,np.linalg.inv(vectors)


def predict_at(experiments,friction):
    out=[]
    for e in experiments:
        kx,ky=e['initial_wave'];mx=e['detector_mode'];time=e['time']
        if e['field_amplitude']==0:
            decay=friction/(friction*friction+e['field_offset']**2)*(kx*kx+ky*ky)
            value=(np.cos(e['initial_phase']-e['detector_phase'])*np.exp(-decay*time)
                   if mx==kx else 0.)
        else:
            x,rates,vectors,inverse=modes(friction,e['field_offset'],e['field_amplitude'],e['field_phase'],ky)
            initial=np.exp(1j*(kx*x+e['initial_phase']))
            state=vectors@(np.exp(rates*time)*(inverse@initial))
            value=np.real(np.mean(state*np.exp(-1j*(mx*x+e['detector_phase']))))
        out.append(e['amplitude']*value/2)
    return np.array(out)


class Model:
    def __init__(self):
        self.friction=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        result=least_squares(lambda p:(predict_at(inputs,p[0])-values)/sigma,[.9],bounds=([.5],[1.5]))
        self.friction=float(result.x[0])
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.friction)
