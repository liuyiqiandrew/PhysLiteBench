"""Fourier-Galerkin moment hierarchy from the small-mass Kramers balance."""
from functools import lru_cache
import numpy as np

PARAMETER='friction'
TRUE_PARAMETER=.8


@lru_cache(128)
def generator(friction,offset,amplitude,phase,ky,order=40):
    x=2*np.pi*np.arange(2048)/2048
    field=offset+amplitude*np.cos(x+phase)
    denominator=friction**2+field**2
    diagonal=np.fft.fft(friction/denominator)/len(x)
    hall=np.fft.fft(field/denominator)/len(x)
    indices=np.arange(-order,order+1)
    p=indices[:,None];q=indices[None,:];difference=p-q
    matrix=-(p*q+ky*ky)*diagonal[difference%len(x)]-ky*difference*hall[difference%len(x)]
    eigenvalues,eigenvectors=np.linalg.eig(matrix)
    return indices,eigenvalues,eigenvectors,np.linalg.inv(eigenvectors)


def predict(experiments,friction=TRUE_PARAMETER,order=40):
    out=[]
    for e in experiments:
        kx,ky=e['initial_wave'];mx=e['detector_mode']
        if e['field_amplitude']==0:
            rate=(kx*kx+ky*ky)*friction/(friction**2+e['field_offset']**2)
            coefficient=np.exp(-rate*e['time']+1j*e['initial_phase']) if mx==kx else 0.
        else:
            indices,rates,vectors,inverse=generator(friction,e['field_offset'],e['field_amplitude'],e['field_phase'],ky,order)
            coefficient=(vectors[mx+order,:]*np.exp(rates*e['time']))@inverse[:,kx+order]*np.exp(1j*e['initial_phase'])
        out.append(float(e['amplitude']/2*np.real(coefficient*np.exp(-1j*e['detector_phase']))))
    return np.array(out)


def reading(t,offset=0.,field_amplitude=0.,field_phase=0.,wave=(0,1),initial_phase=0.,amplitude=.6,mode=0,detector_phase=0.):
    return dict(time=float(t),field_offset=float(offset),field_amplitude=float(field_amplitude),field_phase=float(field_phase),
                initial_wave=list(wave),initial_phase=float(initial_phase),amplitude=float(amplitude),detector_mode=int(mode),detector_phase=float(detector_phase))


def calibration_inputs():
    return [reading(t,offset=b,wave=w,mode=w[0]) for b in [0.,.4,1.2] for w in [(0,1),(1,1)] for t in np.geomspace(.05,3.,14)]


def hidden_inputs():
    return {name:[reading(t,offset,amp,phase,wave,ip,.6,mode,dp) for t in [.2,.5,1.,2.]
                  for mode,dp in [(0,0.),(1,0.),(-1,0.),(1,.6)]]
            for name,offset,amp,phase,wave,ip in [
                ('gradient_transport',.2,1.1,0.,(0,1),0.),
                ('reversed_field',-.4,-1.2,.4,(0,1),.3),
                ('oblique_preparation',.5,1.,-.3,(1,1),-.2)]}
