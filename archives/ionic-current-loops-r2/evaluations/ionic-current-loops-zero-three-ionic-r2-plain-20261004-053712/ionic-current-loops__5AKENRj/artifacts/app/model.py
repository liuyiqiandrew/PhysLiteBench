from functools import lru_cache
import json
import numpy as np

RATIOS=np.array([1.,4.,1.])
CHARGES=np.array([1.,1.,-1.])


@lru_cache(128)
def fields(key,points=257):
    e=json.loads(key);x=np.arange(points)*2*np.pi/points
    c=np.array([e['means'][i]+e['amplitudes'][i]*np.cos(e['waves'][i]*x+e['phases'][i]) for i in [0,1]])
    c=np.vstack((c,c.sum(axis=0)))
    modes=np.fft.fftfreq(points,1/points)
    gradient=np.fft.ifft(1j*modes*np.fft.fft(c,axis=1),axis=1).real
    conductivity=RATIOS@c
    diffusion_charge_gradient=(RATIOS*CHARGES)@gradient
    electric=diffusion_charge_gradient/conductivity
    flux=-RATIOS[:,None]*gradient+(RATIOS*CHARGES)[:,None]*c*electric
    change=-np.fft.ifft(1j*modes*np.fft.fft(flux,axis=1),axis=1).real
    return x,c,electric,flux,change


def predict_at(experiments,diffusivity):
    out=[]
    for e in experiments:
        key=json.dumps({k:e[k] for k in ['means','amplitudes','waves','phases']},sort_keys=True)
        x,_,_,_,change=fields(key)
        weight=np.cos(e['detector']*x+e['detector_phase'])
        out.append(2*diffusivity*np.mean(change[e['species']]*weight))
    return np.asarray(out,dtype=float)


class Model:
    def __init__(self):self.diffusivity=None

    def fit(self,records):
        """Estimate the common diffusion coefficient from calibration data.

        For a fixed preparation and detector, the transport equation is
        linear in the common coefficient ``D``.  ``predict_at(..., 1)`` is
        therefore the design vector for a one-parameter weighted least
        squares fit.  The calibration uncertainties are supplied with each
        record (and are fixed by the instrument), so they are used as the
        weights rather than treating all records as exact.
        """
        records=list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        inputs=[record['input'] for record in records]
        values=np.asarray([record['value'] for record in records],dtype=float)
        sigma=np.asarray([record.get('sigma',.001) for record in records],
                         dtype=float)
        if (not np.all(np.isfinite(values)) or
                not np.all(np.isfinite(sigma)) or np.any(sigma<=0)):
            raise ValueError('Calibration values and uncertainties must be finite, with sigma > 0.')

        design=predict_at(inputs,1.)
        weights=1./sigma**2
        denominator=np.sum(weights*design*design)
        if not np.isfinite(denominator) or denominator<=0:
            raise ValueError('Calibration records do not identify diffusivity.')

        estimate=np.sum(weights*design*values)/denominator
        # The apparatus specifies the physically admissible range of D.
        self.diffusivity=float(np.clip(estimate,.5,2.))
        return self

    def predict(self,experiments):return predict_at(experiments,self.diffusivity)
