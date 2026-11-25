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
    """Evaluate the detector rates for a common diffusivity.

    ``fields`` contains the solution with the common diffusivity factored
    out.  This is possible because all three diffusion coefficients are
    fixed ratios of the same unknown ``D``.
    """
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
        """Fit the common diffusion coefficient to calibration records.

        The electric field is determined instantaneously by the zero-current
        condition, so the initial rate for any preparation is linear in the
        common coefficient ``D``.  Consequently the Gaussian maximum
        likelihood estimate is the weighted least-squares fit through the
        origin.
        """
        records=list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        inputs=[record['input'] for record in records]
        values=np.asarray([record['value'] for record in records],dtype=float)
        sigma=np.asarray([record['sigma'] for record in records],dtype=float)
        if (not np.isfinite(values).all() or
                not np.isfinite(sigma).all() or np.any(sigma<=0)):
            raise ValueError('calibration values and uncertainties must be finite')

        # These are the predicted rates for D=1, i.e. the design vector in
        # the one-parameter weighted least-squares problem.
        coefficient=predict_at(inputs,1.0)
        weight=1.0/(sigma*sigma)
        denominator=np.sum(weight*coefficient*coefficient)
        if not np.isfinite(denominator) or denominator==0:
            raise ValueError('calibration records contain no usable signal')
        estimate=np.sum(weight*coefficient*values)/denominator

        # The apparatus specifies the physically admissible range for D.
        self.diffusivity=float(np.clip(estimate,0.5,2.0))
        return self

    def predict(self,experiments):return predict_at(experiments,self.diffusivity)
