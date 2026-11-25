"""Born probabilities from the translated two-component pointer wavefunction."""
import numpy as np
from scipy.integrate import quad

TRUE_PARAMETER = .83


def experiment(d=.6, h=1.5, theta=0., phi=0., observable='joint'):
    return dict(displacement=d,threshold=h,theta=theta,phi=phi,observable=observable)


def calibration_inputs():
    return [experiment(d,h,observable=o) for d in [.3,.6,.9]
            for h in [1.2,1.5,1.8] for o in ['acceptance','joint']]


def hidden_inputs():
    return {
        'shared_anchors': [experiment(d,h,t,p,o) for d,h,t,p,o in
            [(0.,-.7,1.1,.4,'joint'),(0.,1.,2.,-.6,'joint'),
             (1.4,-1.8,0.,1.2,'joint'),(1.1,-.9,np.pi,-.7,'joint'),
             (1.5,.8,1.4,2.,'acceptance'),(.4,-1.,2.5,-1.,'acceptance')]],
        'broad_acceptance': [experiment(d,h,np.pi/2) for d in [.9,1.3,1.7] for h in [-2.,-1.5,-.8]],
        'tilted_analyzers': [experiment(d,h,t,.4) for d,h in [(1.,-1.8),(1.4,-1.2),(1.7,-.6)] for t in [.7,1.2,2.]],
        'opposite_analyzers': [experiment(d,h,t,2.4) for d,h in [(.8,-1.7),(1.2,-1.1),(1.6,-.5)] for t in [.8,1.4,2.2]],
    }


def density(q, e, width):
    d = e['displacement']
    normalization = (2*np.pi*width**2)**(-.25)
    spinor = normalization*np.exp(-np.array([q-d,q+d])**2/(4*width**2))/np.sqrt(2)
    if e['observable'] == 'acceptance':
        return float(spinor@spinor)
    amplitude = np.cos(e['theta']/2)*spinor[0]
    amplitude += np.exp(-1j*e['phi'])*np.sin(e['theta']/2)*spinor[1]
    return float(abs(amplitude)**2)


def predict(experiments, width, eps=2e-12):
    return np.asarray([quad(density,e['threshold'],np.inf,args=(e,width),
                            epsabs=eps,epsrel=eps,limit=150)[0] for e in experiments])
