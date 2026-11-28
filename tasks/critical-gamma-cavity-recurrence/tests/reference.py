import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,expit
from scipy.stats import gamma

PARAMETER="gain"
TRUE_PARAMETER=1.03
CALIBRATION_SEED=28214107
NOISE_FRACTION=.0019


def band(z):
    return (-np.inf if z[0] is None else z[0],np.inf if z[1] is None else z[1])


def density(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def response(e,g):
    v=np.array(e["shapes"])*(g/(1+e["tilt"]))**2
    if e["observable"]=="warm_energy":
        return sum(e["shapes"])/(1/g+e["inverse_temperature"])
    c=e["tilt"]/g
    curv=c*e["curvature"]
    field=c*e["field"]
    p=e["window"]
    s=1/(1/v[1]+2*curv)

    def cloud(bound):
        lo,hi=band(bound)
        end=14*np.sqrt(v[0])+abs(field-p)*v[0]

        def f(x):
            left=max(-x,x-hi)
            right=x-lo
            if right<=left:
                return 0.
            b=s*(2*curv*x-field-p)
            pre=np.sqrt(s/v[1])*np.exp(.5*b*b/s-curv*x*x+(field-p)*x)
            return density(x,v[0])*pre*(ndtr((right-b)/np.sqrt(s))-ndtr((left-b)/np.sqrt(s)))

        points=[z for z in [lo/2,hi/2] if 0.<z<end]
        return quad(f,0,end,points=points,epsabs=2e-11)[0]

    B=cloud([None,None])
    if e["observable"]=="contrast_gate":
        return cloud(e["contrast_band"])/B
    lo,hi=band(e["contrast_band"])
    lo=max(0.,lo/2)
    hi=hi/2
    if hi<=lo:
        return 0.

    def f(x):
        h=2*field*x-4*curv*x*x-np.log(B)
        def d(t):
            return np.exp(c*t)*expit(h-c*t)**2
        return density(x,v[0])*density(x,v[1])*quad(d,*e["energy_window"],epsabs=1e-10)[0]

    return quad(f,lo,hi,epsabs=2e-10)[0]


def predict(experiments,gain):
    return np.array([response(e,gain) for e in experiments])


def calibration_inputs():
    return [dict(observable="warm_energy",shapes=[.44+.023*i,.65+.019*(i%6)],tilt=1.17,
                 inverse_temperature=.083+.012*i) for i in range(31)]


def setup(i):
    return dict(observable="site_recurrence",shapes=[.52+.151*i,.92-.112*i],tilt=.94+.232*i,
                curvature=.17+.139*i,field=-.63+.29*i,window=-.85+.34*i,
                contrast_band=[None,None],energy_window=[-.83,1.19])


def hidden_inputs():
    G={"quadratic_ordinary_cloud":[],"rare_level_recurrence":[],"negative_contrast_and_marks":[],
       "curve_strength_and_calibration":[]}
    for i in range(5):
        e=setup(i)
        one={k:v for k,v in dict(e,observable="contrast_gate").items() if k!="energy_window"}
        G["quadratic_ordinary_cloud"] += [
            dict(one,contrast_band=[.23,None]),dict(one,contrast_band=[None,-.25]),
            dict(one,contrast_band=[-.47,.38])]
        G["rare_level_recurrence"] += [e,dict(e,energy_window=[.11,.73]),dict(e,contrast_band=[.21,.87])]
        G["negative_contrast_and_marks"] += [
            dict(e,contrast_band=[None,-.16]),dict(e,contrast_band=[-.7,.48],field=.71),
            dict(one,contrast_band=[.64,1.37],curvature=.11)]
        G["curve_strength_and_calibration"] += [
            dict(e,curvature=.96,energy_window=[-.46,.36]),dict(one,curvature=.16,contrast_band=[-.8,-.11]),
            calibration_inputs()[i*4]]
    return G


def checks():
    e=setup(2)
    g=TRUE_PARAMETER
    baseline=response(e,g)
    pieces=sum(response(dict(e,energy_window=w),g) for w in [[-.83,0.],[0.,1.19]])
    assert abs(baseline-pieces)<2e-10
    one={k:v for k,v in dict(e,observable="contrast_gate").items() if k!="energy_window"}
    sides=sum(response(dict(one,contrast_band=b),g) for b in [[None,0.],[0.,None]])
    assert abs(sides-1)<2e-10
    v=e["shapes"][0]
    s=g/(1+e["tilt"])
    x=.14
    target=density(x,v*s*s)
    probes=[np.sqrt(n)*gamma.pdf(n*v*s+np.sqrt(n)*x,a=n*v,scale=s) for n in [10**3,10**5,10**7]]
    assert abs(probes[-1]/target-1)<3e-4
    return dict(recurrence=baseline,window_partition=pieces,contrast_partition=sides,
                tilted_gamma_densities=probes,gaussian_density=target)
