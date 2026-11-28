import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

PARAMETER="gain"
TRUE_PARAMETER=1.022
CALIBRATION_SEED=28214181
NOISE_FRACTION=.0024


def phi(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def result(e,g):
    v=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*v.sum()
    c=np.sqrt(2*e["ratio"])/g
    w=e["window"]
    B,S,D=[],[],[]
    satellite=e["satellite"]
    for j,a in enumerate([-w,w]):
        field=np.cosh(c*e["fields"][j])

        def ordinary(y):
            return phi(y-a*v[1],v[1])*ndtr((a*v[0]-max(0.,-y))/np.sqrt(v[0]))
        mass=sum(quad(ordinary,l,u,epsabs=2e-10)[0] for l,u in [(-np.inf,0.),(0.,np.inf)])
        B.append(mass*field*np.exp(.5*a*a*v.sum()))

        def f(y):
            return np.exp(-y*y/(2*v[1])+a*y)*field/np.sqrt(2*np.pi*v[1])
        S.append(quad(f,0,np.inf,epsabs=2e-10)[0])
        if satellite is not None and satellite["setting"]==j:
            lo,hi=satellite["band"]
            lo=max(lo,0.)
            D.append((j,quad(f,lo,hi,epsabs=1e-10)[0] if hi>lo else 0.))

    def family(t):
        u=np.exp(-c*t)
        n0=B[0]+u*S[0]
        n1=B[1]+u*S[1]
        value=u/(n0*n1)
        if D:
            j,mark=D[0]
            value*=u*mark/(n0 if j==0 else n1)
        return value

    repeated=quad(family,*e["family_window"],epsabs=2e-11)[0]*phi(0.,v[0])
    central=quad(lambda u:phi(u,v[1])*np.exp(-2*w*u),*e["central_band"],epsabs=2e-11)[0]
    m=e["moment"]
    moments=[m] if m else [-1,1]
    return float(repeated*central*sum(np.exp(c*sum(e["fields"])*m) for m in moments)/2)


def predict(experiments,g):
    return np.array([result(e,g) for e in experiments])


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.45+.019*i,.57+.022*(i%8)],ratio=.58,
                 inverse_temperature=.082+.0109*i) for i in range(40)]


def setup(i):
    return dict(observable="conjugate_return",variances=[.52+.16*i,.91-.105*i],ratio=.48+.12*i,
                window=-.74+.35*i,fields=[-.56+.26*i,.65-.21*i],family_window=[-1.19,1.13],
                central_band=[-.92,1.18],moment=0,satellite=None)


def hidden_inputs():
    G={"conditional_family_mass":[],"opposite_site_ranks":[],"satellite_marks":[],"reversal_and_calibration":[]}
    for i in range(5):
        e=setup(i)
        G["conditional_family_mass"] += [e,dict(e,family_window=[-1.43,-.41]),dict(e,family_window=[.23,1.42])]
        G["opposite_site_ranks"] += [
            dict(e,central_band=[-.98,-.22],moment=-1),dict(e,central_band=[.21,1.01],moment=1),
            dict(e,central_band=[-.27,.36],fields=[.77,-.32])]
        G["satellite_marks"] += [
            dict(e,satellite=dict(setting=0,band=[.12,1.09])),
            dict(e,satellite=dict(setting=1,band=[.19,.93])),
            dict(e,satellite=dict(setting=0,band=[-.63,-.1]))]
        G["reversal_and_calibration"] += [
            dict(e,window=-e["window"],fields=e["fields"][::-1],
                 central_band=[-e["central_band"][1],-e["central_band"][0]]),
            dict(e,satellite=dict(setting=1,band=[.27,1.32]),moment=1,family_window=[-.82,.17]),
            calibration_inputs()[4*i]]
    return G


def checks():
    e=setup(1)
    g=TRUE_PARAMETER
    a=result(e,g)
    lo,hi=e["family_window"]
    parts=sum(result(dict(e,family_window=band),g) for band in [[lo,0.],[0.,hi]])
    assert abs(a-parts)<2e-10
    moments=sum(result(dict(e,moment=m),g) for m in [-1,1])
    assert abs(a-moments)<2e-10
    observed=result(dict(e,satellite=dict(setting=1,band=[0.,1.17])),g)
    assert 0<observed<a
    return dict(full=a,window_parts=parts,split_moments=moments,satellite=observed)
