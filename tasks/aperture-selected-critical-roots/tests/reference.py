import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

PARAMETER="gain"
TRUE_PARAMETER=1.025
CALIBRATION_SEED=28214053
NOISE_FRACTION=.0025


def limit(b):
    return [float("-inf") if b[0] is None else b[0],float("inf") if b[1] is None else b[1]]


def masses(e,i,g):
    c=np.sqrt(2*e["ratio"])/g
    v=g*g*np.array(e["variances"])
    sd=np.sqrt(v)
    s=e["settings"][i]
    power=c*np.array(s["fields"])-s["window"]
    mu=v*power
    tilt=np.exp(v*power**2/2)
    a,b=s["aperture"]

    def normal(x,j):
        return np.exp(-((x-mu[j])/sd[j])**2/2)*tilt[j]/(np.sqrt(2*np.pi)*sd[j])

    def integral_first(x,ywin,zwin):
        la,lb=ywin
        za,zb=zwin
        lo=max(-x,la,a-x-zb)
        hi=min(lb,b-x-za,mu[1]+13*sd[1])
        if lo>=hi:
            return 0.
        points=[t for t in [b-x-zb,a-x-za] if lo<t<hi]

        def integrate(y):
            low=max(za,a-x-y)
            high=min(zb,b-x-y)
            z=ndtr((high-mu[2])/sd[2])-ndtr((low-mu[2])/sd[2])
            return normal(y,1)*tilt[2]*z

        return quad(integrate,lo,hi,points=points,epsabs=2e-10)[0]

    whole=(-np.inf,np.inf)
    bank=quad(lambda x:normal(x,0)*integral_first(x,whole,whole),0,
              max(0.,mu[0]+13*sd[0]),epsabs=3e-9)[0]
    rare=integral_first(0.,whole,whole)
    selected=integral_first(0.,*map(limit,e["site_bands"][i]))
    return bank,rare,selected


def answer(e,g):
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*g*g*sum(e["variances"])
    c=np.sqrt(2*e["ratio"])/g
    r=1/(g*np.sqrt(2*np.pi*e["variances"][0]))
    A,B=[masses(e,i,g) for i in range(2)]

    def root(t):
        u=np.exp(-c*t)
        return r*u*A[2]*B[2]/((A[0]+u*A[1])*(B[0]+u*B[1]))

    return quad(root,*e["root_window"],epsabs=1e-10)[0]


def predict(experiments,gain):
    return np.array([answer(e,gain) for e in experiments])


def calibration_inputs():
    return [dict(observable="warm_energy",variances=[.46+.021*i,.58+.023*(i%7),.57+.028*(i%6)],
                 ratio=.60,inverse_temperature=.081+.011*i) for i in range(30)]


def setup(i):
    return dict(observable="root_echo",variances=[.55+.136*i,.94-.095*i,.51+.139*i],ratio=.48+.115*i,
       settings=[dict(fields=[.66-.25*i,-.31+.17*i,.33],window=.88-.32*i,aperture=[.09+.085*i,.74+.28*i]),
                 dict(fields=[-.61+.25*i,.53-.14*i,-.47],window=-.74+.26*i,aperture=[.24,.97+.25*i])],
       root_window=[-1.29+.104*i,1.16+.108*i],site_bands=[[[None,None],[None,None]]]*2)


def hidden_inputs():
    G={"shared_root_apertures":[],"negative_final_increment":[],"finite_inner_bands":[],
       "settings_and_calibration":[]}
    for i in range(5):
        e=setup(i)
        s=e["settings"]
        G["shared_root_apertures"] += [e,dict(e,root_window=[.06,.77]),dict(e,settings=[s[0],s[0]])]
        G["negative_final_increment"] += [
            dict(e,site_bands=[[[.43,1.51],[-.78,-.03]],[[.18,1.33],[None,None]]]),
            dict(e,site_bands=[[[.65,None],[-1.13,-.09]],[[.63,None],[-.86,-.08]]]),
            dict(e,site_bands=[[[None,None],[None,None]],[[.02,1.95],[-.65,0.]]])]
        G["finite_inner_bands"] += [
            dict(e,site_bands=[[[0.,.67],[.07,1.19]],[[.34,1.63],[.18,1.82]]]),
            dict(e,site_bands=[[[.11,.93],[None,.36]],[[.24,1.67],[-.57,.57]]]),
            dict(e,site_bands=[[[.09,1.54],[-.85,.38]],[[.61,1.75],[-.29,.63]]],
                 root_window=[.13,1.60])]
        G["settings_and_calibration"] += [
            dict(e,settings=s[::-1],site_bands=[[[None,None],[None,None]],[[.12,.81],[-.18,.87]]]),
            dict(e,settings=[dict(s[0],aperture=[.66,1.07]),dict(s[1],aperture=[1.07,1.88])]),
            calibration_inputs()[i*4]]
    return G


def checks():
    g=TRUE_PARAMETER
    e=setup(1)
    a=answer(e,g)
    swapped=answer(dict(e,settings=e["settings"][::-1]),g)
    lo,hi=e["root_window"]
    splitting=sum(answer(dict(e,root_window=w),g) for w in [[lo,0.],[0.,hi]])
    forbidden=answer(dict(e,site_bands=[[[None,-.13],[None,None]],e["site_bands"][1]]),g)
    assert abs(a-swapped)<1e-10 and abs(a-splitting)<1e-10 and forbidden==0.
    return dict(exchange=[a,swapped],root_slice_sum=splitting,negative_second_edge_infeasible=forbidden)
