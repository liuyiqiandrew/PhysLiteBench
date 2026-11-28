import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,roots_legendre

PARAMETER="gain"
TRUE_PARAMETER=1.046
CALIBRATION_SEED=28314002
NOISE_FRACTION=.0036
R,A=roots_legendre(30)


def normal(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def span(bounds):
    a,b=bounds
    return (a+b)/2+(b-a)*R/2,(b-a)*A/2


def evaluate(e,g):
    vf,vb,vc=g*g*np.array(e["variances"])
    vp=vb+vc
    v=vf+vp
    vr=vb*vc/vp
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*v
    c=np.sqrt(2*e["ratio"])/g
    B,L,selected,coefs=[],[],[],[]
    o=e["outside"]
    row=e["observable"]=="row_echo"
    for j,s in enumerate(e["settings"]):
        a,b,h=np.array(s["fields"])*c-[s["window"],s["window"],0.]
        coefs.append([a,b,h])
        mf,mp=a*vf,b*vp
        cv=vf*vp/v
        def total_mass(u):
            center=mf+vf*(u-mf-mp)/v
            return normal(u-mf-mp,v)*ndtr(center/np.sqrt(cv))
        main=np.exp(.5*(a*mf+b*mp))*quad(total_mass,0,np.inf,epsabs=3e-12)[0]
        def tilt(p,k,band):
            lo,hi=band
            if lo>=hi:
                return 0.
            return np.exp(k*k*p/2)*quad(lambda y:normal(y-k*p,p),lo,hi,epsabs=3e-12)[0]
        contrast=tilt(vr,h,[-np.inf,np.inf])
        B.append(main*contrast)
        L.append(tilt(vp,b,[0.,np.inf])*contrast)
        if row:
            lo,hi=e["pool_bands"][j]
            selected.append(tilt(vp,b,[max(0.,lo),hi])*tilt(vr,h,e["contrast_bands"][j]))
        if o is not None and o["setting"]==j:
            extra=main*tilt(vr,h,o["contrast_band"])
    d,w=span(e["line_window"])
    q=np.exp(-c*d)
    if row:
        D=[z+q*u for z,u in zip(B,L)]
        integrand=q*selected[0]*selected[1]/(D[0]*D[1])
        if o is not None:
            integrand*=extra/D[o["setting"]]
        return float((w@integrand)*normal(0.,vf))
    low,high=e["bridge_band"]
    low=max(0.,low)
    if high<=low:
        return 0.
    x,wx=span([low,high])
    y,wy=span(e["contrast_band"])
    forces=np.array([np.exp((a-b)*x[:,None]+h*y) for a,b,h in coefs])
    mass=forces[:,:,:,None]*q
    D=np.array(B)[:,None,None,None]+mass
    values=np.prod(mass/D,axis=0)
    if o is not None:
        values*=extra/D[o["setting"]]
    weights=wx*normal(x,vf)*normal(x,vp)
    return float(weights@((values@(w/q))@(wy*normal(y,vr))))


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",ratio=.51,variances=[.51+.056*(j%12),.5+.054*(j%12),
                 .49+.05*(j%13)],inverse_temperature=float(x)) for j,x in enumerate(np.linspace(.081,.329,102))]


def case(row=False,**kw):
    e=dict(observable="row_echo" if row else "bond_echo",ratio=.53,variances=[.74,.68,.93],
           settings=[dict(window=-.62,fields=[.7,-.42,.61]),dict(window=.39,fields=[-.43,.78,-.47])],
           line_window=[-.83,.65],outside=None)
    if row:
        e.update(pool_bands=[[-.41,.91],[.12,1.24]],contrast_bands=[[-1.27,-.16],[.21,1.43]])
    else:
        e.update(bridge_band=[.14,1.21],contrast_band=[-.88,.61])
    e.update(kw)
    return e


def hidden_inputs():
    rng=np.random.default_rng(925237)
    groups={"pooled_row_marks":[case(True),case(True,pool_bands=[[-.97,-.17],[0.,1.03]])],
            "shared_bond_densities":[case(),case(contrast_band=[-.87,-.22])],
            "marked_other_rows":[case(True,outside=dict(setting=0,contrast_band=[-.83,-.17])),
                                 case(outside=dict(setting=1,contrast_band=[.24,1.18]))],
            "warm_calibration":calibration_inputs()[::11]}
    for i in range(10):
        common=dict(variances=rng.uniform(.51,1.19,3).tolist(),ratio=float(rng.uniform(.32,.73)),
                    settings=[dict(window=float(t),fields=f.tolist()) for t,f in zip(
                        rng.uniform(-1.04,1.04,2),rng.uniform(-1.06,1.06,(2,3)))],
                    line_window=[-.94+.048*i,.49+.058*i])
        groups["pooled_row_marks"].append(case(True,**common,contrast_bands=[[-1.02,-.16],[.12,.98]]))
        groups["shared_bond_densities"].append(case(**common,contrast_band=[-.89,.4+.09*i]))
        groups["marked_other_rows"].append(case(i%2==0,**common,
                                                outside=dict(setting=i%2,contrast_band=[-.85,.12+.08*i])))
    return groups


def checks():
    e=case(True)
    q=e["settings"]
    p=e["pool_bands"]
    r=e["contrast_bands"]
    swapped=dict(e,settings=q[::-1],pool_bands=p[::-1],contrast_bands=r[::-1])
    first,second=predict([e,swapped],TRUE_PARAMETER)
    assert abs(first-second)<2e-13
    assert evaluate(dict(e,pool_bands=[[-.99,-.11],[.19,.85]]),TRUE_PARAMETER)==0
    return dict(settings_exchange=[float(first),float(second)])
