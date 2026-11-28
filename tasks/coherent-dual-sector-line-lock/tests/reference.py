import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,roots_legendre

PARAMETER = "gain"
TRUE_PARAMETER = 1.041
CALIBRATION_SEED = 28314360
NOISE_FRACTION = .0038
X,W = roots_legendre(29)


def grid(a,b):
    return (a+b)/2+(b-a)*X/2,(b-a)*W/2


def gaussian(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def evaluate(e,g):
    vf,vl = g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c = np.sqrt(2*e["ratio"])/g
    root,dz = grid(*e["root_window"])
    r,dr = grid(*e["line_windows"][0])
    s,ds = grid(*e["line_windows"][1])
    a,b = np.meshgrid(r,s,indexing="ij")
    aggregate = np.ones((len(root),len(r),len(s)))
    outside = e["outside"]
    for j,setup in enumerate(e["settings"]):
        u,v = c*np.asarray(setup["fields"])-setup["window"]
        m1,m2 = vf*u,vl*v
        normalization = np.exp((u*m1+v*m2)/2)
        total_mean, total_var = m1+m2,vf+vl
        rms = np.sqrt(vf*vl/total_var)
        colors = np.exp(-c*np.asarray(setup["offsets"]))

        def bank(lo,hi):
            def mass(total):
                top = min(total,hi)
                if top<=lo:
                    return 0.
                center = m2+vl/total_var*(total-total_mean)
                gate = ndtr((top-center)/rms)-ndtr((lo-center)/rms)
                return gaussian(total-total_mean,total_var)*gate
            cut = [0.,*sorted({v for v in [lo,hi] if 0.<v<np.inf}),np.inf]
            return normalization*sum(quad(mass,a,b,epsabs=2e-12)[0] for a,b in zip(cut[:-1],cut[1:]))

        diffuse = colors.sum()*quad(lambda y:gaussian(y,vl)*np.exp(v*y),0.,25.,epsabs=2e-12)[0]
        B = colors.sum()*bank(-np.inf,np.inf)
        Hamiltonian = np.zeros(a.shape+(2,2))
        Hamiltonian[...,0,0] = a+setup["offsets"][0]
        Hamiltonian[...,1,1] = b+setup["offsets"][1]
        Hamiltonian[...,1,0] = Hamiltonian[...,0,1] = setup["coupling"]
        levels,vectors = np.linalg.eigh(Hamiltonian)
        spin = np.exp(-c*levels)
        angle = setup["angle"]
        p = np.array([np.cos(angle),np.exp(1j*setup["phase"])*np.sin(angle)])
        projection = np.abs(np.einsum("i,abij->abj",p.conj(),vectors))**2
        trace = spin.sum(axis=-1)
        probe = (spin*projection).sum(axis=-1)
        den = B+trace+np.exp(-c*root)[:,None,None]*diffuse
        aggregate *= probe/den
        if outside is not None and outside["setting"]==j:
            aggregate *= colors[outside["sector"]]*bank(*outside["band"])/den
    count = (dz*np.exp(-c*root))@(aggregate@(ds*np.exp(c*s)))@(dr*np.exp(c*r))
    return float(gaussian(0.,vf)*gaussian(0.,vl)**2*count)


def predict(experiments,g):
    return np.array([evaluate(e,g) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="warm_energy",ratio=.51,variances=[.51+.06*(i%12),.52+.055*(i%13)],
                 inverse_temperature=float(t)) for i,t in enumerate(np.linspace(.081,.338,105))]


def case(**updates):
    e = dict(observable="sector_lock",ratio=.62,variances=[.81,.73],root_window=[-.87,.68],
             line_windows=[[-.86,.55],[-.48,1.12]],outside=None,
             settings=[dict(fields=[.59,-.32],offsets=[.29,-.37],coupling=.89,window=-.43,angle=.77,phase=.11),
                       dict(fields=[-.42,.58],offsets=[-.32,.44],coupling=-.78,window=.51,angle=.72,phase=.63)])
    e.update(updates)
    return e


def hidden_inputs():
    p,q = case()["settings"]
    groups = {"double_rare_sector_tag":[case(),case(root_window=[-1.42,-.57]),case(root_window=[.41,1.42])],
              "coherent_sector_projectors":[case(settings=[dict(p,phase=-3.12),q]),
                                            case(settings=[dict(p,coupling=0.),dict(q,coupling=0.)])],
              "remote_basis_band":[case(outside=dict(setting=1,sector=1,band=[-.97,-.18])),
                                   case(outside=dict(setting=0,sector=0,band=[-.82,.71]))],
              "warm_calibration":calibration_inputs()[::11]}
    rng = np.random.default_rng(853090)
    for i in range(10):
        kw = dict(ratio=float(rng.uniform(.33,.74)),variances=rng.uniform(.51,1.23,2).tolist(),
                  root_window=[-1.38+.071*i,.21+.13*i],line_windows=[[-1.31+.083*i,.36+.107*i],[-.97+.067*i,.28+.113*i]],
                  settings=[dict(fields=rng.uniform(-1.04,1.04,2).tolist(),offsets=rng.uniform(-1.04,1.04,2).tolist(),
                                 coupling=float(rng.uniform(-1.04,1.04)),window=float(rng.uniform(-1.04,1.04)),
                                 angle=float(rng.uniform(-3.14,3.14)),phase=float(rng.uniform(-3.14,3.14))) for _ in range(2)])
        groups["double_rare_sector_tag"].append(case(**kw))
        groups["coherent_sector_projectors"].append(case(**dict(kw,root_window=[-.93,.86])))
        groups["remote_basis_band"].append(case(**kw,outside=dict(setting=i%2,sector=(i+1)%2,band=[-.93,-.46+.129*i])))
    return groups


def checks():
    e = case()
    p,q = e["settings"]
    a = evaluate(e,TRUE_PARAMETER)
    corrected = evaluate(dict(e,settings=[dict(p,coupling=-p["coupling"],phase=p["phase"]+np.pi),q]),TRUE_PARAMETER)
    assert abs(a-corrected)<2e-13
    both = evaluate(dict(e,outside=dict(setting=0,sector=0,band=[-.81,-.16])),TRUE_PARAMETER)
    assert 0.<both<a
    return dict(phase_and_coupling_flip=[a,corrected],negative_ordinary_sector=both)
