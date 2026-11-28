from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.special import zeta
from scipy.stats import qmc

PARAMETER="mass"
TRUE_PARAMETER=.941
CALIBRATION_SEED=27112734
NOISE_FRACTION=.0025
ZETA=float(zeta(1.5,1))


@lru_cache(maxsize=7)
def sphere(d,power,seed):
    q=qmc.Sobol(2*d,scramble=True,seed=seed).random_base2(power)
    a=-np.log(q[:,:d])
    weights=a/a.sum(axis=1,keepdims=True)
    phase=np.exp(2j*np.pi*q[:,d:])
    return weights,np.sqrt(weights)*phase


def predict(experiments,mass,power=18,seed=423):
    out,cache=[],{}
    for e in experiments:
        T,rho=e["temperature"],e["density"]
        gap=np.array(e["splitting"])
        d=len(gap)
        mean_condensate=max(0.,rho-d*ZETA*(mass*T/(2*np.pi))**1.5)
        fraction=mean_condensate/rho
        if e["observable"]=="zero_fraction":
            out.append(fraction)
            continue
        if not fraction:
            out.append(0.)
            continue
        key=str({k:v for k,v in e.items() if k not in ("detectors","observable")})
        if key not in cache:
            points,y=sphere(d,power,seed)
            weights=np.exp(-mean_condensate/T*(points@(gap-gap.min())))
            y=y.copy()
            for pulse in e["pulses"]:
                lam,v=eigh(pulse["matrix"])
                u=(v*np.exp(-1j*lam*pulse["duration"]))@v.T
                y=y@u.T
            cache[key]=weights,abs(y)**2
        weight,observation=cache[key]
        detectors=e["detectors"]
        event=np.prod(observation[:,detectors],axis=1)
        out.append(fraction**len(detectors)*np.dot(event,weight)/weight.sum())
    return np.array(out)


X=np.array([[.12,.79,-.45], [.79,-.2,.55],[-.45,.55,.34]])
Y=np.array([[-.29,.31,.52],[.31,.18,-.82],[.52,-.82,.21]])
G=np.array([[.2,.53,-.26,.14],[.53,-.32,.56,.32],[-.26,.56,.17,-.62],[.14,.32,-.62,.28]])


def pulse(M,t):
    return dict(matrix=M.tolist(),duration=t)


def case(T,rho,delta,mix,detectors,observable="coincidence"):
    return dict(temperature=T,density=rho,splitting=delta,pulses=mix,detectors=detectors,observable=observable)


def calibration_inputs():
    return [case(T,rho,[0.]*d,[],[0],"zero_fraction")
            for d in (2,3,4) for T in (.49,.67,.82,1.04)
            for rho in (.67,1.09,1.54)]


def hidden_inputs():
    return {
        "unsplit_finite_fraction":[case(T,rho,[0,0,0],[pulse(X,t)],det)
                                   for T,rho,t in [(.52,1.1,.68),(.68,.77,.94),(.86,1.24,1.24)]
                                   for det in ([0,0],[1,1,1],[1,1,2],[0,2,1,2])],
        "weak_asymmetric_splitting":[case(T,rho,gap,[pulse(X,k),pulse(Y,.63)],det)
                                     for T,rho,gap,k in [(.45,.99,[.89,-.7,.33],.54),
                                                         (.63,1.44,[-.82,.41,-.15],.91),
                                                         (.82,1.32,[.14,.81,-.92],1.35)]
                                     for det in ([0],[1,1],[2,0],[0,0,1],[1,1,1,0])],
        "four_internal_channels":[case(T,rho,delta,[pulse(G,k)],det)
                                   for T,rho,delta,k in [(.48,1.12,[.5,-.97,-.54,.84],.78),
                                                        (.66,1.44,[1.1,.27,-.38,.61],.72),
                                                        (.73,1.66,[.93,-.69,.43,.14],1.59)]
                                   for det in ([0],[2,2],[3,2,3],[0,2,2],[1,1,1,1])],
        "ordered_probe_weak_field":[case(T,rho,[1.26,-1.05,.56],mix,det)
                                    for T,rho in [(.38,1.1),(.56,1.53),(.74,1.87)]
                                    for mix in ([pulse(X,.83),pulse(Y,.79)],[pulse(Y,.79),pulse(X,.83)])
                                    for det in ([0,0,0],[2,2,1],[0,2,2,1])],
    }


def checks():
    cases=hidden_inputs()["four_internal_channels"][::4]
    conv=float(max(abs(predict(cases,TRUE_PARAMETER)-predict(cases,TRUE_PARAMETER,19,871))))
    e=hidden_inputs()["unsplit_finite_fraction"][0]
    coherent_fraction=predict([{**e,"observable":"zero_fraction"}],TRUE_PARAMETER)[0]
    exact=coherent_fraction**2/6
    residual=float(abs(predict([e],TRUE_PARAMETER)[0]-exact))
    ms=hidden_inputs()["weak_asymmetric_splitting"][0]
    cond=predict([{**ms,"observable":"zero_fraction"}],TRUE_PARAMETER)[0]
    count=float(abs(sum(predict([{**ms,"detectors":[i]} for i in range(3)],TRUE_PARAMETER))-cond))
    assert conv<1e-4 and residual<1e-4 and count<1e-9
    return {"haar_integral_refinement_max_abs":conv,"unsplit_dirichlet_second_moment_abs":residual,
            "mode_resolved_total_fraction_residual":count}
