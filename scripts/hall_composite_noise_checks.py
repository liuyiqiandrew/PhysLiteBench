from pathlib import Path
import numpy as np
from scipy.linalg import expm
from noise import voltage
from laminate import noise_probe


def sample(materials):
    return dict(mode="noise", materials=materials, bias=[1.1, -1.2],
        baths=[dict(phase_modes=[[1.8,.3],[1.1,.15]],load=.7),dict(phase_modes=[[.2,1.6],[1.85,.3]],load=.6)],
        capacitance=[[1.2,.12,-.13,0],[.12,.84,.08,.05],[-.13,.08,.96,.11],[0,.05,.11,1.1]],
        load=[[.4,.1,0,0],[.1,.5,0,.08],[0,0,.3,-.02],[0,.08,-.02,.45]],
        wait=.18,lag=.23,contacts=[[.7,.3,-.5,.2],[-.3,.6,.4,.45]])


def shots(materials):
    rng = np.random.default_rng(18731)

    def spd(lo, hi):
        u, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        vals = np.sort(rng.uniform(lo, hi, 4))
        m = (u * vals) @ u.T
        return np.round((m + m.T) / 2, 5).tolist()

    def channel():
        u = rng.normal(size=4)
        return np.round(.98 * u / np.linalg.norm(u), 5).tolist()
    def bath():
        t=np.round(rng.uniform(0.12,2.1,(2,2)),5)
        return dict(phase_modes=t.tolist(),load=round(rng.uniform(.15,2.),5))
    groups = {}
    for code, family in enumerate(("thermal_equilibrium","split_baths","bias_quench")):
        rows=[]
        for j in range(8):
            b=round(rng.choice([-1,1])*rng.uniform(.9,1.45),5)
            s=round(rng.choice([-1,1])*rng.uniform(.75,1.49),5) if code==2 else b
            start=bath()
            finish=bath() if code==2 else start
            if code==0:
                t=round(rng.uniform(.28,2),5)
                start=finish=dict(phase_modes=[[t,t],[t,t]],load=t)
            rows.append(dict(mode="noise",materials=materials[(code*8+j)%len(materials)],
                bias=[b,s],baths=[start,finish],capacitance=spd(.56,1.9),load=spd(.12,1.08),
                wait=round(rng.uniform(0,.7),5) if code==2 else 0.,
                lag=round(rng.uniform(.025,.48),5),contacts=[channel(),channel()]))
        groups[family]=rows
    return groups


def checks(materials):
    q=sample(materials)
    for p in (q,dict(q,bias=[1.1,1.2])):
        exact=voltage(p,.94)
        wrong=voltage(p,.94,adjoint=False)
        print('split-bath worked',p['bias'],exact,wrong,flush=True)
        assert abs(exact-noise_probe(p,.94))<1e-10
        assert abs(exact-wrong)>.005
    m=dict(longitudinal=[[1,0],[0,1.9]],hall=[[.7,0],[0,-.45]])
    q=dict(q,materials=[m,m],bias=[-.3,1.2],capacitance=np.eye(4),load=.4*np.eye(4),
           baths=[dict(phase_modes=[[1.3,1.3],[1.3,1.3]],load=1.3)]*2,
           contacts=[[1,0,0,0],[0,0,1,0]],wait=.27,lag=.4)
    target=-1.3*np.exp(-1.4*.4)*np.sin(.94*1.2*.7*.4)
    assert abs(target-voltage(q,.94))<1e-12 and abs(target-noise_probe(q,.94))<1e-12
    print('homogeneous Hall lag',target,flush=True)
