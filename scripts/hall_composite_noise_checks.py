from pathlib import Path
import numpy as np
from scipy.linalg import expm
from thermal import voltage
from laminate import noise_probe


def sample(materials):
    return dict(
        mode="noise",
        materials=materials,
        bias=[1.1, -1.2],
        baths=[
            dict(phase_modes=[[1.8, 0.3], [1.1, 0.15]], load=0.7),
            dict(phase_modes=[[0.2, 1.6], [1.85, 0.3]], load=0.6),
        ],
        capacitance=[
            [1.2, 0.12, -0.13, 0],
            [0.12, 0.84, 0.08, 0.05],
            [-0.13, 0.08, 0.96, 0.11],
            [0, 0.05, 0.11, 1.1],
        ],
        load=[
            [0.4, 0.1, 0, 0],
            [0.1, 0.5, 0, 0.08],
            [0, 0, 0.3, -0.02],
            [0, 0.08, -0.02, 0.45],
        ],
        wait=0.18,
        lag=0.23,
        contacts=[[0.7, 0.3, -0.5, 0.2], [-0.3, 0.6, 0.4, 0.45]],
    )


def shots(materials):
    rng = np.random.default_rng(18731)

    def spd(lo, hi):
        u, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        vals = np.sort(rng.uniform(lo, hi, 4))
        m = (u * vals) @ u.T
        return np.round((m + m.T) / 2, 5).tolist()

    def channel():
        u = rng.normal(size=4)
        return np.round(0.98 * u / np.linalg.norm(u), 5).tolist()

    def bath():
        t = np.round(rng.uniform(0.12, 2.1, (2, 2)), 5)
        return dict(phase_modes=t.tolist(), load=round(rng.uniform(0.15, 2.0), 5))

    groups = {}
    for code, family in enumerate(
        ("thermal_equilibrium", "split_baths", "bias_quench")
    ):
        rows = []
        for j in range(8):
            b = round(rng.choice([-1, 1]) * rng.uniform(0.9, 1.45), 5)
            s = (
                round(rng.choice([-1, 1]) * rng.uniform(0.75, 1.49), 5)
                if code == 2
                else b
            )
            start = bath()
            finish = bath() if code == 2 else start
            if code == 0:
                t = round(rng.uniform(0.28, 2), 5)
                start = finish = dict(phase_modes=[[t, t], [t, t]], load=t)
            rows.append(
                dict(
                    mode="noise",
                    materials=materials[(code * 8 + j) % len(materials)],
                    bias=[b, s],
                    baths=[start, finish],
                    capacitance=spd(0.56, 1.9),
                    load=spd(0.12, 1.08),
                    wait=round(rng.uniform(0, 0.7), 5) if code == 2 else 0.0,
                    lag=round(rng.uniform(0.025, 0.48), 5),
                    contacts=[channel(), channel()],
                )
            )
        groups[family] = rows
    return groups


def checks(materials):
    q = sample(materials)
    for p in (q, dict(q, bias=[1.1, 1.2])):
        exact = voltage(p, 0.94)
        print("split-bath worked", p["bias"], exact, flush=True)
        assert abs(exact - noise_probe(p, 0.94)) < 1e-10
    m = dict(longitudinal=[[1, 0], [0, 1.9]], hall=[[0.7, 0], [0, -0.45]])
    q = dict(
        q,
        materials=[m, m],
        bias=[-0.3, 1.2],
        capacitance=np.eye(4),
        load=0.4 * np.eye(4),
        baths=[dict(phase_modes=[[1.3, 1.3], [1.3, 1.3]], load=1.3)] * 2,
        contacts=[[1, 0, 0, 0], [0, 0, 1, 0]],
        wait=0.27,
        lag=0.4,
    )
    target = -1.3 * np.exp(-1.4 * 0.4) * np.sin(0.94 * 1.2 * 0.7 * 0.4)
    assert (
        abs(target - voltage(q, 0.94)) < 1e-12
        and abs(target - noise_probe(q, 0.94)) < 1e-12
    )
    print("homogeneous Hall lag", target, flush=True)
