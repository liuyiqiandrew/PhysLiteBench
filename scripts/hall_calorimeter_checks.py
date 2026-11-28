import numpy as np
from calorimeter import characteristic, heat
from heat_ref import heat_probe
from hall_composite_noise_checks import sample


def shots(batches):
    rng = np.random.default_rng(1238914)
    out = {}
    for group, family in [
        ("thermal_equilibrium", "heat_equilibrium"),
        ("split_baths", "heat_fluctuations"),
        ("bias_quench", "heat_quench"),
    ]:
        rows = []
        for j, q in enumerate(batches[group]):
            a = {k: v for k, v in q.items() if k not in ("contacts", "lag")}
            a.update(
                mode="heat",
                duration=round(rng.uniform(0.19, 0.72), 5),
                tilt=round(rng.choice([-1, 1]) * rng.uniform(0.43, 1.12), 5),
                angle=0
                if group == "thermal_equilibrium"
                else round(rng.uniform(-2.9, 2.9), 5),
            )
            rows.append(a)
        out[family] = rows
    return out


def checks(materials):
    q = sample(materials)
    a = dict(q, mode="heat", duration=0.43, tilt=0.87, angle=0)
    for p in [a, dict(a, angle=np.pi / 2)]:
        ans = heat(p, 0.94)
        wrong = heat(p, 0.94, False)
        print("split-bath load heat", p["angle"], ans, wrong, flush=True)
        assert abs(heat_probe(p, 0.94) - ans) < 1e-7
    # Independent closed form for diagonal equilibrium conductors.
    C = np.array([1.0, 0.86])
    f = np.array([1.0, 1.9])
    l = np.array([0.4, 0.35])
    t = 1.3
    dt = 0.4
    k = 0.75
    v = (f + l) / C
    lam = np.sqrt(v * v + 4 * k * k * t * t * f * l / C**2)
    exact = np.prod(
        np.exp(v * dt / 2)
        / np.sqrt(
            np.cosh(lam * dt)
            + (v + 2 * k * k * t * t * l / C) * np.sinh(lam * dt) / lam
        )
    )
    A = np.diag(v)
    D = np.diag(2 * t * (f + l) / C**2)
    ans = characteristic(A, D, np.diag(t / C), np.diag(1 / C), np.diag(l), t, dt, k)
    assert abs(ans - exact) < 1e-12
    print("homogeneous equilibrium four ports", exact**2, flush=True)
