"""Construct an isotropic sequence of rotated laminates, transporting both local probe fields."""
import numpy as np

I = np.eye(2)
O = np.zeros((2, 2))
TURN = np.block([[O, -I], [I, O]])


def physical(material, bias, gain):
    a = np.asarray(material["longitudinal"])
    h = gain * bias * np.asarray(material["hall"])
    return np.block([[a, -h], [h, a]])


def layer(c, d):
    # Normal current and tangential gradient are continuous across the interface.
    parts = [c, d]
    i = [np.linalg.inv(z[:2, :2]) for z in parts]
    j = np.linalg.inv((i[0] + i[1]) / 2)
    connect = np.column_stack((j, j @ (i[0] @ c[:2, 2:] + i[1] @ d[:2, 2:]) / 2))
    maps = [
        np.vstack((m @ (connect - np.column_stack((O, z[:2, 2:]))), np.column_stack((O, I))))
        for z, m in zip(parts, i)
    ]
    return (c @ maps[0] + d @ maps[1]) / 2, maps


def probe(q, gain, steps=9):
    bulk = q["mode"] == "bulk"
    bias = [q["bias"]] * 2 if bulk else q["bias"]
    mats = [[physical(m, b, gain) for m in q["materials"]] for b in bias]
    z1, t1 = layer(*mats[0])
    z2, t2 = layer(*mats[1])
    if bulk:
        effect = np.zeros((4, 4))
    else:
        u, v = map(np.array, q["contacts"])
        mix = np.outer(u, v)
        phi = q["angle"]
        effect = np.block(
            [[np.cos(phi) * mix, np.sin(phi) * mix], [-np.sin(phi) * mix, np.cos(phi) * mix]]
        )
        p = q["phase"]
        effect = t1[p].T @ effect @ t2[p] / 2
    for k in range(steps):
        z1, t1 = layer(z1, TURN @ z1 @ TURN.T)
        z2, t2 = layer(z2, TURN @ z2 @ TURN.T)
        effect = (t1[0].T @ effect @ t2[0] + t1[1].T @ (TURN @ effect @ TURN.T) @ t2[1]) / 2
    if bulk:
        u = np.array(q["contact"])
        a = q["angle"]
        detect = np.r_[np.cos(a) * u, np.sin(a) * u]
        return float(detect @ z1 @ np.asarray(q["drive"]).T.ravel())
    a, b = [np.asarray(d).T.ravel() for d in q["drives"]]
    return float(a @ effect @ b)
