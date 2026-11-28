"""Pulse, field and counter conventions. Includes a single-photon scattering operation."""
import numpy as np


def pairs(values):
    a = np.asarray(values, dtype=float)
    return a[..., 0] + 1j * a[..., 1]


def spin(angles):
    t, p = angles
    return np.array([np.cos(t), np.exp(1j * p) * np.sin(t)])


def mixer(pair):
    c, s = spin(pair)
    return np.array([[c, -s.conjugate()], [s, c]])


def pulse(q):
    _, theta, phase, stark = q["control"]
    u = np.zeros((3, 3), complex)
    u[:2, :2] = mixer((theta, phase))
    u[2, 2] = np.exp(1j * stark)
    return u


def fields(q):
    dt = np.sqrt(q["durations"])
    f = pairs(q["input_modes"]) / dt
    h = pairs(q["output_modes"]).conjugate() / dt
    # inp[i,j,c] is the ket field of source i in interval j, polarization c.
    inp = f[:, :, None] * mixer(q["input_rotation"]).T[:, None, :]
    # out[k,j,r,c] is the bra field for output mode k and detected polarization r.
    out = (
        h[:, :, None, None]
        * np.array([mixer(p) for p in q["output_rotations"]])[:, None]
    )
    return inp, out


def signal(q, probabilities):
    transfer = np.array([[1, 0], [0, -1]])
    values = np.cos(q["lambda"] * transfer - q["phase"])
    norm = probabilities.sum()
    return float(np.sum(values * probabilities) / norm), float(norm)


def one_photon(q, incident, detected):
    """Ground-to-ground transfer for one isolated photon, with the RF control off.

    incident[j,c] is the ket envelope, detected[j,c] the conjugated receiver envelope.
    The columns propagate both possible initial atomic ground states.
    """
    coupling = np.sqrt(q["rates"])
    excited = np.zeros(2, complex)
    outgoing = np.zeros((2, 2), complex)
    for dt, detune, f, h in zip(q["durations"], q["detuning"], incident, detected):
        rate = sum(q["rates"]) / 2 + 1j * detune
        integral = -np.expm1(-rate * dt) / rate
        drive = -coupling * f
        integrated_excited = integral * excited + (dt - integral) * drive / rate
        outgoing += np.eye(2) * (h @ f) * dt + np.outer(
            coupling * h, integrated_excited
        )
        excited = np.exp(-rate * dt) * excited + integral * drive
    return outgoing
