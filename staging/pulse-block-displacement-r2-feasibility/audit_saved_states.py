"""Finish the predeclared instantaneous checks using saved reference states."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from scipy.special import beta

stage = Path(__file__).resolve().parent
report = json.loads((stage / "report.json").read_text())
run = Path(report["run"])
states = json.loads((run / "reference-cases.json").read_text())
nodes, weights = np.polynomial.legendre.leggauss(64)
rows = []
for row in states:
    f, center, width, ratio, clock = row["case"]
    t = clock/width
    frequencies = center+width*nodes
    profile = weights*(1-nodes**2)**4
    norm = np.sqrt(width/(np.pi*beta(.5, 9)))
    e = norm*np.dot(profile, np.cos(frequencies*t))
    edot = -norm*np.dot(profile*frequencies, np.sin(frequencies*t))
    p, current = row["readout_state"][:2]
    reflected, transmitted = -current/2, e-current/2
    e_left, b_left = e+reflected, e-reflected
    e_right = b_right = transmitted
    e_sheet = (e_left+e_right)/2
    stress = (e_left**2+b_left**2-e_right**2-b_right**2)/2
    energy_flux = e**2-reflected**2-transmitted**2
    pdotdot = f*e_sheet-p
    energy_rate = (current*pdotdot+p*current)/f
    rows.append({"index": row["index"], "case": row["case"],
                 "electric_jump": e_left-e_right,
                 "magnetic_jump_residual": b_right-b_left+current,
                 "stress_lorentz_difference": stress-current*e,
                 "energy_flux_derivative_difference": energy_flux-energy_rate,
                 "canonical_kinetic_identity_difference":
                     stress-(-p*edot+current*e+p*edot)})
keys = [k for k in rows[0] if k not in ["index", "case"]]
result = {"scope": "Predeclared instantaneous identities at all40 saved reference readout states; no changed domain, gap survey or new ODE solve.",
          "input_sha256": sha256((run / "reference-cases.json").read_bytes()).hexdigest(),
          "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
          "incident_evaluation": "64-point direct spectral quadrature, separate from the reference Bessel incident evaluation and oracle FFT.",
          "maximum_absolute_residuals": {k: max(abs(r[k]) for r in rows) for k in keys},
          "rows": rows}
(run / "instantaneous-audit.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result["maximum_absolute_residuals"], indent=2))
