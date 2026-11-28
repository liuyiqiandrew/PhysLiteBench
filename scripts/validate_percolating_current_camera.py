"""Validate signed winding-cluster counts and their measured gate response."""
from pathlib import Path
import json, sys
import numpy as np
import math
from importlib.util import spec_from_file_location, module_from_spec

task = Path(__file__).resolve().parents[1] / "tasks/percolating-current-camera"
for d in ("environment", "solution", "tests"):
    sys.path.insert(0, str(task / d))
from model import Model
from pickup import geometry
from reference import classes, value
from sample_check import estimate

spec = spec_from_file_location(
    "common", Path(__file__).with_name("percolating_current_camera_baseline.py")
)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
gain = 0.8532
cal = [
    dict(input=dict(field=h), value=round(float(np.tanh(h * gain)), 9))
    for h in [-1.2, -0.85, -0.55, -0.25, 0.15, 0.42, 0.75, 1.1]
]
rng = np.random.default_rng(28436)
fields = [(1.25, -0.25), (2.0, 0.25), (3.0, 0.0), (3.0, 0.375)]
groups = {}
for name in ("neutral", "residue", "oblique"):
    rs = []
    minimum = 1.0
    for j in range(8):
        r, t = fields[j % len(fields)]
        gate = (
            [1, 0, 2 + j % 2, 0]
            if name == "neutral"
            else [1, 0, 3, 1 + j % 2]
            if name == "residue"
            else [[1, 1, 2, 1], [2, -1, 3, 0], [-1, 2, 3, 1], [1, -2, 2, 0]][j % 4]
        )
        q = dict(
            aspect=r,
            shear=t,
            bias=list(rng.uniform(-0.9, 0.9, 2).round(3)),
            gate=gate,
            phase=round(rng.uniform(-2.65, 2.65), 3),
            probe=list(rng.uniform(-2.8, 2.8, 2).round(3)),
        )
        y, p = value(q, gain)
        minimum = min(p, minimum)
        rs.append(dict(input=q, value=round(y, 9)))
    groups[name] = rs
    qs = [row["input"] for row in rs]
    ys = [row["value"] for row in rs]
    delta = max(abs(Model().fit(cal).predict(qs) - ys))
    error = max(abs(mod.Model().fit(cal).predict(qs) - ys))
    print(name, delta, error, minimum)
    assert delta < 1e-7 and error > 0.025 and minimum > 0.015
for r, t in fields:
    bys = {}
    for a, b, n, w in classes(r, t):
        bys[(a, b)] = bys.get((a, b), 0) + w
    for a, b, c, w in geometry(r, t):
        assert abs(bys.get((a, b), 0.0) - w) < 1e-12
    print(
        "geom",
        r,
        t,
        [
            (
                n,
                round(
                    sum(w for a, b, k, w in classes(r, t) if k == n)
                    / sum(bys.values()),
                    7,
                ),
            )
            for n in (1, 2, 3)
        ],
    )
# Compare continuum values with independently sampled physical bond clusters.
residual = []
for rs in groups.values():
    for row in rs:
        q = row["input"]
        idx = fields.index((q["aspect"], q["shear"]))
        y, se = estimate(q, gain, idx)
        residual.append(
            (abs(y - row["value"]), se, abs(y - row["value"]) / max(se, 1e-14))
        )
print(
    "finite 24-site films: maximum discrepancy, SE, standardized residual",
    np.max(residual, axis=0),
)
assert all(error < 5 * se + 0.0015 for error, se, _ in residual)


# Independent form for an unmarked rank-one class (Pinson's three Gaussian sums).
def poisson(c, ell):
    n = np.arange(-32, 33)
    return np.sqrt(c) * np.exp(-np.pi * c * ell * n * n).sum()


for r, t in fields:
    for a, b, co, w in geometry(r, t):
        ell = (a + b * t) ** 2 / r + b * b * r
        pin = (poisson(6, ell) - 2 * poisson(8 / 3, ell) + poisson(2 / 3, ell)) / (
            2 * np.sqrt(2 / 3)
        )
        assert abs(w - pin) < 2e-14
q = dict(
    aspect=3.0,
    shear=0.0,
    bias=[0.0, 0.0],
    gate=[1, 0, 2, 0],
    probe=[1.1, 0.83],
    phase=0.2,
)
print("worked", value(q, 1.0), mod.Model().fit(cal).predict([q]))
for path, obj in [
    (task / "environment/data/calibration.json", cal),
    (task / "tests/data/calibration.json", cal),
    (task / "tests/data/predictions.json", groups),
]:
    if "--write" in sys.argv:
        path.write_text(json.dumps(obj, indent=2) + "\n")
    assert json.loads(path.read_text()) == obj
