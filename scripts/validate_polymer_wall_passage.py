"""Validate frozen readings and the contact-density recurrence; --write regenerates."""
from pathlib import Path
import json
import sys
import numpy as np

task = Path(__file__).resolve().parents[1] / "tasks/polymer-wall-passage"
for sub in ("environment", "solution", "tests"):
    sys.path.insert(0, str(task/sub))
from reference import angular, extrapolate, value
from passage import density, response
from model import Model

aspect = .9163
rng = np.random.default_rng(28931)


def experiment(walls, gate):
    while True:
        points = rng.uniform([-1.6,.48], [1.6,1.9], size=(2,2)).round(2)
        if np.linalg.norm(points[0]-points[1]) > .35:
            return dict(dyes=points.tolist(), walls=walls, gate=gate)


def rows(inputs):
    return [{"input": r, "value": round(float(value(r,aspect)),9)} for r in inputs]


cal = rows([experiment([], "off") for _ in range(9)])
groups = {}
for group in ("one","same","opposite"):
    inputs = []
    for j in range(10):
        a,b = rng.uniform([.58,.7],[1.6,1.8]).round(2)
        sign = (-1)**j
        probes = ([sign*a] if group=="one" else
                  [sign*a,sign*(a+b)] if group=="same" else [sign*a,-sign*b])
        inputs.append(experiment(probes, ("right","left","left","right")[j%4]))
    groups[group] = rows(inputs)
for path, content in [(task/"environment/data/calibration.json",cal),
                      (task/"tests/data/calibration.json",cal),
                      (task/"tests/data/predictions.json",groups)]:
    if "--write" in sys.argv:
        path.write_text(json.dumps(content, indent=2)+"\n")
    assert content == json.loads(path.read_text())
m = Model().fit(cal)
assert abs(m.aspect-aspect) < 1e-7
for group, rs in groups.items():
    shortcut, error, density_error, min_gate = [], [], [], 1.
    for r in rs:
        q = r["input"]
        z,w = [complex(x,aspect*y) for x,y in q["dyes"]]
        walls = q["walls"]
        cutoff = density(z,w,walls)
        ref = extrapolate(z,w,walls,.0025)
        rev = density(z,w,walls[::-1])
        np.testing.assert_allclose(rev,cutoff,rtol=1e-5,atol=2e-8)
        np.testing.assert_allclose(ref,cutoff,rtol=1e-5,atol=2e-8)
        a = walls[0]
        total = (5/8)/a**2
        if len(walls)==2:
            b=walls[1]
            total = total*(5/8)/b**2 + 2*(5/8)/(a*b*(a-b)**2)
        density_error.append(abs(total-cutoff[0]))
        min_gate = min(min_gate,(cutoff[1] if q["gate"]=="right" else cutoff[0]-cutoff[1])/total)
        error.append(abs(r["value"]-m.predict([q])[0]))
        shortcut.append(abs(cutoff[2]/cutoff[0]-r["value"]))
    print(group,"max oracle error",max(error),"independent dyes",max(shortcut),
          "contact density",max(density_error),"minimum gate fraction",min_gate)
    assert max(error)<.00001 and max(shortcut)>.03 and max(density_error)<1e-8 and min_gate>.001
# Generalization under reflection, scaling, wall reordering, and interchange of the dyes.
q = dict(dyes=[[-.42,.84],[.94,.66]],walls=[-.74,1.34],gate="left")
for walls in ([],[-.74],[1.34],q["walls"]):
    z,w = [complex(*xy) for xy in q["dyes"]]
    a,b,c,d=density(z,w,walls)
    mirror=[-u for u in walls]
    np.testing.assert_allclose(density(-z.conjugate(),-w.conjugate(),mirror),
                               [a,a-b,a-c,a-b-c+d],rtol=1e-6,atol=2e-8)
    np.testing.assert_allclose(density(w,z,walls),[a,c,b,d],rtol=1e-6,atol=2e-8)
    np.testing.assert_allclose(np.array(density(2*z,2*w,[2*u for u in walls]))*4**len(walls),
                               [a,b,c,d],rtol=1e-6,atol=2e-8)
    print("worked",walls,"R after L",response(z,w,walls,"left"),"R after R",d/b)
# The angular probability is a martingale under the common Loewner driving noise.
for z,w in [(.8+.7j,-.5+.9j),(-.3+1.2j,-1.5+.6j),(.6+1.2j,.8+.6j)]:
    e=.0001
    drift=(angular(z+e*2/z,w+e*2/w)-angular(z-e*2/z,w-e*2/w))/(2*e)
    diff=(angular(z+e,w+e)-2*angular(z,w)+angular(z-e,w-e))/e**2
    assert np.max(abs(drift+4/3*diff))<3e-6
