"""Validate coincidence imaging against independent spatial first-exit calculations."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/pore-echo-memory"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load(TASK / "solution/model.py")
shortcut = load(ROOT / "scripts/pore_echo_memory_baseline.py")
ref = load(TASK / "tests/reference.py")


def records():
    calibration, groups = [], {}
    for u in [.53,.85,1.07,1.31]:
        for strength in [.35,.9,1.41]:
            q = dict(u=u,y=0,f=0,a=strength,window=None)
            z = 2*u*math.sqrt(ref.TRUE_PARAMETER*strength)
            calibration.append(dict(input=q,value=round(z/math.sinh(z),8)))
    for name in ["separated","pickup","directed"]:
        rows = []
        for i,u in enumerate([.71,.95,1.35]):
            for j,y in enumerate([.15,.55,1.4]):
                f = 0. if name != "directed" else [.56,-.48,.39][(i+j)%3]
                window = None if name=="separated" else [[-.95,-.08],[.31,.78],[y+.12,y+.96]][(i+j)%3]
                q = dict(u=u,y=y,f=f,a=[.3,.81,1.56][(i+2*j)%3],window=window)
                # Richardson extrapolation for the second-order mesh.
                coarse, fine = [ref.predict(q, order=n) for n in [128,256]]
                value = (4*fine - coarse)/3
                assert abs(value-fine) < 5e-5
                rows.append(dict(input=q,value=value))
        groups[name] = rows
    return calibration, groups


def validate(regenerate):
    calibration, groups = records()
    for path in [TASK/"environment/data/calibration.json", TASK/"tests/data/calibration.json"]:
        if regenerate:
            path.write_text(json.dumps(calibration,indent=2)+"\n")
        assert json.loads(path.read_text()) == calibration
    path = TASK/"tests/data/predictions.json"
    if regenerate:
        path.write_text(json.dumps(groups,indent=2)+"\n")
    stored = json.loads(path.read_text())
    assert stored.keys()==groups.keys()
    for group, rows in groups.items():
        assert [r["input"] for r in stored[group]] == [r["input"] for r in rows]
        np.testing.assert_allclose([r["value"] for r in stored[group]], [r["value"] for r in rows], atol=1e-8, rtol=0)
    for role, module in [("oracle",oracle),("shortcut",shortcut)]:
        model = module.Model().fit(calibration)
        assert abs(model.intensity-ref.TRUE_PARAMETER) < 1e-6
        cal = model.predict([r["input"] for r in calibration])
        error = max(abs(cal - [r["value"] for r in calibration]))
        assert error < 1e-6
        errors = {k: float(max(abs(model.predict([r["input"] for r in rows])-[r["value"] for r in rows])))
                  for k,rows in groups.items()}
        print(role, model.intensity, error, errors)
        assert max(errors.values()) < 1e-6 if role=="oracle" else min(errors.values()) > .003

    # Reflect the entire apparatus, then translate by the source separation.
    q = groups["directed"][0]["input"]
    a,b = q["window"]
    reflected = q | {"f":-q["f"],"window":[q["y"]-b,q["y"]-a]}
    np.testing.assert_allclose(oracle.response(q,ref.TRUE_PARAMETER),oracle.response(reflected,ref.TRUE_PARAMETER), atol=1e-10)
    # Independent bulk-valley stop-loss expansion at zero field.
    for row in groups["separated"]:
        q = row["input"]
        s = q["y"]/q["u"]**2
        r = ref.TRUE_PARAMETER*q["a"]*q["u"]**2
        k = (np.pi*(np.arange(64)+.5))**2
        exact = r/np.tanh(np.sqrt(r))**2*np.sum(np.exp(-(k+r)*s) * ((4*k*s-2)/(k+r)**2+8*k/(k+r)**3))
        np.testing.assert_allclose(row["value"],exact,atol=1e-7)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate",action="store_true")
    validate(parser.parse_args().regenerate)
