"""Validate the frozen Majorana island predictions and physical controls."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/majorana-island-fringe"
sys.path[:0] = [str(TASK/"solution"), str(TASK/"environment")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


oracle = load("oracle", TASK/"solution/model.py")
ref = load("ref", TASK/"tests/reference.py")
baseline = load("baseline", ROOT/"scripts/majorana_island_fringe_baseline.py")
starter = load("starter", TASK/"environment/model.py")


def runs():
    rng = np.random.default_rng(718305)
    def setting(n):
        return dict(n=n, beta=round(rng.uniform(1.,4.),4), ripple=round(rng.uniform(-.4,.4),4),
                    parity=int(rng.choice([-1,1])), frame=str(rng.choice(["site","wave"])),
                    meter_phase=round(rng.uniform(-2.,2.),4), outcome=int(rng.choice([-1,1])),
                    phase=round(rng.uniform(-2.,2.),4))
    def pulse(indices):
        a,b = rng.choice(indices,2,replace=False)
        return [int(a),int(b),round(rng.uniform(-1.8,1.8),4),round(rng.uniform(-1.6,1.6),4)]
    calibration = []
    for i in range(12):
        q = setting(3+i%5)
        arm = [pulse(range(2*q["n"])) for _ in range(3)]
        q.update(arms=[arm,arm], outcome=1, meter_phase=0.)
        calibration.append(dict(input=q,value=round(ref.TRUE_PARAMETER*np.cos(q["phase"]),8)))

    groups = {}
    for group in ("winding","meter","charge","extended"):
        rows = []
        for i in range(8):
            n = [3,4,5,6,8,11,17,21][i] if group!="extended" else [12,16,13,22,24,28,30,32][i]
            q = setting(n)
            indices = rng.choice(2*n,min(8,2*n),replace=False)
            arm = [pulse(indices) for _ in range(5)]
            if group == "winding":
                a,b = map(int,indices[:2])
                turn = [[a,b,round(np.pi*.75,6),0.]]*2+[[a,b,round(np.pi*.5,6),0.]]
                second = turn+arm
                second = second+[pulse(indices)]
            elif group == "meter":
                second = [p[:] for p in arm]
                for k in (0,3):
                    second[k][3] = -arm[k][3]
                q.update(meter_phase=.68+.13*i,outcome=-1 if i%2 else 1)
            elif group == "charge":
                q.update(frame="site",beta=5.5,parity=-1,phase=1.12+i*.025,
                         outcome=1,meter_phase=.35)
                # Rotate several occupied-level detectors before reading one parity.
                indices = list(map(int,rng.choice(n,min(4,n),replace=False)))
                arm = [[2*j,2*j+1,.0,.45] for j in indices]
                second = [[2*j,2*j+1,1.6,.78] for j in indices]
            else:
                q.update(frame="wave",outcome=1)
                arm = [pulse(indices) for _ in range(12)]
                second = [p[:] for p in arm]
                for k in (1,4,7,11):
                    second[k] = pulse(indices)
            q["arms"] = [arm, second]
            acceptance, signal = ref.measurement(q)
            assert acceptance>=.05 and abs(signal)<=1+1e-12
            rows.append(dict(input=q,value=round(ref.TRUE_PARAMETER*signal,9)))
        groups[group] = rows
    return calibration, groups


def validate(generate):
    calibration, groups = runs()
    for path, data in [(TASK/"environment/data/calibration.json",calibration),
                       (TASK/"tests/data/calibration.json",calibration),
                       (TASK/"tests/data/hidden.json",groups)]:
        if generate:
            path.write_text(json.dumps(data, separators=(",",":")) + "\n")
        saved = json.loads(path.read_text())
        if isinstance(data, list):
            assert saved == data
        else:
            for group, rows in data.items():
                assert [r["input"] for r in rows] == [r["input"] for r in saved[group]]
                np.testing.assert_allclose([r["value"] for r in rows],
                                           [r["value"] for r in saved[group]],atol=2e-9,rtol=0)
    for group, rows in groups.items():
        for row in rows:
            q = row["input"]
            if q["n"] <= 6:
                np.testing.assert_allclose(ref.dense(q),ref.measurement(q),atol=1e-12,rtol=0)
    for model in (oracle, baseline, starter):
        fitted = model.Model().fit(calibration)
        assert abs(fitted.visibility-ref.TRUE_PARAMETER)<1e-8
        errors = {name:float(max(abs(fitted.predict([r["input"] for r in rows])
                                    - [r["value"] for r in rows]))) for name,rows in groups.items()}
        assert (max(errors.values())<.003) == (model is oracle)
        print(model.__name__,fitted.visibility,errors)
    q = dict(calibration[0]["input"],phase=0.,arms=[[],[[0,1,2*np.pi,0.]]])
    np.testing.assert_allclose(oracle.Model().fit(calibration).fringe(q),-1.,atol=1e-12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    validate(parser.parse_args().generate)
