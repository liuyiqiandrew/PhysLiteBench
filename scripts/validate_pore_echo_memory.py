"""Reproduce both data files and check the range contraction against an endpoint optimization."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/pore-echo-memory"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load(TASK / "tests/reference.py")


def records():
    calibration = []
    for d in np.linspace(.21,1.97,12):
        x = dict(mode="single", D_A=d)
        calibration.append(dict(input=x, value=round(ref.predict(x), 8)))
    cases = {}
    for name, coats in [("absorbing", ["DD","DD"]), ("opposite_faces", ["ND","DN"]),
                        ("mixed_faces", ["DD","ND"])]:
        rows = []
        for da,db in [(.3,.65), (.78,1.55), (1.67,.27)]:
            for left,right in [([.31,-.62],[-.48,.83]), ([-.65,.9],[.8,-.34]),
                               ([1.48,.25],[-.92,-1.56]), ([-1.85,-1.22],[1.2,.74])]:
                x = dict(mode="pair", D_A=da, D_B=db, coat=coats, left=left, right=right)
                rows.append(dict(input=x, value=ref.predict(x)))
        cases[name] = rows
    return calibration, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    regenerate = parser.parse_args().regenerate
    calibration, cases = records()
    for path in [TASK / "environment/data/calibration.json", TASK / "tests/data/calibration.json"]:
        if regenerate:
            path.write_text(json.dumps(calibration, indent=2) + "\n")
        assert json.loads(path.read_text()) == calibration
    path = TASK / "tests/data/predictions.json"
    if regenerate:
        path.write_text(json.dumps(cases, indent=2) + "\n")
    stored = json.loads(path.read_text())
    assert stored.keys() == cases.keys()
    for name, rows in cases.items():
        assert [r["input"] for r in rows] == [r["input"] for r in stored[name]]
        np.testing.assert_allclose([r["value"] for r in rows], [r["value"] for r in stored[name]], atol=1e-8, rtol=0)
    for role, source in [("oracle", TASK / "solution/model.py"),
                         ("shortcut", ROOT / "scripts/pore_echo_memory_baseline.py")]:
        model = load(source).Model().fit(calibration)
        assert abs(model.intensity - ref.TRUE_PARAMETER) < 1e-7
        calibration_error = max(abs(model.predict([r["input"] for r in calibration]) - [r["value"] for r in calibration]))
        assert calibration_error < 1e-6
        errors = {}
        for name, rows in cases.items():
            actual = model.predict([r["input"] for r in rows])
            truth = np.array([r["value"] for r in rows])
            errors[name] = float(max(abs(actual/truth - 1)))
        print(role, model.intensity, calibration_error, errors)
        if role == "oracle":
            assert max(errors.values()) < 1e-6
        else:
            assert min(errors.values()) > .2

    for group, rows in cases.items():
        for row in rows:
            q = row["input"]
            # Refine the finite observation time and both wall quadratures.
            fine = ref.predict(q, T=4e8, order=96)
            np.testing.assert_allclose(row["value"], fine, rtol=3e-6, atol=1e-12)
        q = rows[0]["input"]
        translated = q | {"left":[u+.4 for u in q["left"]], "right":[u+.4 for u in q["right"]]}
        np.testing.assert_allclose(ref.predict(translated), ref.predict(q), atol=1e-12)
        assert ref.predict(q | {"left":[1.,1.],"right":[1.,1.]}) == 1.


if __name__ == "__main__":
    main()
