"""Regenerate rounded calibration and frozen counts only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/stellar-tracer-mixing"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load("ref", TASK / "tests/reference.py")
oracle = load("oracle", TASK / "solution/model.py")
control = load("control", ROOT / "scripts/stellar_tracer_mixing_baseline.py")


def inputs(field, steps, radii):
    return [dict(slow=field, steps=steps, peri=rp, apo=ra) for rp, ra in radii]


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    cal_path = TASK / "environment/data/calibration.json"
    private = TASK / "tests/data"
    if args.generate:
        cal = inputs([1, 1], [], [(.2, .6), (.3, 1.8), (.8, 2.3), (1.2, 4.8),
                                   (.5, 6.7), (1.6, 3.9), (.7, 1.1), (2, 7.4)])
        rows = [dict(input=q, value=float(format(ref.signal(q), ".9g"))) for q in cal]
        write(cal_path, rows)
        write(private / "calibration.json", rows)
        a, b, c = [1.9, .41], [.64, 1.7], [1.28, .69]
        bins = [(.2, 1.2), (.65, 2.8), (.31, 5.2), (1.1, 4.9)]
        groups = {"slow": inputs(a, [], bins), "one_switch": inputs(c, [a], bins),
                  "mixed_cycle": inputs(a, [b, a], bins),
                  "two_switches": inputs(b, [a, c], bins)}
        write(private / "predictions.json", {g: [dict(input=q, value=ref.signal(q, order=80))
                                                 for q in qs] for g, qs in groups.items()})
    assert cal_path.read_bytes() == (private / "calibration.json").read_bytes()
    rows = json.loads(cal_path.read_text())
    cases = json.loads((private / "predictions.json").read_text())
    for impl in (oracle, control):
        model = impl.Model().fit(rows)
        assert abs(model.exponent - ref.TRUE_PARAMETER) < 1e-7
        assert np.allclose(model.predict([r["input"] for r in rows]),
                           [r["value"] for r in rows], rtol=1e-7)
        errors = {}
        for group, sample in cases.items():
            qs = [r["input"] for r in sample]
            truth = np.array([r["value"] for r in sample])
            check = [ref.signal(q, order=56) for q in qs]
            assert np.max(abs(check / truth - 1)) < 0.001
            errors[group] = float(np.max(abs(model.predict(qs) / truth - 1)))
        if impl is oracle:
            assert max(errors.values()) < 0.001
        else:
            assert errors["slow"] < 0.001 and min(v for k, v in errors.items() if k != "slow") > .02
        print(impl.__name__, "exponent", model.exponent, "max relative errors", errors)


if __name__ == "__main__":
    main()
