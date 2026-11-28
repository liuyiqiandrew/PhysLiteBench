"""Verify the double-dimer probabilities against independent slit and loop counts."""
import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/dimer-defect-window"
sys.path.insert(0, str(TASK / "environment"))
reference = runpy.run_path(str(TASK / "tests/reference.py"))
oracle = runpy.run_path(str(TASK / "solution/model.py"))
shortcut = runpy.run_path(str(ROOT / "scripts/dimer_defect_window_baseline.py"))


def preparations():
    rng = np.random.default_rng(81913)
    groups = {}
    for name in ("clear", "shared", "veto", "nesting"):
        rows = []
        sizes = [(3, 4), (4, 4), (5, 4), (8, 9), (12, 10), (16, 18), (22, 21), (32, 32)]
        if name == "nesting":
            sizes = [(14, 16), (16, 16), (18, 20), (20, 21), (22, 24), (25, 24), (28, 30), (32, 32)]
        for i, (w, h) in enumerate(sizes):
            x, y = (w - 2) // 2, (h - 2) // 2
            if w > 5:
                x += int(rng.integers(-2, 2))
                y += int(rng.integers(-2, 2))
            offset = 1 if i % 3 or w < 5 else 2
            marks = [[x, y], [min(x + offset, w - 2), min(y + i % 2, h - 2)]]
            assert marks[0][0] != marks[1][0]
            n = {"clear": 0, "shared": 1, "veto": i % 2, "nesting": 2}[name]
            q = dict(
                mode="loops",
                w=w,
                h=h,
                ratio=round(float(rng.uniform(0.52, 2.05)), 4),
                marks=marks,
                count=n,
                allow_separate=name != "veto" and (name != "nesting" or i % 3 != 0),
            )
            v = reference["reading"](q, reference["TRUE_PARAMETER"])
            assert np.isfinite(v) and -1e-9 <= v <= 1 + 1e-9
            if w * h <= 20:
                exact = reference["brute"](q, reference["TRUE_PARAMETER"])
                assert abs(v - exact) < 1e-11
            rows.append(dict(input=q, value=round(v, 9)))
        groups[name] = rows
    return groups


def validate(generate):
    z = reference["TRUE_PARAMETER"]
    cal = []
    for ratio in [0.43, 0.58, 0.72, 0.85, 1.0, 1.19, 1.42, 1.75, 2.02, 2.15]:
        t = (z * ratio) ** 2
        cal.append(dict(input=dict(mode="bond", ratio=ratio), value=round(t / (1 + t), 8)))
    groups = preparations()
    for path, data in [
        (TASK / "environment/data/calibration.json", cal),
        (TASK / "tests/data/calibration.json", cal),
        (TASK / "tests/data/predictions.json", groups),
    ]:
        if generate:
            path.write_text(json.dumps(data, indent=1) + "\n")
        saved = json.loads(path.read_text())
        if isinstance(data, list):
            assert data == saved
        else:
            assert saved.keys() == data.keys()
            for group, rows in data.items():
                assert [r["input"] for r in rows] == [r["input"] for r in saved[group]]
                np.testing.assert_allclose(
                    [r["value"] for r in rows],
                    [r["value"] for r in saved[group]],
                    atol=2e-9,
                    rtol=0,
                )
    q = dict(
        mode="loops", w=3, h=2, ratio=0.8, marks=[[0, 0], [1, 0]], count=1, allow_separate=True
    )
    for value in (reference["brute"](q, z), reference["reading"](q, z), oracle["reading"](q, z)):
        assert abs(value - 2 / (2 + (0.8 * z) ** 2) ** 2) < 1e-12
    q.update(w=10, h=12, marks=[[4, 3], [5, 4]])
    for fun in (oracle["reading"], reference["reading"]):
        unrestricted = [fun(dict(q, count=i), z) for i in range(5)]
        veto = [fun(dict(q, count=i, allow_separate=False), z) for i in range(5)]
        assert abs(sum(unrestricted) - 1) < 1e-10
        assert all(-1e-10 <= a <= b + 1e-10 for a, b in zip(veto, unrestricted))
    for label, module in [("oracle", oracle), ("shortcut", shortcut)]:
        model = module["Model"]().fit(cal)

        def max_error(rows):
            return float(
                max(abs(model.predict([r["input"] for r in rows]) - [r["value"] for r in rows]))
            )

        assert max_error(cal) < 2e-5 and abs(model.activity - z) < 1e-7
        errors = {g: max_error(rows) for g, rows in groups.items()}
        assert max(errors.values()) < 1e-7 if label == "oracle" else min(errors.values()) > 0.004
        print(label, model.activity, errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    validate(parser.parse_args().generate)
