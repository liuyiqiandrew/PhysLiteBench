"""Check both physics controls, independent references, and calibration noise."""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def import_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def validate(task, draws):
    root = ROOT / "tasks" / task
    tests = root / "tests"
    oracle = import_source(root / "solution/model.py", "oracle")
    shortcut_path = ROOT / "scripts" / f"{task.replace('-', '_')}_baseline.py"
    shortcut = import_source(shortcut_path, "shortcut")
    ref = import_source(tests / "reference.py", "reference")
    meta = json.loads((tests / "metadata.json").read_text())
    param = meta["parameter"]
    true_parameter = meta["true_parameter"]
    records = json.loads((tests / "data/calibration.json").read_text())
    assert (root / "environment/data/calibration.json").read_bytes() == (tests / "data/calibration.json").read_bytes()
    inputs = [r["input"] for r in records]
    y = np.array([r["value"] for r in records])
    sigma = np.array([r["sigma"] for r in records])
    noiseless = ref.predict(inputs, true_parameter)
    actual_models = {name: mod.Model().fit(records) for name, mod in (("oracle", oracle), ("shortcut", shortcut))}
    results = {}
    for name, model in actual_models.items():
        prediction = model.predict(inputs)
        cal_chi2 = float(np.sum(((prediction-y)/sigma)**2) / (len(y) - 1))
        relative_error = abs(getattr(model, param)/true_parameter - 1)
        assert cal_chi2 < 1.5 and relative_error < 0.03, (task, name, cal_chi2, relative_error)
        assert model.predict([]).shape == (0,)
        results[name] = {"calibration_chi2": cal_chi2, "parameter_relative_error": relative_error, "hidden": {}}
    frozen_path = tests / "data/hidden_truth.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.is_file() else None
    for name, group in ref.hidden_inputs().items():
        truth = ref.predict(group, true_parameter)
        normalization = max(float(np.sqrt(np.mean(truth**2))), 0.02)
        if frozen is not None:
            assert frozen[name]["inputs"] == group, (task, name, "frozen inputs changed")
            difference = np.max(np.abs(np.asarray(frozen[name]["truth"]) - truth))
            assert difference < 1e-7 * normalization, (task, name, "frozen reference drift", difference)
        accuracy = np.sqrt(np.mean((oracle._response(group, true_parameter)-truth)**2))/normalization
        assert accuracy < 0.002, (task, name, "oracle-reference agreement", accuracy)
        for control, model in actual_models.items():
            error = np.sqrt(np.mean((model.predict(group)-truth)**2))/normalization
            results[control]["hidden"][name] = float(error)
    limit = meta["prediction_limit"]
    assert max(results["oracle"]["hidden"].values()) < limit, results
    assert max(results["shortcut"]["hidden"].values()) > limit, results

    rng = np.random.default_rng(ref.CALIBRATION_SEED + 99619)
    fitted = {"oracle": [], "shortcut": []}
    for _ in range(draws):
        simulated = [{**r, "value": float(value)}
                     for r, value in zip(records, noiseless + rng.normal(0, sigma), strict=True)]
        for name, mod in (("oracle", oracle), ("shortcut", shortcut)):
            fitted[name].append(float(getattr(mod.Model().fit(simulated), param)))
    noise = {}
    for name, values in fitted.items():
        if values:
            relative_error = float(np.max(np.abs(np.array(values)/true_parameter - 1)))
            assert relative_error < 0.03, (task, name, "noise sensitivity", relative_error)
            noise[name] = relative_error
    # At the worst fitted parameter extremes, preserve the oracle/shortcut separation.
    for name, mod in (("oracle", oracle), ("shortcut", shortcut)):
        if not fitted[name]:
            continue
        for estimate in (min(fitted[name]), max(fitted[name])):
            errors = []
            for group in ref.hidden_inputs().values():
                truth = ref.predict(group, true_parameter)
                errors.append(float(np.sqrt(np.mean((mod._response(group, estimate)-truth)**2))
                                    / max(float(np.sqrt(np.mean(truth**2))), 0.02)))
            assert (max(errors) < limit) == (name == "oracle"), (task, name, estimate, errors)
    evidence = dict(task=task, revision=meta["revision"], draws=draws, noise_seed=ref.CALIBRATION_SEED + 99619,
                    controls=results, noise_max_relative_errors=noise,
                    source_sha256={str(p.relative_to(root)): sha(p)
                                   for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts},
                    shortcut_sha256=sha(shortcut_path))
    if hasattr(ref, "checks"):
        evidence["independent_checks"] = ref.checks()
    if draws:
        write_json(ROOT / "results" / f"{task}-validation.json", evidence)
    print(task, results, "noise draws", draws, noise, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--draws", type=int, default=256)
    args = parser.parse_args()
    for task in args.tasks:
        validate(task, args.draws)
