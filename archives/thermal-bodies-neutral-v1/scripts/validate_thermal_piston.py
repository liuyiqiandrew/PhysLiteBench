"""Generate and validate the free-piston thermal task."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/thermal-bodies"
OUT = ROOT / "jobs/thermal-piston-validation-v9"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-data", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ref = load("piston_reference", TASK / "tests/reference.py")
    oracle = load("piston_oracle", TASK / "solution/model.py")
    baseline = load("piston_baseline", ROOT / "scripts/thermal_piston_baseline.py")
    times = np.arange(0, 241, 2, dtype=float)
    specs = [{"file": f"data/run_{i+1}.csv", "initial_temperature": initial}
             for i, initial in enumerate(([298., 288.], [273., 313.], [313., 273.]))]
    truths = [ref.trajectory(times, s["initial_temperature"]) for s in specs]
    if args.generate_data:
        rng = np.random.default_rng(20260921)
        for directory in (TASK / "environment", TASK / "tests"):
            (directory / "data").mkdir(parents=True, exist_ok=True)
            for old in (directory / "data").glob("*.csv"):
                old.unlink()
            (directory / "metadata.json").write_text(json.dumps({"runs": specs}, indent=2)+"\n")
        for spec, truth in zip(specs, truths):
            data = np.column_stack((times, truth+rng.normal(0., .03, truth.shape), np.full(truth.shape, .03)))
            for directory in (TASK / "environment", TASK / "tests"):
                np.savetxt(directory / spec["file"], data, delimiter=",", fmt="%.17g", comments="",
                           header="time,temperature_1,temperature_2,sigma_1,sigma_2")
    for filename in ["metadata.json", *[s["file"] for s in specs]]:
        assert (TASK / "environment" / filename).read_bytes() == (TASK / "tests" / filename).read_bytes()
    runs = ref.load_runs(TASK / "environment")
    hidden = ref.hidden_cases()
    hidden_truth = [ref.trajectory(r["t"], r["initial_temperature"]) for r in hidden]
    modules = {"exact": oracle, "fixed_pressure": baseline}
    models = {name: module.ThermalModel().fit(runs) for name, module in modules.items()}
    controls = {name: ref.metrics(model, runs) for name, model in models.items()}
    print(json.dumps(controls, indent=2), flush=True)
    exact = oracle.ThermalModel()
    exact.conductance = ref.TRUE_CONDUCTANCE
    fixed = baseline.ThermalModel()
    fixed.conductance = ref.TRUE_CONDUCTANCE
    numerical_gap = 0.
    for case in [*runs, *hidden]:
        for t in (times, np.array([.1, 17.3, 149.]), np.array([0.])):
            value = exact.predict(t, case["initial_temperature"])
            truth = ref.trajectory(t, case["initial_temperature"])
            numerical_gap = max(numerical_gap, float(np.max(np.abs(value-truth))))
            np.testing.assert_allclose(value, truth, atol=5e-8, rtol=0.)
    for g in [4e-4, 8e-4, 1.2e-3]:
        fixed.conductance = g
        for run in runs:
            np.testing.assert_allclose(fixed.predict(times, run["initial_temperature"]),
                                       ref.trajectory(times, run["initial_temperature"], g), atol=1e-12, rtol=0.)
    samples = {name: [] for name in modules}
    rng = np.random.default_rng(1729)
    for iteration in range(256):
        noisy = copy.deepcopy(runs)
        for run, truth in zip(noisy, truths):
            run["temperature"] = truth+rng.normal(0., run["sigma"])
        for name, module in modules.items():
            model = module.ThermalModel().fit(noisy)
            chi = sum(np.sum(((model.predict(r["t"], r["initial_temperature"])-r["temperature"])/r["sigma"])**2) for r in noisy)/(sum(r["temperature"].size for r in noisy)-1)
            errors = [np.linalg.norm(model.predict(r["t"], r["initial_temperature"])-truth)/np.linalg.norm(truth-ref.TB)
                      for r, truth in zip(hidden, hidden_truth)]
            samples[name].append([abs(model.conductance/ref.TRUE_CONDUCTANCE-1), chi, *errors])
        if (iteration+1) % 64 == 0:
            print(f"Validated {iteration+1}/256 noise realizations", flush=True)
    monte_carlo = {}
    for name, rows in samples.items():
        a = np.asarray(rows)
        assert (a[:, 0] < .05).all() and (a[:, 1] < 1.5).all()
        assert (a[:, 2:] < .01).all() if name == "exact" else (a[:, 2:] > .1).all()
        monte_carlo[name] = {"columns": ["relative_parameter_error", "reduced_chi2", "mean_1", "mean_2", "mean_3"],
                             "min": a.min(axis=0).tolist(), "max": a.max(axis=0).tolist(),
                             "median": np.median(a, axis=0).tolist(), "calibration_parameter_passes": 256,
                             "all_criteria_passes": int((a[:, 2:] < .05).all(axis=1).sum())}
    tests = {}
    for name, source in (("starting", TASK / "environment/model.py"), ("exact", TASK / "solution/model.py"),
                         ("fixed_pressure", ROOT / "scripts/thermal_piston_baseline.py")):
        with tempfile.TemporaryDirectory(prefix="thermal-piston-check-") as tmp:
            app, private = Path(tmp) / "app", Path(tmp) / "tests"
            for src, dest in ((TASK / "environment", app), (TASK / "tests", private)):
                shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
            shutil.copyfile(source, app / "model.py")
            env = dict(os.environ, PYTHONPATH=str(app), PYTHONDONTWRITEBYTECODE="1", THERMAL_METRICS_PATH=str(OUT / f"{name}-metrics.json"))
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", tmp,
                                     "--confcutdir", tmp, str(app / "test_public.py"), str(private / "test_hidden.py")],
                                    cwd=app, env=env, capture_output=True, text=True)
            log = result.stdout+result.stderr
            (OUT / f"{name}-tests.txt").write_text(log)
            tests[name] = {"returncode": result.returncode, "summary": log.strip().splitlines()[-1]}
            assert result.returncode == (0 if name == "exact" else 1), tests[name]
            print(name, tests[name], flush=True)
    report = {"revision": 9, "true_conductance_W_per_K": ref.TRUE_CONDUCTANCE, "dataset_seed": 20260921,
              "measurement_noise_K": .03, "monte_carlo_seed": 1729, "monte_carlo_samples": 256,
              "maximum_oracle_reference_difference_K": numerical_gap,
              "controls": controls, "monte_carlo": monte_carlo, "tests": tests}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez(OUT / "plot-data.npz", t=times, observations=runs[-1]["temperature"],
             calibration_exact=models["exact"].predict(times, runs[-1]["initial_temperature"]),
             calibration_shortcut=models["fixed_pressure"].predict(times, runs[-1]["initial_temperature"]),
             hidden_exact=models["exact"].predict(times, hidden[1]["initial_temperature"]),
             hidden_shortcut=models["fixed_pressure"].predict(times, hidden[1]["initial_temperature"]),
             hidden_truth=hidden_truth[1], mc_exact=np.array(samples["exact"]), mc_shortcut=np.array(samples["fixed_pressure"]))


if __name__ == "__main__":
    main()
