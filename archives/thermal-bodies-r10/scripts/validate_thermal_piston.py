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
OUT = ROOT / "jobs/thermal-piston-validation-r10"


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
    times = np.arange(0, 601, 5, dtype=float)
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
    modules = {"exact": oracle, "independent_modes": baseline}
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
    balance = {"maximum_first_law_residual_W": 0., "maximum_total_energy_residual_W": 0.,
               "minimum_volume_m3": np.inf, "minimum_temperature_K": np.inf,
               "minimum_shortcut_temperature_K": np.inf, "maximum_total_volume_error_m3": 0.,
               "minimum_entropy_production_W_per_K": np.inf}
    checked_initials = [[80., 80.], [1200., 1200.], [80., 1200.], [1200., 80.]]
    checked_initials += [case["initial_temperature"] for case in hidden]
    for g in [4e-4, 8e-4, 1.2e-3]:
        fixed.conductance = g
        for initial in checked_initials:
            temp = ref.trajectory(times, initial, g)
            total = temp.sum(axis=1)
            difference = temp[:, 0] - temp[:, 1]
            total_dot = -g * (total - 2 * ref.TB) / (ref.AMOUNT * ref.CV)
            difference_dot = difference * (-(g + 2 * ref.LINK) / (ref.AMOUNT * ref.CP)
                                            + ref.R / ref.CP * total_dot / total)
            temp_dot = np.column_stack((total_dot + difference_dot, total_dot - difference_dot)) / 2
            volume = ref.VOLUME * temp / total[:, None]
            volume_dot = ref.VOLUME * (temp_dot * total[:, None] - temp * total_dot[:, None]) / total[:, None]**2
            pressure = ref.AMOUNT * ref.R * total / ref.VOLUME
            transfer = ref.LINK * (temp[:, 0] - temp[:, 1])
            heat = -g * (temp - ref.TB) + np.column_stack((-transfer, transfer))
            residual = ref.AMOUNT * ref.CV * temp_dot - heat + pressure[:, None] * volume_dot
            production = g * np.sum((temp - ref.TB)**2 / (temp * ref.TB), axis=1)
            production += ref.LINK * difference**2 / np.prod(temp, axis=1)
            balance["maximum_first_law_residual_W"] = max(balance["maximum_first_law_residual_W"], float(abs(residual).max()))
            balance["maximum_total_energy_residual_W"] = max(balance["maximum_total_energy_residual_W"], float(abs(ref.AMOUNT * ref.CV * total_dot - heat.sum(axis=1)).max()))
            balance["minimum_volume_m3"] = min(balance["minimum_volume_m3"], float(volume.min()))
            balance["minimum_temperature_K"] = min(balance["minimum_temperature_K"], float(temp.min()))
            balance["minimum_shortcut_temperature_K"] = min(balance["minimum_shortcut_temperature_K"], float(fixed.predict(times, initial).min()))
            balance["maximum_total_volume_error_m3"] = max(balance["maximum_total_volume_error_m3"], float(abs(volume.sum(axis=1) - ref.VOLUME).max()))
            balance["minimum_entropy_production_W_per_K"] = min(balance["minimum_entropy_production_W_per_K"], float(production.min()))
    assert balance["maximum_first_law_residual_W"] < 1e-12
    assert balance["maximum_total_energy_residual_W"] < 1e-12
    assert balance["minimum_volume_m3"] > 0 and balance["minimum_temperature_K"] > 0
    assert balance["minimum_shortcut_temperature_K"] > 0
    assert balance["maximum_total_volume_error_m3"] < 1e-15
    assert balance["minimum_entropy_production_W_per_K"] >= 0
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
        assert (a[:, 2:] < .01).all() if name == "exact" else (a[:, 2:] > .05).all()
        monte_carlo[name] = {"columns": ["relative_parameter_error", "reduced_chi2", "hidden_1", "hidden_2", "hidden_3"],
                             "min": a.min(axis=0).tolist(), "max": a.max(axis=0).tolist(),
                             "median": np.median(a, axis=0).tolist(), "calibration_parameter_passes": 256,
                             "all_criteria_passes": int((a[:, 2:] < .05).all(axis=1).sum())}
    tests = {}
    for name, source in (("starting", TASK / "environment/model.py"), ("exact", TASK / "solution/model.py"),
                         ("independent_modes", ROOT / "scripts/thermal_piston_baseline.py")):
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
    report = {"revision": 10, "true_conductance_W_per_K": ref.TRUE_CONDUCTANCE, "dataset_seed": 20260921,
              "measurement_noise_K": .03, "monte_carlo_seed": 1729, "monte_carlo_samples": 256,
              "maximum_oracle_reference_difference_K": numerical_gap,
              "controls": controls, "monte_carlo": monte_carlo, "tests": tests, "physical_balance_checks": balance}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    (ROOT / "results/thermal-r10-validation.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez(OUT / "plot-data.npz", t=times, observations=runs[-1]["temperature"],
             calibration_exact=models["exact"].predict(times, runs[-1]["initial_temperature"]),
             calibration_shortcut=models["independent_modes"].predict(times, runs[-1]["initial_temperature"]),
             hidden_exact=models["exact"].predict(times, hidden[1]["initial_temperature"]),
             hidden_shortcut=models["independent_modes"].predict(times, hidden[1]["initial_temperature"]),
             hidden_truth=hidden_truth[1], mc_exact=np.array(samples["exact"]), mc_shortcut=np.array(samples["independent_modes"]))


if __name__ == "__main__":
    main()
