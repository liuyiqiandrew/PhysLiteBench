from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

stage = Path("staging/waving-sheet-r2/tasks/waving-sheet")
reference = load("wave_ref", stage/"tests/reference.py")
oracle = load("wave_oracle", stage/"solution/model.py")
job = Path("jobs/waving-sheet-zero-three-waving-r2-plain-20261004-150103")
groups = reference.hidden_inputs()
truth = {name: reference.predict(inputs, reference.TRUE_PARAMETER) for name, inputs in groups.items()}
def errors(mod, nu):
    return {name: float(np.sqrt(np.mean((mod.predict_at(inputs, nu)-truth[name])**2)/np.mean(truth[name]**2))) for name,inputs in groups.items()}
report = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Author-only forward diagnostic at each submitted fitted viscosity. No model call, fitting, source edit, resampling or scoring change.", "reference_sha256": sha(stage/"tests/reference.py"), "oracle_sha256": sha(stage/"solution/model.py"), "prediction_limit": .04, "trials": {}, "qualification": "The correction isolates missing nonlinear polymer stress in the final forward model; it does not establish why a trajectory retained that model. CDf5Poe also made tensor-index errors in discarded scratch work, so causal review must retain that mixed qualification."}
for trial in ["waving-sheet__CDf5Poe", "waving-sheet__LYtkN2Z"]:
    source = job/trial/"artifacts/app/model.py"
    metrics_path = job/trial/"verifier/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    nu = metrics["parameters"]["viscosity"]
    mod = load(trial, source)
    actual = errors(mod,nu)
    repaired = errors(oracle,nu)
    assert all(abs(actual[name]-metrics["hidden"][name]) < 1e-10 for name in groups)
    assert all(value < .04 for value in repaired.values())
    report["trials"][trial] = {"viscosity_unchanged": nu, "submitted_source": str(source), "submitted_source_sha256": sha(source), "metrics": str(metrics_path), "metrics_sha256": sha(metrics_path), "submitted_hidden_relative_rms": actual, "physical_forward_model_same_viscosity_relative_rms": repaired, "all_groups_pass_after_physical_forward_correction": True}
report["diagnostic_script_sha256"] = sha(__file__)
output = Path("results/zero-three-waving-r2-fixed-parameter-diagnostic.json")
output.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"report":str(output),"trials":report["trials"]},indent=2))
