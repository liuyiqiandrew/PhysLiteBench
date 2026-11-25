"""Author-only causal check. No fitting, task edits, or model-agent execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / 'jobs/finite-band-reservoir-zero-three-finiteband-r1-plain-20261004-072705'
TRIAL = JOB / 'finite-band-reservoir__bfe6ZmK'
TASK = ROOT / 'staging/finite-band-reservoir-r1/tasks/finite-band-reservoir'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nrms(prediction, truth):
    return float(np.sqrt(np.mean((prediction-truth)**2)/np.mean(truth**2)))


source_path = TRIAL / 'artifacts/app/model.py'
source_hash = sha(source_path)
source = source_path.read_text()
old = '        result += weight*expit(-pole/temperature)'
new = '''        initial_chain = contact**2*float(weights@(
            (2*sine**2/np.pi)*filling/(pole-band_energy)**2))
        result += weight**2*(1+initial_chain)'''
assert source.count(old) == 1
original = load('finiteband_submitted', source_path)
repaired = types.ModuleType('finiteband_population_repair')
exec(compile(source.replace(old, new), '<population-only-repair>', 'exec'), repaired.__dict__)
reference = load('finiteband_reference', TASK / 'tests/reference.py')
metrics = json.loads((TRIAL / 'verifier/metrics.json').read_text())
coupling = metrics['parameter']
before, after = original.Model(), repaired.Model()
before.coupling_scale = after.coupling_scale = coupling
records = json.loads((TASK / 'environment/data/calibration.json').read_text())
calibration = [r['input'] for r in records]

# Reproduce the submitted scratch formula as written in public trajectory step9.
trajectory = json.loads((TRIAL / 'agent/trajectory.json').read_text())
step = next(s for s in trajectory['steps'] if s['step_id'] == 9)
js = step['tool_calls'][0]['arguments']['input']
command = json.loads(re.search(r'cmd:("(?:[^"\\]|\\.)*")', js).group(1))
scratch_python = command.split("python - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
definitions = scratch_python.split('for V in [1.95,2.4,3.2,3.8]:', 1)[0]
sys.modules['model'] = original
scratch = {}
exec(definitions, scratch)

hidden = {}
for name, experiments in reference.hidden_inputs().items():
    truth = reference.predict(experiments)
    fixed_truth = reference.predict(experiments, coupling_scale=coupling)
    uncorrected, corrected = before.predict(experiments), after.predict(experiments)
    scratch_prediction = []
    for e in experiments:
        continuum, poles = scratch['calc'](e['orbital_energy'], e['temperature'], coupling*e['contact_multiplier'])
        scratch_prediction.append(continuum+sum(p[3] for p in poles))
    hidden[name] = {
        'submitted_nrms': nrms(uncorrected, truth),
        'population_repair_nrms': nrms(corrected, truth),
        'repair_vs_independent_reference_at_fixed_coupling_max_abs': float(np.max(np.abs(corrected-fixed_truth))),
        'submitted_scratch_vs_population_repair_max_abs': float(np.max(np.abs(np.asarray(scratch_prediction)-corrected))),
        'recorded_metric_difference': abs(nrms(uncorrected, truth)-metrics['hidden'][name]),
        'repaired_passes_unchanged_gate': nrms(corrected, truth) < .04,
    }
assert sha(source_path) == source_hash
report = {
    'trial': TRIAL.name,
    'purpose': 'Replace only final bound-mode Gibbs populations by their projection of the initial product state, at the submitted fitted coupling.',
    'fitted_coupling_held_fixed': coupling,
    'refitting_performed': False,
    'source_sha256_before_and_after': source_hash,
    'original_artifacts_unchanged': True,
    'replacement': {'old': old, 'new': new},
    'calibration_predictions_max_abs_change': float(np.max(np.abs(before.predict(calibration)-after.predict(calibration)))),
    'hidden': hidden,
    'scratch_evidence': {
        'trajectory_step': 9,
        'native_call_id': step['tool_calls'][0]['tool_call_id'],
        'formula_source': 'The public command itself; executed unchanged except retaining its function definitions and using scored experiment inputs.',
        'finding': 'The scratch quench-projection formula is correct. It was not installed. No algebraic or numerical error in that formula explains retaining the Gibbs kernel.',
    },
    'limitations': 'This causal check establishes the final physical error and adequacy of the available scratch correction. It does not infer the agent\'s unobserved reason for discarding the correction.',
    'reference_sha256': sha(TASK / 'tests/reference.py'),
    'diagnostic_script_sha256': sha(Path(__file__)),
}
output = ROOT / 'results/zero-three-finiteband-r1-fixed-coupling-repair.json'
output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
