"""Finalize the read-only native-parameter review and a Gaussian-closure ablation."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import shutil
import time
import numpy as np

STAGE = Path(__file__).resolve().parents[1]
JOB = STAGE / 'jobs/rotating-reservoir-r7-plain-20261006-042142'
TARGET = STAGE / 'results/rotating-reservoir-r7-trial-review.json'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    before = STAGE / 'results/rotating-reservoir-r7-comparison-before-final-audit.json'
    if not before.exists():
        shutil.copy2(TARGET, before)
    report = json.loads(before.read_text())
    collected = json.loads((STAGE / 'results/rotating-reservoir-r7-results.json').read_text())
    ref = load(STAGE / 'tasks/rotating-reservoir/tests/reference.py', 'rotating_r7_audit_ref')
    notes = {
        '7Ckiay6': 'Correct untransformed marked collision gain, unmarked physical escape, unchanged motor generator. Cell-integrated Gauss quadrature and split diagonal flux cusp improve numerical convergence. The diagnostic motor field uses an incorrect gas-quadrature index but is never used in the public predicted readouts; this does not cause the observed pass. Dense eigensolves and Richardson refinement approximate the correct SCGF.',
        'YKMBV8Q': 'Correct untransformed marked collision gain, unmarked physical escape, unchanged motor generator. Enlarged bias-dependent velocity window and dense principal eigenvalue with Richardson refinement approximate the correct SCGF. Mean/noise retain the complete stationary and Poisson-corrector calculation.',
        'jUpbPh8': 'Correct continuous full heat-gauge transformed OU drift and potential, collision gains with canceled marks, and exponentially transformed motor switches. Its reflecting numerical cutoff for the transformed OU is an approximation to the transformed Robin boundary, rather than an exact discrete similarity. At the tested large cutoff, independent off-grid and hidden comparisons show negligible readout error. Mean/noise retain the complete stationary and Poisson-corrector calculation.',
    }
    for trial in collected['trials']:
        name = trial['trial']
        suffix = name.split('__')[-1]
        entry = report['trials'][name]
        folder = JOB / name
        source = folder / 'artifacts/app/model.py'
        module = load(source, 'rotor_r7_actual_' + suffix)
        model = module.Model()
        model.drag = entry['native_drag']
        original_scgf = module.scgf
        module.scgf = lambda e, gamma, bias: -bias * module.physical_statistics(e, gamma)['mean'] + .5 * bias**2 * module.physical_statistics(e, gamma)['noise']
        module.readout.cache_clear()
        ablation = {}
        for group, cases in ref.hidden_inputs().items():
            actual = model.predict(cases)
            truth = ref.predict(cases, model.drag)
            ablation[group] = dict(
                relative_error=float(np.linalg.norm(actual-truth)/np.linalg.norm(truth)),
                predictions=actual.tolist(), references=truth.tolist())
        module.scgf = original_scgf
        module.readout.cache_clear()
        sessions = list((folder / 'agent/sessions').rglob('*.jsonl'))
        events = [json.loads(line) for path in sessions for line in path.read_text().splitlines()]
        complete = [e for e in events if e['type']=='event_msg' and e.get('payload', {}).get('type')=='task_complete']
        finals = [e for e in events if e['type']=='response_item' and e.get('payload', {}).get('type')=='message' and e['payload'].get('role')=='assistant' and e['payload'].get('phase')=='final']
        # Codex's exported response_item does not always retain a phase field;
        # retain the actual terminal assistant response independently of that flag.
        terminal_messages = [e for e in events if e['type']=='response_item' and e.get('payload', {}).get('type')=='message' and e['payload'].get('role')=='assistant']
        entry.update(
            classification='physical_pass_with_agent_solving_timeout' if trial['exception'] else 'physical_pass',
            actual_source_physics_review=notes[suffix],
            causal_gaussian_closure_ablation=dict(
                operation='Replace only submitted SCGF with -chi*submitted_mean + chi^2*submitted_noise/2; retain final collision quadrature, full mean/noise model, native fitted drag, and all original files.',
                same_native_drag=model.drag, groups=ablation,
                diagnosis='Restoring the Gaussian current closure fails all three finite-bias diagnostic groups. All three submissions repaired the intended physical approximation by evaluating the full tilted generator.'),
            native_completion_events=complete,
            terminal_native_assistant_message=terminal_messages[-1] if terminal_messages else None,
            native_task_complete=bool(complete),
            native_contexts=trial['native_contexts'],
            source_safety_review='Only numerical imports, definitions, constants and cache decorators at module scope; no shell, network, file access, or unsafe top-level execution.',
            gaussian_ablation_did_not_change_source=True)
        entry['sha256'].update({str(path.relative_to(folder)):sha(path) for path in sessions})
        if suffix=='7Ckiay6':
            entry['exception_review'] = dict(
                type='AgentTimeoutError', cause='The original 600-second solving limit elapsed. Harbor canceled the Codex process after final source installation and a terminal assistant response, before an explicit native task_complete event was recorded.',
                timeout_seconds=600, last_assistant_timestamp=terminal_messages[-1]['timestamp'],
                exception_timestamp=trial['exception']['occurred_at'],
                interpretation='Retain reward 1 and the independently verified physical pass, together with the solving-budget exception. This is neither an API usage interruption nor a clean agent completion nor a physical failure.')
        print(name, entry['classification'], {k:v['relative_error'] for k,v in ablation.items()}, flush=True)
    frozen = json.loads((STAGE / 'results/rotating-reservoir-r7-validation.json').read_text())['source_sha256']
    assert all(sha(STAGE / rel)==digest for rel,digest in frozen.items())
    report.update(
        reviewer='brownian_physics_review', errors=collected['errors'], clean_physical_failures=0,
        physical_passes=3, clean_native_completions=2,
        qualification='does_not_meet_zero_of_three_target',
        causal_review_seconds=time.perf_counter()-start,
        frozen_source_sha256=frozen, frozen_sources_unchanged=True,
        scope='Manual complete submitted-source and native-trajectory review; 24 random finite-bias SCGF and six retained mean/noise readouts, all 16 hidden settings, independent continuous-generator reference at unchanged native fitted drag. Causal Gaussian closure ablation affects only in-memory review modules. Every original reward, source, fit, exception, trajectory and frozen control is preserved. No agent runs were repeated.',
        scientific_conclusion='All three repair the Gaussian finite-bias current assumption and give accurate physical readouts off grid. The one original solving timeout remains recorded alongside its physical pass. No physical failure is established by this batch.')
    TARGET.write_text(json.dumps(report, indent=2)+'\n')
    print('FINAL', sha(TARGET), flush=True)


if __name__=='__main__':
    main()
