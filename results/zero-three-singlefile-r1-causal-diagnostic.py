"""Author diagnostic: change the covariance closure at the recorded diffusivity."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

ORIGINAL = Path('jobs/single-file-memory-zero-three-singlefile-r1-plain-20261004-090706')
REPLACEMENT_PLAN = Path('results/zero-three-singlefile-r1-unstarted-replacement-plan.json')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_covariance(diffusivity, density, time_a, time_b):
    return np.sqrt(diffusivity/np.pi)/density * (np.sqrt(time_a+time_b)-np.sqrt(abs(time_a-time_b)))


def physical_predict_at(experiments, diffusivity):
    return np.asarray([physical_covariance(diffusivity, e['density'], e['time_a'], e['time_b'])
                       for e in experiments])


def main():
    frozen = ORIGINAL/'frozen-task'
    reference = load(frozen/'tests/reference.py', 'crossing_reference')
    starter = load(frozen/'environment/model.py', 'stationary_increment_source')
    inputs = [r['input'] for r in json.loads((frozen/'environment/data/calibration.json').read_text())]
    hidden = reference.hidden_inputs()
    cases = inputs+[e for group in hidden.values() for e in group]
    replacement = Path('jobs')/json.loads(REPLACEMENT_PLAN.read_text())['replacement_job']
    report = {'scope':'Deterministic author-only diagnostic, not a model trial. Original artifacts and rewards are unchanged.',
              'script_sha256':sha(Path(__file__)), 'reference_sha256':sha(frozen/'tests/reference.py'), 'trials':{}}
    for job in [ORIGINAL, replacement]:
        for trial in sorted(job.glob('single-file-memory__*')):
            path = trial/'result.json'
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if result.get('exception_info') or (result.get('verifier_result') or {}).get('rewards',{}).get('reward') != 0:
                continue
            source = trial/'artifacts/app/model.py'
            submitted = load(source, 'submitted_'+trial.name)
            repaired = load(source, 'repaired_'+trial.name)
            metrics = json.loads((trial/'verifier/metrics.json').read_text())
            diffusivity = metrics['parameter']
            a,b = submitted.Model(),repaired.Model()
            a.diffusivity = b.diffusivity = diffusivity
            equivalence = float(np.max(abs(a.predict(cases)-starter.predict_at(cases,diffusivity))))
            assert equivalence < 1e-12, 'Inspect a changed closure before assigning this diagnostic.'
            # Rebind only the physical covariance/prediction closure. Never call or change fit.
            repaired.covariance = physical_covariance
            repaired.predict_at = physical_predict_at
            row = {'job':job.name,'source_sha256':sha(source),'diffusivity_held_fixed':diffusivity,
                   'refit':False,'fit_function_unchanged':True,'submitted_source_closure_equivalence_max':equivalence,
                   'repair':'Replace the stationary-increment covariance by the fixed-preparation conserved-rank covariance. Keep fit, inferred diffusivity and Model API unchanged.',
                   'calibration_max_absolute_change':float(np.max(abs(a.predict(inputs)-b.predict(inputs)))), 'hidden':{}}
            for name,es in hidden.items():
                truth=reference.predict(es,reference.TRUE_PARAMETER)
                before=float(np.linalg.norm(a.predict(es)-truth)/np.linalg.norm(truth))
                after=float(np.linalg.norm(b.predict(es)-truth)/np.linalg.norm(truth))
                row['hidden'][name]={'submitted_nrms':before,'repaired_nrms':after,
                                     'recorded_metric_difference':abs(before-metrics['hidden'][name])}
                assert after < .04 and abs(before-metrics['hidden'][name]) < 1e-10
            assert row['calibration_max_absolute_change'] < 1e-12
            assert sha(source)==row['source_sha256']
            report['trials'][trial.name]=row
    Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print({name:max(x['repaired_nrms'] for x in row['hidden'].values()) for name,row in report['trials'].items()})


if __name__=='__main__':
    main()
