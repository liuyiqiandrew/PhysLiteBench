"""Author-only kernel repair at each failed submission's fixed viscosity."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import types
import numpy as np

JOB = Path('jobs/finite-layer-phoresis-zero-three-phoresis-r2-outage-recovery-20261004-0815')
FROZEN = Path('jobs/finite-layer-phoresis-zero-three-phoresis-r2-plain-20261004-080542/frozen-task')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def helper(source):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'response'), include_attributes=False)


def main():
    reference_path = FROZEN/'tests/reference.py'
    reference = module(reference_path, 'independent_reference')
    starter = (FROZEN/'environment/model.py').read_text()
    inputs = [r['input'] for r in json.loads((FROZEN/'environment/data/calibration.json').read_text())]
    report = {'scope': 'Deterministic author diagnostic, not an additional model trial.',
              'reference_sha256': sha(reference_path), 'script_sha256': sha(Path(__file__)), 'trials': {}}
    for trial in sorted(JOB.glob('finite-layer-phoresis__*')):
        path = trial/'result.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result.get('exception_info') or result.get('verifier_result', {}).get('rewards', {}).get('reward') != 0:
            continue
        source_path = trial/'artifacts/app/model.py'
        source = source_path.read_text()
        if helper(source) != helper(starter):
            continue
        old = 'weight = (r-radius)**2*(2*r+radius)/(2*r)'
        new = 'weight = (r-radius)*((r-radius)*(2*r+radius)/(2*r)+3*beta*r)/(1+3*beta)'
        assert source.count(old) == 1
        submitted = module(source_path, 'submitted_'+trial.name)
        repaired = types.ModuleType('repaired_'+trial.name)
        exec(compile(source.replace(old, new), '<author-only-kernel-repair>', 'exec'), repaired.__dict__)
        metrics = json.loads((trial/'verifier/metrics.json').read_text())
        eta = metrics['parameters']['viscosity']
        a, b = submitted.Model(), repaired.Model()
        a.viscosity = b.viscosity = eta
        row = {'source_sha256': sha(source_path), 'response_ast_identical_to_starter': True,
               'viscosity_held_fixed': eta, 'refit': False, 'repair': 'Replace only the distributed force kernel; retain the exact curved solute solve, exact Navier drag, readout and fitted viscosity.',
               'calibration_max_absolute_change': float(max(abs(a.predict(inputs)-b.predict(inputs)))), 'hidden': {}}
        for name, cases in reference.hidden_inputs().items():
            truth = reference.predict(cases)
            before = float(np.linalg.norm(a.predict(cases)-truth)/np.linalg.norm(truth))
            after = float(np.linalg.norm(b.predict(cases)-truth)/np.linalg.norm(truth))
            row['hidden'][name] = {'submitted_nrms': before, 'repaired_nrms': after,
                                   'recorded_metric_difference': abs(before-metrics['hidden'][name])}
            assert after < .04 and abs(before-metrics['hidden'][name]) < 1e-9
        assert row['calibration_max_absolute_change'] < 1e-13
        assert sha(source_path) == row['source_sha256']
        report['trials'][trial.name] = row
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print({name: max(x['repaired_nrms'] for x in row['hidden'].values()) for name, row in report['trials'].items()})


if __name__ == '__main__':
    main()
