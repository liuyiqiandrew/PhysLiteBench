"""Validate fixed data, both controls, independent gold and 256 noise draws."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/ring-current-fluctuations'
EVIDENCE_ROOT = ROOT/'staging/ring-current-fluctuations-r1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coefficient(experiments):
    field = np.array([e['field'] for e in experiments])
    bias = np.array([e['bias'] for e in experiments])
    return bias*(bias+2*field)/4


def hidden_errors(module, diffusivity, reference):
    errors = {}
    for name, experiments in reference.hidden_inputs().items():
        truth = reference.predict(experiments, reference.TRUE_PARAMETER)
        values = module.predict_at(experiments, diffusivity)
        errors[name] = float(np.linalg.norm(values-truth)/np.linalg.norm(truth))
    return errors


def metrics(module, records, reference):
    model = module.Model().fit(records)
    predictions = model.predict([r['input'] for r in records])
    residual = (predictions-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    return dict(diffusivity=model.diffusivity,
                parameter_relative_error=abs(model.diffusivity/reference.TRUE_PARAMETER-1),
                calibration_chi2=float(residual@residual)/(len(records)-1),
                hidden=hidden_errors(module, model.diffusivity, reference))


def ast_functions(path):
    return {n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def approved_elliptic_ast():
    text = (EVIDENCE_ROOT/'evidence/screen.py').read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'elliptic')
    text = ast.get_source_segment(text, node)
    text = text.replace("        m = brentq(lambda m: branch_point(field, m, order)['eta']-eta,\n"
                        "                   0., upper, xtol=5e-15)",
                        "        def residual_at(m):\n"
                        "            return -eta if m == upper else branch_point(field, m, order)['eta']-eta\n"
                        "        m = brentq(residual_at, 0., upper, xtol=5e-15)")
    text = text.replace("    result = branch_point(field, m, order)\n    result.update",
                        "    result = branch_point(field, m, order)\n"
                        "    if m == upper:\n        result['eta'] = 0.\n    result.update")
    return ast.dump(ast.parse(text).body[0], include_attributes=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--recompute-reference', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    attempt = ROOT/'results'/('ring-current-validation-'+stamp)
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='running', attempt=attempt.name, gates={}, controls={}, noise=[],
                  versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__))

    def save():
        report['seconds'] = time.monotonic()-started
        (attempt/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    try:
        plan = json.loads((EVIDENCE_ROOT/'packaging-plan.json').read_text())
        calibration = plan['calibration']
        experiments = [dict(field=float(field), bias=float(bias))
                       for field in calibration['fields']
                       for bias in [-field-.75, .5, 1.5]
                       for _ in range(calibration['replicates_per_setting'])]
        mean = calibration['true_diffusivity']*coefficient(experiments)
        sigma = calibration['sigma']
        public = TASK/'environment/data/calibration.json'
        private = TASK/'tests/data/calibration.json'
        if args.regenerate:
            rng = np.random.default_rng(calibration['data_seed'])
            records = [dict(input=e, value=float(y), sigma=sigma)
                       for e, y in zip(experiments, mean+rng.normal(0, sigma, len(mean)))]
            text = json.dumps(records, indent=2)+'\n'
            for path in [public, private]:
                if path.exists():
                    assert path.read_text() == text, 'Fixed data would change; preserve and review.'
                else:
                    path.write_text(text)
        assert public.read_bytes() == private.read_bytes()
        records = json.loads(public.read_text())
        assert len(records) == len(experiments) == 192
        assert [r['input'] for r in records] == experiments
        assert all(r['sigma'] == sigma for r in records)
        report['calibration_sha256'] = hashlib.sha256(public.read_bytes()).hexdigest()
        report['gates']['fixed_calibration'] = True
        oracle = load('ring_oracle', TASK/'solution/model.py')
        shortcut = load('ring_shortcut', ROOT/'scripts/ring_current_fluctuations_baseline.py')
        reference = load('ring_reference', TASK/'tests/reference.py')
        source = ast_functions(EVIDENCE_ROOT/'evidence/screen.py')
        polishing = ast_functions(EVIDENCE_ROOT/'evidence/polishing.py')
        physical = ast_functions(TASK/'solution/model.py')
        independent = ast_functions(TASK/'tests/reference.py')
        transfer = {name: source[name] == physical[name]
                    for name in ['quadrature', 'branch_point', 'elliptic']}
        transfer.update({name: source[name] == independent[name]
                         for name in ['functional', 'optimize_profile']})
        transfer.update({name: polishing[name] == independent[name]
                         for name in ['state', 'cost_hessian', 'polish']})
        report['source_transfer'] = transfer
        report['approved_elliptic_endpoint_difference'] = physical['elliptic'] == approved_elliptic_ast()
        report['gates']['source_transfer'] = (all(v for k, v in transfer.items() if k != 'elliptic')
                                             and report['approved_elliptic_endpoint_difference'])
        endpoint_rows = []
        for field in plan['endpoint_correction']['fields']:
            center = oracle.elliptic(field, -field, 128)['psi']
            for eta in plan['endpoint_correction']['eta']:
                for sign in [-1, 1]:
                    value = oracle.elliptic(field, -field+sign*eta, 128)
                    endpoint_rows.append(dict(field=field, eta=sign*eta, psi=value['psi'],
                        root_residual=value['residual'], relative_center_change=abs(value['psi']-center)/max(1., abs(center))))
        report['endpoint_checks'] = endpoint_rows
        report['gates']['endpoint_limit'] = all(np.isfinite(x['psi']) and x['relative_center_change'] < 1e-7
                                               for x in endpoint_rows)
        report['gates']['exact_calibration_law'] = bool(np.max(abs(oracle.predict_at(experiments, reference.TRUE_PARAMETER)-mean)) < 1e-12)
        for name, module in [('oracle', oracle), ('shortcut', shortcut)]:
            row = metrics(module, records, reference)
            report['controls'][name] = row
            report['gates'][name+'_calibration'] = row['calibration_chi2'] < 1.5 and row['parameter_relative_error'] < .03
            report['gates'][name+'_predictions'] = (all(x < .04 for x in row['hidden'].values()) if name == 'oracle'
                else row['hidden']['field10'] > .04 and row['hidden']['field12'] > .04 and row['hidden']['exact_limits'] < .04)
        save()
        if args.recompute_reference:
            result = reference.reproduce()
            (attempt/'reference-reproduction.json').write_text(json.dumps(result, indent=2)+'\n')
            errors = []
            valid = len(result['raw']) == len(result['refined']) == 36
            for row in result['comparisons']:
                if None in row['values']:
                    valid = False
                    continue
                errors.append(abs(row['values'][-1]-row['expected'])/max(1., abs(row['expected'])))
                valid &= errors[-1] < .002
                valid &= abs(row['values'][1]-row['values'][0])/max(1., abs(row['expected'])) < .002
            report['reference_reproduction'] = dict(profiles=len(result['refined']),
                max_relative_error=max(errors) if errors else None)
            report['gates']['reference_reproduction'] = bool(valid)
            save()
        rng = np.random.default_rng(calibration['noise_seed'])
        for index in range(calibration['noise_realizations']):
            sample = [dict(input=e, value=float(y), sigma=sigma)
                      for e, y in zip(experiments, mean+rng.normal(0, sigma, len(mean)))]
            report['noise'].append(dict(index=index, oracle=metrics(oracle, sample, reference),
                                         shortcut=metrics(shortcut, sample, reference)))
            if (index+1) % 16 == 0:
                save()
        report['gates']['noise'] = len(report['noise']) == 256 and all(
            x['oracle']['calibration_chi2'] < 1.5 and x['shortcut']['calibration_chi2'] < 1.5
            and x['oracle']['parameter_relative_error'] < .03 and x['shortcut']['parameter_relative_error'] < .03
            and all(v < .04 for v in x['oracle']['hidden'].values())
            and x['shortcut']['hidden']['field10'] > .04 and x['shortcut']['hidden']['field12'] > .04
            and x['shortcut']['hidden']['exact_limits'] < .04 for x in report['noise'])
        report['noise_summary'] = dict(realizations=len(report['noise']),
            min_diffusivity=min(x['oracle']['diffusivity'] for x in report['noise']),
            max_diffusivity=max(x['oracle']['diffusivity'] for x in report['noise']),
            max_chi2=max(x['oracle']['calibration_chi2'] for x in report['noise']),
            max_oracle_prediction=max(v for x in report['noise'] for v in x['oracle']['hidden'].values()),
            min_shortcut_distinguishing_prediction=min(x['shortcut']['hidden'][k] for x in report['noise'] for k in ['field10', 'field12']))
        save()
        report['local_pytest'] = {}
        for name, source_path in [('oracle', TASK/'solution/model.py'),
                                  ('shortcut', ROOT/'scripts/ring_current_fluctuations_baseline.py')]:
            with tempfile.TemporaryDirectory(prefix='ring-current-control-') as directory:
                app = Path(directory)
                shutil.copyfile(source_path, app/'model.py')
                shutil.copyfile(TASK/'environment/test_public.py', app/'test_public.py')
                shutil.copytree(TASK/'environment/data', app/'data')
                before = time.monotonic()
                run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--import-mode=importlib',
                                      str(app/'test_public.py'), str(TASK/'tests/test_hidden.py')],
                                     cwd=app, env=dict(os.environ, PYTHONPATH=str(app)),
                                     capture_output=True, text=True, timeout=60)
                (attempt/('local-'+name+'.txt')).write_text(run.stdout+run.stderr)
                report['local_pytest'][name] = dict(exit_code=run.returncode, seconds=time.monotonic()-before)
                report['gates']['pytest_'+name] = run.returncode == (0 if name == 'oracle' else 1)
        payloads = list(TASK.rglob('*.py')) + [public, private, TASK/'tests/data/hidden.json', Path(__file__),
                                            EVIDENCE_ROOT/'packaging-plan.json', ROOT/'scripts/ring_current_fluctuations_baseline.py']
        report['source_sha256'] = {str(x.relative_to(ROOT)): hashlib.sha256(x.read_bytes()).hexdigest() for x in payloads}
        report['status'] = 'all_local_validation_gates_pass' if all(report['gates'].values()) else 'validation_failed'
        save()
    except BaseException as error:
        report.update(status='validation_exception', exception=repr(error))
        (attempt/'exception.txt').write_text(traceback.format_exc())
        save()
        shutil.copyfile(attempt/'report.json', ROOT/'results/ring-current-validation.json')
        raise
    shutil.copyfile(attempt/'report.json', ROOT/'results/ring-current-validation.json')
    print(json.dumps({k: report[k] for k in ['status', 'attempt', 'seconds', 'gates', 'controls', 'noise_summary', 'local_pytest']}, indent=2))
    if report['status'] != 'all_local_validation_gates_pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
