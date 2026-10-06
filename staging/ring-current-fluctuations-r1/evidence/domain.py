"""Fixed domain check using fresh independent spatial starts and KKT refinement."""
from pathlib import Path
import hashlib
import json
import signal
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipk

from screen import branch_point, elliptic, optimize_profile
from polishing import polish, evaluate

ROOT = Path(__file__).resolve().parent


def main():
    plan = json.loads((ROOT/'domain-plan.json').read_text())
    polish_plan = json.loads((ROOT/'polishing-plan.json').read_text())
    for name, expected in plan['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    output = ROOT/'domain-report.json'
    assert not output.exists(), 'Never overwrite a scientific execution.'
    started = time.monotonic()
    report = dict(status='domain_running', spatial=[], raw_spatial=[], elliptic=[],
                  monotonicity=[], anchors=[], original_comparisons=[], comparisons=[],
                  convexity=[], gates={}, source_sha256={
                      p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__), ROOT/'domain-plan.json', ROOT/'screen.py',
                                ROOT/'polishing.py', ROOT/'polishing-plan.json']})

    def save():
        report['seconds'] = time.monotonic()-started
        output.write_text(json.dumps(report, indent=2)+'\n')

    def time_limit(signum, frame):
        report['status'] = 'time_limit_unresolved'
        save()
        raise TimeoutError('Fixed domain-study budget reached.')

    signal.signal(signal.SIGALRM, time_limit)
    signal.alarm(plan['budget_seconds'])
    save()
    for field in plan['map_fields']:
        upper = brentq(lambda m: 4*ellipk(m)-field, 0., 1-1e-12, xtol=5e-15)
        rows = [branch_point(field, m, 128) for m in np.linspace(0., upper, 201)]
        eta = np.array([r['eta'] for r in rows])
        report['monotonicity'].append(dict(field=field, rows=rows,
            maximum_increase=float(np.max(np.diff(eta))), endpoint_residual=float(eta[-1])))
        biases = np.linspace(-2*field-2, 2., 121)
        values = np.array([elliptic(field, bias, 128)['psi'] for bias in biases])
        report['convexity'].append(dict(field=field, biases=biases.tolist(),
            values=values.tolist(), normalized_minimum_second_difference=
            float(np.min(np.diff(values, n=2))/max(1., np.max(abs(values))))))
    settings = [(f, f*r, 'prediction') for f, r in plan['cases']]
    settings += [(f, b, 'control') for f, b in plan['controls']]
    assert len(settings) == plan['case_count']
    for field, bias, kind in settings:
        for order in plan['quadrature_orders']:
            report['elliptic'].append(dict(field=field, bias=bias, order=order, kind=kind,
                                          **elliptic(field, bias, order)))
        value = report['elliptic'][-1]['psi']
        mirror = elliptic(field, -2*field-bias, 128)['psi']
        report['original_comparisons'].append(dict(field=field, bias=bias,
                                                  symmetry_absolute=abs(value-mirror)))
        for n in plan['grids']:
            for seed in plan['starts']:
                raw = optimize_profile(field, bias, n, seed)
                report['raw_spatial'].append(raw)
                row = polish(raw, polish_plan)
                row['parent_row_index'] = len(report['raw_spatial'])-1
                report['spatial'].append(row)
                save()
                print(json.dumps({k: row[k] for k in ['field', 'bias', 'n', 'seed',
                    'success', 'polish_status', 'psi', 'normalized_tangent_gradient']}), flush=True)
    for field, bias in plan['exact_anchors']:
        value = elliptic(field, bias, 128)
        report['anchors'].append(dict(field=field, bias=bias, **value,
            absolute_error=abs(value['psi']-bias*(bias+2*field)/4)))
    evaluate(report, plan)
    report['gates']['convexity'] = all(r['normalized_minimum_second_difference'] >=
        -plan['convexity_tolerance'] for r in report['convexity'])
    numerical = all(v for k, v in report['gates'].items() if k != 'separation')
    report['status'] = ('numerical_unresolved' if not numerical else
                        'domain_pass' if report['gates']['separation'] else 'insufficient_separation')
    save()
    signal.alarm(0)
    print(json.dumps({k: report[k] for k in ['status', 'gates', 'group_source_errors', 'seconds']}, indent=2))


if __name__ == '__main__':
    main()
