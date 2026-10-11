"""Freeze a task's hidden-prediction truth so the verifier does not recompute it.

The hidden tests used to evaluate tests/reference.py at the true parameter on every
verifier run, which used most of the 60-second budget for some tasks. This writes
tests/data/hidden_truth.json with each hidden group's inputs and reference values,
computed with the same call the hidden test made. --check recomputes and compares.

    python scripts/freeze_hidden_truth.py tasks/superconducting-heat
    python scripts/freeze_hidden_truth.py tasks/superconducting-heat --check
"""
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

OUTPUT = 'data/hidden_truth.json'


def canonical(value):
    """JSON form of hidden inputs, as stored in hidden_truth.json."""
    return json.loads(json.dumps(value, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x))


def same_inputs(a, b):
    """Equal structure and strings; numbers equal up to last-bit differences between CPUs."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_inputs(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(same_inputs, a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-300)
    return a == b


def load_reference(task):
    tests = task/'tests'
    sys.path.insert(0, str(tests))
    spec = importlib.util.spec_from_file_location('physical_reference', tests/'reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute(task):
    reference = load_reference(task)
    groups = {}
    for name, experiments in reference.hidden_inputs().items():
        truth = np.asarray(reference.predict(experiments, reference.TRUE_PARAMETER), dtype=float)
        assert truth.ndim == 1 and np.isfinite(truth).all(), name
        groups[name] = {'inputs': canonical(experiments), 'truth': truth.tolist()}
    return {
        'description': 'Reference values of tests/reference.py predict(hidden_inputs, TRUE_PARAMETER), '
                       'frozen by scripts/freeze_hidden_truth.py; regenerate after changing reference.py.',
        'reference_sha256': hashlib.sha256((task/'tests/reference.py').read_bytes()).hexdigest(),
        'true_parameter': reference.TRUE_PARAMETER,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'machine': platform.machine()},
        'groups': groups,
    }


def check(task, frozen, tolerance=1e-7):
    """Return the largest truth difference relative to its group's RMS, the normalization the
    hidden tests grade with; raise if it exceeds tolerance or the inputs or reference changed."""
    fresh = compute(task)
    assert frozen['reference_sha256'] == fresh['reference_sha256'], 'reference.py changed since freezing'
    assert frozen['true_parameter'] == fresh['true_parameter']
    assert list(frozen['groups']) == list(fresh['groups']), 'hidden group names or order changed'
    worst = 0.
    for name, group in fresh['groups'].items():
        assert same_inputs(frozen['groups'][name]['inputs'], group['inputs']), f'{name}: hidden inputs changed'
        old, new = np.array(frozen['groups'][name]['truth']), np.array(group['truth'])
        assert old.shape == new.shape, name
        worst = max(worst, float(np.max(np.abs(old-new))/np.sqrt(np.mean(new**2))))
    assert worst < tolerance, f'frozen truth differs from reference by {worst:.3g} of the group RMS'
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('task', type=Path, help='Task directory, e.g. tasks/superconducting-heat')
    parser.add_argument('--check', action='store_true', help='Recompute and compare instead of writing')
    args = parser.parse_args()
    task = args.task.resolve()
    path = task/'tests'/OUTPUT
    if args.check:
        worst = check(task, json.loads(path.read_text()))
        print(f'{task.name}: frozen truth matches reference (max difference {worst:.3g} of group RMS)')
    else:
        path.write_text(json.dumps(compute(task), indent=1)+'\n')
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
