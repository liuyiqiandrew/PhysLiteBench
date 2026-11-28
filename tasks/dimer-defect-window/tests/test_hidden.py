import json
from pathlib import Path
import numpy as np
import pytest
from model import Model

DATA = Path(__file__).parent/'data'


@pytest.fixture(scope='module')
def errors():
    rows = json.loads((DATA/'calibration.json').read_text())
    m = Model()
    assert m.fit(rows) is m
    report = {'activity': float(m.activity)}
    groups = json.loads((DATA/'predictions.json').read_text())
    for label,items in [('calibration',rows),*groups.items()]:
        pred = m.predict([r['input'] for r in items])
        assert isinstance(pred,np.ndarray) and pred.shape==(len(items),)
        assert pred.dtype.kind in 'iuf' and np.isfinite(pred).all()
        report[label] = float(max(abs(pred-[r['value'] for r in items])))
    log = Path('/logs/verifier/metrics.json')
    if log.parent.is_dir():
        log.write_text(json.dumps(report,indent=2)+'\n')
    return report


def test_fit(errors):
    assert .6 <= errors['activity'] <= 1.4
    assert errors['calibration'] < 2e-5


@pytest.mark.parametrize('group',['clear','shared','veto','nesting'])
def test_contours(errors,group):
    assert errors[group] < .004
