import json
from pathlib import Path

import numpy as np
import pytest

from model import Model


DATA = Path(__file__).parent/'data'


@pytest.fixture(scope='module')
def errors():
    rows=json.loads((DATA/'calibration.json').read_text())
    groups=json.loads((DATA/'probabilities.json').read_text())
    model=Model()
    assert model.fit(rows) is model
    result={'rate': float(model.rate), 'max_probability_error': {}}
    for name,records in [('calibration',rows),*groups.items()]:
        y=model.predict([r['input'] for r in records])
        assert isinstance(y,np.ndarray) and y.shape==(len(records),) and np.isfinite(y).all()
        result['max_probability_error'][name]=float(max(abs(y-[r['value'] for r in records])))
    log=Path('/logs/verifier/metrics.json')
    if log.parent.exists():
        log.write_text(json.dumps(result,indent=2)+'\n')
    return result


def test_calibration(errors):
    assert abs(errors['rate']-1.061)<.0001
    assert errors['max_probability_error']['calibration']<2e-5


@pytest.mark.parametrize('group',['bands','locked','reversible_bands','reversible_lock'])
def test_joint(errors,group):
    assert errors['max_probability_error'][group]<.001
