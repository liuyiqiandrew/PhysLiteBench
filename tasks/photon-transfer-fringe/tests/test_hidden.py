import json
from pathlib import Path

import numpy as np
import pytest
from model import Model

DATA = Path(__file__).parent/'data'


@pytest.fixture(scope='module')
def metrics():
    calibration = json.loads((DATA/'calibration.json').read_text())
    readings = json.loads((DATA/'predictions.json').read_text())
    model = Model()
    assert model.fit(calibration) is model
    result = {'visibility': float(model.visibility), 'max_absolute_error': {}}
    for name,rows in [('calibration',calibration),*readings.items()]:
        pred = model.predict([row['input'] for row in rows])
        assert isinstance(pred,np.ndarray) and pred.shape == (len(rows),)
        assert pred.dtype.kind in 'iuf' and np.isfinite(pred).all()
        result['max_absolute_error'][name] = float(max(abs(pred-[r['value'] for r in rows])))
    logs = Path('/logs/verifier')
    if logs.is_dir():
        (logs/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def test_fit(metrics):
    assert .65 <= metrics['visibility'] <= .95
    assert metrics['max_absolute_error']['calibration'] < .00002


@pytest.mark.parametrize('family',['mixing','squeezed','even_gate','odd_gate',
                                   'ordered_clicks','erased_order','subtraction_port','finite_pulses','finite_eraser','pump_number','pump_erasure'])
def test_transfer(metrics,family):
    assert metrics['max_absolute_error'][family] < .001
