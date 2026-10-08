import json
from pathlib import Path
import numpy as np
from model import Model


def records():
    return json.loads(Path('/app/data/calibration.json').read_text())


def test_fit_and_calibration():
    data=records()
    model=Model()
    assert model.width is None
    assert model.fit(iter(data)) is model
    predicted=np.asarray(model.predict_calibration(data),dtype=float)
    assert predicted.shape==(192,) and np.all(np.isfinite(predicted))
    assert .8<=model.width<=1.2
    measured=np.asarray([r['value'] for r in data])
    sigma=np.asarray([r['sigma'] for r in data])
    assert np.mean(((predicted-measured)/sigma)**2)<1.5


def test_iterables_and_overrides():
    model=Model().fit(records())
    rows=[{'beta':1.5,'replicas':2,'observable':'parent_collision'},
          {'beta':4.,'replicas':3,'observable':'state_collision'},
          {'beta':3.2,'replicas':4,'observable':'leaf_partition','partition':[2,2]}]
    expected=np.asarray(model.predict(rows))
    assert expected.shape==(3,) and np.all(np.isfinite(expected))
    assert np.array_equal(expected,model.predict(iter(rows)))
    assert np.array_equal(expected[::-1],model.predict(rows[::-1]))
    assert np.array_equal(expected,model.predict(rows,width=model.width))
    stored=model.width
    assert model.predict(rows,width=.8).shape==(3,)
    assert model.width==stored
    assert model.predict([]).shape==(0,)
    assert model.predict_calibration(iter(())).shape==(0,)
    assert Model().predict([],width=1.).shape==(0,)
