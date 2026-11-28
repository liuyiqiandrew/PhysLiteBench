import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration_and_beam_order():
    rows=json.loads((Path(__file__).parent/"data/calibration.json").read_text())
    m=Model()
    assert m.fit(rows) is m and .7<=m.visibility<=.95
    pred=m.predict([r["input"] for r in rows])
    assert isinstance(pred,np.ndarray) and pred.dtype.kind in "ifu"
    assert pred.shape==(len(rows),) and m.predict([]).shape==(0,)
    assert max(abs(pred-[r["value"] for r in rows]))<.000002
    q=dict(length=10,beta=1.4,mu=.2,flip=[.15,.22],moves=[[5,1],[6,3]],
           rotations=[[.8,-.3,1.1],[-1.1,.9,-.7]],plate=[.65,.4,.3],phase=.3)
    reverse=dict(q,moves=q["moves"][::-1],rotations=q["rotations"][::-1])
    a,b=m.predict([q,reverse])
    assert abs(a-b)<.004
