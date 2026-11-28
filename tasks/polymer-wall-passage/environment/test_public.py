import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration():
    rows = json.loads((Path(__file__).parent / "data/calibration.json").read_text())
    m=Model()
    assert m.fit(rows) is m and .78 <= m.aspect <= 1.22
    pred=m.predict([r["input"] for r in rows])
    assert isinstance(pred,np.ndarray) and pred.dtype.kind in "ifu"
    assert pred.shape==(len(rows),) and m.predict([]).shape==(0,)
    assert max(abs(pred-[r["value"] for r in rows]))<.00002


def test_reflection():
    m=Model()
    m.aspect=1.
    q=dict(radius=2.25,dyes=[[-.42,.84],[.94,.66]],walls=[-.74,1.04],gate="inside")
    other=dict(q,dyes=[[.42,.84],[-.94,.66]],walls=[.74,-1.04])
    a,b=m.predict([q,other])
    assert abs(a-b)<.004
