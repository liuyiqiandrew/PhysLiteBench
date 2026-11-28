import json
from pathlib import Path
import numpy as np
from model import Model


def test_calibration():
    rows = json.loads((Path(__file__).parent/"data/calibration.json").read_text())
    m = Model()
    assert m.fit(rows) is m and .78 <= m.aspect <= 1.22
    pred = m.predict([r["input"] for r in rows])
    assert isinstance(pred, np.ndarray) and pred.shape == (len(rows),)
    assert pred.dtype.kind in "ifu" and m.predict([]).shape == (0,)
    assert max(abs(pred-[r["value"] for r in rows])) < .00002


def test_reflection():
    m = Model()
    m.aspect = 1.
    q = dict(dyes=[[-.42,.84],[.94,.66]], walls=[-.74,1.34],gate="left")
    reflected = dict(dyes=[[.42,.84],[-.94,.66]],walls=[.74,-1.34],gate="right")
    assert abs(sum(m.predict([q,reflected]))-1) < .004
