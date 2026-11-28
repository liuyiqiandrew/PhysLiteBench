from pathlib import Path
import json
import numpy as np
from model import Model


def test_calibration_and_reflection():
    rows=json.loads((Path(__file__).parent/"data/calibration.json").read_text())
    m=Model()
    assert m.fit(rows) is m and .38<=m.diffusivity<=.70
    p=m.predict([r["input"] for r in rows])
    assert isinstance(p,np.ndarray) and p.dtype.kind in "ifu" and p.shape==(len(rows),)
    assert m.predict([]).shape==(0,) and max(abs(p-[r["value"] for r in rows]))<.000002
    q=dict(length=3.,walls=[[1.25,.25],[1.18,.48]],sizes=[2.24,2.20],
           gates=[[1.35,"forward",.25,.65],[1.74,"reverse",.30,.78]])
    rev=dict(q,walls=q["walls"][::-1],sizes=q["sizes"][::-1],
             gates=[[1.65,"reverse",.25,.65],[1.26,"forward",.30,.78]])
    a,b=m.predict([q,rev])
    assert abs(a-b)<.004
