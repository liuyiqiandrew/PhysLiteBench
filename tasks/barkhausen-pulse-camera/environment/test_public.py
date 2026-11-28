from pathlib import Path
import json
import numpy as np
from model import Model


def test_calibration_and_wall_exchange():
    rows=json.loads((Path(__file__).parent/"data/calibration.json").read_text())
    m=Model()
    assert m.fit(rows) is m and .72<=m.diffusivity<=1.12
    p=m.predict([r["input"] for r in rows])
    assert isinstance(p,np.ndarray) and p.dtype.kind in "ifu"
    assert p.shape==(len(rows),) and m.predict([]).shape==(0,)
    assert max(abs(p-[r["value"] for r in rows]))<.000002
    q=dict(time=1.25,size=1.13,walls=[[.48,.95,1.14],[.77,1.32,.84]],
           flashes=[[.42,1.2,.5],[.83,.46,1.8]],identity=0,trigger="last")
    rev=dict(q,walls=q["walls"][::-1],flashes=[[t,b,a] for t,a,b in q["flashes"]],identity=1)
    x,y=m.predict([q,rev])
    assert abs(x-y)<.004
