from pathlib import Path
import json
import numpy as np
from model import Model


def test_rate_and_domain_order():
    rows=json.loads((Path(__file__).parent/"data/calibration.json").read_text())
    m=Model()
    assert m.fit(rows) is m and .09<=m.rate<=.24
    p=m.predict([r["input"] for r in rows])
    assert isinstance(p,np.ndarray) and p.dtype.kind in "ifu" and p.shape==(len(rows),)
    assert m.predict([]).shape==(0,) and max(abs(p-[r["value"] for r in rows]))<.000002
    tr=[[0,1,1,1,1],[1,-1,1,1,1],[0,2,1,1,1],[2,-1,1,1,1],[0,3,1,1,1],[3,-1,1,1,1]]
    q=dict(sites=4,drive=0,domains=[dict(weight=.4,tracks=tr),dict(weight=.6,tracks=[[0,1,1,2,1],*tr[1:]])],
           deficit=2,pickup=[[2,.5],[4,.75]],rounds=[0,3],delay=[-1,1])
    a,b=m.predict([q,dict(q,domains=q["domains"][::-1])])
    assert np.isfinite([a,b]).all() and abs(a-b)<.004
