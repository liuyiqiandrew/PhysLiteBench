from pathlib import Path
import json
import numpy as np
from model import Model


def test_rate():
    rows=json.loads((Path(__file__).parent/"data/calibration.json").read_text())
    m=Model()
    assert m.fit(rows) is m and .09<=m.rate<=.24
    p=m.predict([r["input"] for r in rows])
    assert isinstance(p,np.ndarray) and p.dtype.kind in "ifu" and p.shape==(len(rows),)
    assert m.predict([]).shape==(0,) and max(abs(p-[r["value"] for r in rows]))<.000002


def test_orientation_of_an_undirected_track():
    m=Model();m.rate=.15
    q=dict(sites=5,drive=0,tracks=[[0,1,1],[0,2,3],[0,3,2],[1,2,1],[2,3,2],[1,4,2],[3,4,1],[4,-1,2],[2,-1,1]],
           pickup=[[0,.25],[1,.5]],rounds=[0,8],delay=[-1.125,2.125],**{"class":"last"})
    r=dict(q,tracks=[[1,0,1],*q["tracks"][1:]],pickup=[[0,.75],[1,.5]])
    a,b=m.predict([q,r])
    assert np.isfinite([a,b]).all() and abs(a-b)<.004
