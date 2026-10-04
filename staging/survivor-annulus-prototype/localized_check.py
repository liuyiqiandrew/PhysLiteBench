"""One limited check of localized positive wall capture, with ordinary flows."""
from pathlib import Path
import datetime
import hashlib
import itertools
import json
import time
import numpy as np
from scipy.sparse import diags
from prototype import operator, principal, moments

HERE=Path(__file__).resolve().parent


def localized(d, inner, rotation, high, sharpness, hole=False, nr=24, nt=64, reference=False):
    outer=1.; low=.02
    a,vol,u=operator(d,inner,outer,rotation,low,low,nr,nt)
    dr=(outer-inner)/nr
    cellarea=(outer**2-(outer-dr)**2)/2
    theta=(np.arange(nt)+.5)*2*np.pi/nt
    spot=np.exp(sharpness*(np.cos(theta)-1))
    pattern=1-spot if hole else spot
    capture=low+(high-low)*pattern
    killing=outer*capture/(1+capture*dr/(2*d))/cellarea
    base=outer*low/(1+low*dr/(2*d))/cellarea
    change=np.zeros(nr*nt);change[-nt:]=base-killing
    a=(a+diags(change)).tocsc()
    val,right,res=principal(a,True);lv,left,lres=principal(a.T.tocsc(),True)
    physical=float((left*right)@u/(left@right))
    source=float((right*right)@u/(right@right))
    out={'loss_rate':-val,'physical_drift':physical,'source_drift':source,
         'fractional_error':abs(source/physical-1),'residual':max(res,lres)}
    if reference:
        def diff(h):
            return (principal(a+diags(h*u+d*h*h))-principal(a+diags(-h*u+d*h*h)))/(2*h)
        out['tilted_drift']=(4*diff(.02)-diff(.04))/3
        out['tilted_relative_error']=abs(out['tilted_drift']/physical-1)
    return out


def main():
    start=time.monotonic();rows=[]
    for inner,rotation,high,sharpness,hole in itertools.product(
            [.15,.35,.6],[4.,16.,64.],[20.,200.],[4.,16.],[False,True]):
        controls=dict(d=1.,inner=inner,rotation=rotation,high=high,sharpness=sharpness,hole=hole)
        rows.append({'controls':controls,**localized(**controls)})
    top=sorted(rows,key=lambda x:x['fractional_error'],reverse=True)[:3]
    refinements=[]
    for row in top:
        controls=row['controls'];levels=[]
        for n,m in [(24,64),(48,128),(72,192)]:
            levels.append({'grid':[n,m],**localized(**controls,nr=n,nt=m,reference=True)})
        refinements.append({'controls':controls,'levels':levels})
    anchor=[]
    for rot,klo,khi in [(0.,.02,20.),(16.,2.,2.),(-16.,2.,2.)]:
        anchor.append({'rotation':rot,'capture_low':klo,'capture_high':khi,
                       **moments(1.,.35,1.,rot,klo,khi,nr=32,nt=64,reference=True)})
    result={'status':'prototype_only','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'development_timing':'Developed before any survivor-transport-r1 outcome; not a new family or agent trial.',
            'screen_count':len(rows),'rows':rows,'top_refinements':refinements,'shared_calibration_anchors':anchor,
            'seconds':time.monotonic()-start,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'prototype.py',Path(__file__)]}}
    (HERE/'localized-report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'top_refinements':refinements,'anchors':anchor,'seconds':result['seconds']},indent=2))


if __name__=='__main__':main()
