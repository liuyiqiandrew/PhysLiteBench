from pathlib import Path
import json
import numpy as np
from scipy.special import roots_jacobi, iv
from scipy.optimize import brentq


def exact(b, wi, points=96):
    x,w=roots_jacobi(points,b/2,0)
    u=(x+1)/2
    z=np.dot(w,iv(0,wi*b*u))
    trace=b*np.dot(w,u*iv(0,wi*b*u))/z
    stress=b*np.dot(w,u/(1-u)*iv(1,wi*b*u))/z
    return trace,stress


def peterlin(b,wi):
    f=brentq(lambda f:1-1/f-(1/(f-2*wi)+1/(f+2*wi))/b,max(1.,2*abs(wi))+1e-12,100+2*abs(wi),xtol=1e-13)
    xx,yy=1/(f-2*wi),1/(f+2*wi)
    return xx+yy,f*(xx-yy)

rows=[]
for b in [2,3,4,5,6,8,12,20]:
    for wi in [.05,.15,.3,.45,.6,1.,1.5]:
        trace,stress=exact(b,wi)
        pt,ps=peterlin(b,wi)
        rows.append(dict(b=b,wi=wi,stress=stress,peterlin=ps,relative_error=ps/stress-1,moment_identity_error=abs(stress-2*wi*trace)))
Path(__file__).with_name('screen.json').write_text(json.dumps(rows,indent=2)+'\n')
for b in [2,3,4,5,6,8,12,20]:
    rr=[r for r in rows if r['b']==b]
    print(b,[(r['wi'],round(r['relative_error'],3)) for r in rr])
print('moment error',max(r['moment_identity_error'] for r in rows))
