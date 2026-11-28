"""A singlet, two gap-resolved spin analyzers, and a stationary Poisson spectrum."""
import numpy as np

axis={'x':np.array([[0,1],[1,0]],complex),'y':np.array([[0,-1j],[1j,0]]),'z':np.diag([1.,-1.])}
singlet=np.array([0.,1.,-1.,0.])/np.sqrt(2.)

def rotate(d,angle):return np.cos(angle/2)*np.eye(2)-1j*np.sin(angle/2)*axis[d]

def optics(spec,geometry,s,independent=False):
    states=[]
    for controls in spec['arms']:
        ket=singlet
        for side,d,u,v in controls:
            f=s if side=='L' else -s
            R=rotate(d,u+f*v)
            ket=(np.kron(R,np.eye(2)) if side=='L' else np.kron(np.eye(2),R))@ket
        states.append(ket)
    p=[];projectors=[]
    for i,(eff,d,u,v) in enumerate(spec['detectors']):
        rotation=rotate(d,u+(s if i==0 else -s)*v)
        projector=rotation.conj().T@np.diag([1.,0.])@rotation
        projectors.append(projector);p.append(eff/2 if geometry in (('left','both') if i==0 else ('right','both')) else 0.)
    joint=(p[0]*p[1] if independent else
           spec['detectors'][0][0]*spec['detectors'][1][0]*np.vdot(singlet,np.kron(*projectors)@singlet).real)
    if geometry!='both':joint=0.
    probs=np.array([1-p[0]-p[1]+joint,p[0]-joint,p[1]-joint,joint])
    return np.vdot(*states),probs

def mask(records,t,hit):
    """Probability weight for each possible click pattern at one instant."""
    out=np.ones(4)
    for j,mark in enumerate(records):
        bit=(np.arange(4)>>j)&1
        if mark is not None and hit and mark==t:out*=bit
        elif mark is None or mark>t:out*=1-bit
    return out

def poisson(q):
    ends=np.r_[0.,np.cumsum([b['width'] for b in q['bands']])]
    marks={x for x in q['lines'] if x is not None}
    def run(signal):
        total=1.+0j
        for i,b in enumerate(q['bands']):
            data=[optics(b[g],g,s) for g in ('left','right','both') for s in (-1,1)]
            start,stop=ends[i:i+2]
            points=[start,*sorted(t for t in marks if start<t<stop),stop]
            for l,r in zip(points,points[1:]):
                keep=mask(q['lines'],(l+r)/2,False)
                c=sum((m if signal else 1)*(p@keep)-1 for m,p in data)/12
                total*=np.exp((r-l)*c)
                if r in marks:
                    keep=mask(q['lines'],r,True)
                    total*=sum((m if signal else 1)*(p@keep) for m,p in data)/12
        return total
    denominator=run(False).real
    return 0. if denominator<=1e-14 else (np.exp(-1j*q['phase'])*run(True)).real/denominator

class Readout:
    def fit(self,records):
        f=np.cos([r['input']['phase'] for r in records]);y=[r['value'] for r in records]
        self.visibility=float(np.clip(f@y/(f@f),.65,.95));return self
    def predict(self,inputs):
        return self.visibility*np.array([self.fringe(q) for q in inputs],float)
