"""Integrate finite measured apertures directly over the two original true first-line times."""
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre

dirs={'x':np.array([1.,0,0]),'y':np.array([0,1.,0]),'z':np.array([0,0,1.])}
ops={'x':np.array([[0,1],[1,0]]),'y':np.array([[0,-1j],[1j,0]]),'z':np.diag([1.,-1.])}
eps=np.array([[0.,1.],[-1.,0.]])/np.sqrt(2)

class Raw:
    """Use (a+b*x)e^-x for the common marked bond, with first-crossing flux a."""
    def __init__(self,q):
        self.t=np.r_[0.,np.cumsum([b['width'] for b in q['bands']])]
        self.init=(0.,1.) if q['start']=='renewed' else (1/3,2/3)
        self.tab=[]
        for b in q['bands']:
            ends=[]
            for seq in b['arms']:
                u=[np.eye(2),np.eye(2)]
                for side,d,angle in seq:
                    j=int(side=='R');u[j]=(np.cos(angle/2)*np.eye(2)-1j*np.sin(angle/2)*ops[d])@u[j]
                ends.append(u[0]@eps@u[1].T)
            gain=np.array([1.,np.vdot(*ends)])
            eff=[];n=[]
            for e,d,t in b['detectors']:
                v=dirs[d];z=dirs['z'];eff.append(e)
                n.append(z*np.cos(t)-np.cross(v,z)*np.sin(t)+v*(v@z)*(1-np.cos(t)))
            p11=eff[0]*eff[1]*np.sum((n[0]-n[1])**2)/8
            p10=eff[0]/2-p11;p01=eff[1]/2-p11
            p=[1-p10-p01-p11,p10,p01,p11]
            table={}
            for active in range(4):
                for trigger in range(4):
                    weight=gain*sum(p[w] for w in range(4) if w&active==trigger)
                    table[active,trigger]=weight if trigger else (weight,np.sqrt(5+4.*weight))
            self.tab.append(table)

    def raw(self,left,right,pair):
        active=3
        actual=[(left,3)] if pair else sorted([(left,1),(right,2)])
        a=np.full(2,self.init[0],complex);b=np.full(2,self.init[1],complex)
        j=0;clock=0.
        for band,table in enumerate(self.tab):
            end=self.t[band+1];inside=[]
            while j<len(actual) and actual[j][0]<=end:inside.append(actual[j]);j+=1
            for stop,trigger in [*inside,(end,0)]:
                dt=stop-clock
                if dt:
                    weight,d=table[active,0]
                    co=np.cosh(d*dt/2);si=2*np.sinh(d*dt/2)/d;f=np.exp(-1.5*dt)
                    a,b=(f*(co*a+si*(-.5*a+b)),f*(co*b+si*((1+weight)*a+.5*b)))
                if trigger:
                    b=table[active,trigger]*a;a=np.zeros(2,complex);active&=~trigger
                clock=stop
        return a+b

@lru_cache(None)
def nodes(n):return roots_legendre(n)

def integrate(f,a,b,n=5):
    x,w=nodes(n)
    return (b-a)/2*sum(weight*f((b-a)*val/2+(a+b)/2) for val,weight in zip(x,w))

def partition(points,l,r):
    cuts=sorted({float(z) for z in points if l<z<r}|{l,r})
    return zip(cuts,cuts[1:])

def gate(q,src,h,degree=5):
    c=q['center'];T=src.t[-1];c0,c1=c-h/2,c+h/2
    lower,upper=q['slit'];point=lower==upper
    if point:lower,upper=(max(0.,lower-h/2) if q['kind']=='beat' else lower-h/2),upper+h/2
    measured=[(lower,upper)] if q['kind']=='signed' else ([(lower,upper),(-upper,-lower)] if lower>0 else [(-upper,upper)])
    p,k=q['lock'];lo,hi=q['dither'];omega=q['lag']
    def admit(d):
        v=np.zeros(2,complex)
        for A,B in measured:
            if A<=d+k<=B:v+=p*np.array([1.,np.exp(-1j*omega*k)])
            x,y=max(lo,A-d),min(hi,B-d)
            if x<y:
                amount=(1-p)*(y-x)/(hi-lo)
                e=np.exp(-.5j*omega*(x+y))*np.sinc(omega*(y-x)/(2*np.pi))
                v+=amount*np.array([1.,e])
        return v
    # A genuine double hit has one true coordinate, hence one integral.
    shared=np.zeros(2,complex)
    for x,y in partition(src.t,c0,c1):shared+=integrate(lambda t:src.raw(t,t,True)*admit(0.),x,y,degree)
    # Two distinct hits occupy an ordinary rectangle in (left time,right time). Integrate
    # the right-time slice admitted by the center comparator for every physical left time.
    kinks={v-z for pair in measured for v in pair for z in (lo,hi,k)}
    left_knots=set(src.t)|{c0,c1}
    for u in (c0,c1):
        left_knots.update(2*u-v for v in src.t)
        left_knots.update(u+d for d in kinks)
    left_knots.update(v+2*d for v in src.t for d in kinks)
    def inner(left):
        a=max(0.,2*c0-left);b=min(T,2*c1-left)
        if a>=b:return np.zeros(2,complex)
        out=np.zeros(2,complex)
        for x,y in partition([*src.t,left,*(left-2*d for d in kinks)],a,b):
            if admit((left-(x+y)/2)/2)[0].real==0:continue
            out+=integrate(lambda right:src.raw(left,right,False)*admit((left-right)/2),x,y,degree)
        return out
    area=np.zeros(2,complex)
    for x,y in partition(left_knots,0.,T):area+=integrate(inner,x,y,degree)
    val=(shared+area)/h
    at_atom=point and p>0 and (q['slit'][0]==k if q['kind']=='signed' else q['slit'][0]==abs(k)) and src.raw(c,c,True)[0].real>1e-15
    if point and not at_atom:val/=h
    return val

def truth(q,width=.006,degree=5):
    if q.get('calibration'):return 1.,np.cos(q['phase'])
    src=Raw(q);a,b,c=[gate(q,src,width/k,degree) for k in (1,2,4)]
    p,z=(a-6*b+8*c)/3
    return p.real,(np.exp(-1j*q['phase'])*z).real/p.real
