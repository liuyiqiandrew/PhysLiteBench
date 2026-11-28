"""Project finite-resolution energy windows and independently shrink them to a line or joint density."""
import numpy as np
from scipy.integrate import solve_ivp

axes={'x':np.array([1.,0,0]),'y':np.array([0,1.,0]),'z':np.array([0,0,1.])}
pauli={'x':np.array([[0,1],[1,0]]),'y':np.array([[0,-1j],[1j,0]]),'z':np.diag([1.,-1.])}
eps=np.array([[0.,1.],[-1.,0.]])/np.sqrt(2.)

def response(spec,name,sign):
    ends=[]
    for sequence in spec['arms']:
        u=[np.eye(2),np.eye(2)]
        for side,d,a,v in sequence:
            which=int(side=='R');theta=(a+(-1)**which*sign*v)/2
            u[which]=(np.cos(theta)*np.eye(2)-1j*np.sin(theta)*pauli[d])@u[which]
        ends.append(u[0]@eps@u[1].T)
    directions=[];eff=[]
    for i,(e,d,a,v) in enumerate(spec['detectors']):
        theta=a+(-1)**i*sign*v;axis=axes[d];z=axes['z']
        directions.append(z*np.cos(theta)-np.cross(axis,z)*np.sin(theta)+axis*(axis@z)*(1-np.cos(theta)))
        eff.append(e if name=='both' or name==('left','right')[i] else 0.)
    both=eff[0]*eff[1]*(1-directions[0]@directions[1])/4
    p=np.array([1-sum(eff)/2+both,eff[0]/2-both,eff[1]/2-both,both])
    return np.vdot(*ends),p


def window(q,h):
    widths=[b['width'] for b in q['bands']];end=sum(widths);edges=np.r_[0.,np.cumsum(widths)]
    lines=q['lines'];windows=[];breaks=set(edges)
    for t in lines:
        if t is None:windows.append(None)
        else:
            left=min(max(t-h/2,0.),end-h);bounds=(left,left+h)
            windows.append(bounds);breaks.update(bounds)
    times=sorted(breaks)
    # State includes two optical channels, exact flags for which first clicks have already
    # occurred, the sign, and continuum coefficients for the common-cut density.
    y=np.zeros((2,4,2,7),complex)
    for i,s in enumerate((1,-1)):
        val=.5 if q['left_sign'] is None else float(q['left_sign']==s)
        y[:,0,i,{0:6,1:1,2:0}[q['central']]]=val
    optics=[{(g,s):response(b[g],g,s) for g in ('left','right','both') for s in (1,-1)} for b in q['bands']]
    for l,r in zip(times,times[1:]):
        if r-l<1e-14:continue
        mid=(l+r)/2;j=np.searchsorted(edges[1:],mid)
        tab=optics[j];shape=y.shape
        operators={}
        for g in ('left','right','both'):
            for s in (1,-1):
                m,p=tab[g,s]
                flags=np.zeros((4,4))
                for old in range(4):
                    for d in range(4):
                        fresh=d&~old
                        if all(not (fresh & 1<<i) or w is not None and w[0]<=mid<w[1]
                               for i,w in enumerate(windows)):
                            flags[old|d,old]+=p[d]
                operators[g,s]=[flags,m*flags]
        def rhs(t,raw):
            y=raw.reshape(shape);dy=np.zeros_like(y);k=np.exp(t)
            dy[:,:,:,0]=-(2+k)*y[:,:,:,0]
            dy[:,:,:,1]=-2*y[:,:,:,1]+k*(y[:,:,:,2]+y[:,:,:,3])
            dy[:,:,:,2]=-(1+k)*y[:,:,:,2];dy[:,:,:,3]=-(1+k)*y[:,:,:,3]
            dy[:,:,:,4]=-k*y[:,:,:,4]
            dy[:,:,:,5]=y[:,:,:,6]-2*y[:,:,:,5]
            dy[:,:,:,6]=y[:,:,:,5]-y[:,:,:,6]+k*(y[:,:,:,0]+y[:,:,:,4])
            def move(old,new,source,dest,g,s):
                flags=operators[g,s]
                for channel in (0,1):dy[channel,:,dest,new]+=flags[channel]@y[channel,:,source,old]
            for i,s in enumerate((1,-1)):
                move(0,2,i,1-i,'left',s);move(0,3,i,i,'right',s)
                move(2,4,i,i,'right',-s);move(3,4,i,1-i,'left',s)
                move(1,6,i,1-i,'left',s);move(1,6,i,i,'right',-s);move(5,6,i,1-i,'both',s)
            return dy.ravel()
        sol=solve_ivp(rhs,(l,r),y.ravel(),rtol=3e-11,atol=2e-14,method='DOP853');y=sol.y[:,-1].reshape(shape)
    flag=sum(1<<i for i,v in enumerate(lines) if v is not None)
    return y[:,flag].sum(axis=(1,2))/h**len({v for v in lines if v is not None})


def truth(q,width=.006):
    if not q['bands']:return 1.,np.cos(q['phase'])
    edges=np.r_[0.,np.cumsum([band['width'] for band in q['bands']])]
    knots=sorted(set(edges)|{line for line in q['lines'] if line is not None})
    width=min(width,.2*min(np.diff(knots)))
    a,b,c=[window(q,width/k) for k in (1,2,4)]
    rate,coherence=(a-6*b+8*c)/3
    return rate.real,(np.exp(-1j*q['phase'])*coherence).real/rate.real
