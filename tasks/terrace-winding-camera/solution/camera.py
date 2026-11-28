import numpy as np
from scipy.special import eval_hermitenorm, roots_legendre


def orders(pins):
    out = np.zeros(len(pins), dtype=int)
    for j in range(1,len(pins)):
        if pins[j] == pins[j-1]:
            out[j] = out[j-1] + 1
    return out


def heat(x,y,dx,dy,time,rate,length,lifts,drift):
    scale = np.sqrt(2*rate*time)
    distance = (y[None,None,:]-x[None,:,None]+length*lifts[:,None,None]-drift*time)/scale
    degree = dx[:,None]+dy
    return (np.exp(-.5*distance**2) * eval_hermitenorm(degree,distance) * (-1.)**dy
            / (np.sqrt(2*np.pi) * scale**(degree+1)))


def camera_mesh(length, cuts):
    # Split at all discontinuities of the detector.
    bounds = sorted(set([0.,length,*cuts]))
    xs, ws = [], []
    nodes,weights = roots_legendre(12)
    for lo,hi in zip(bounds,bounds[1:]):
        edges = np.linspace(lo,hi,1+int(np.ceil(4*(hi-lo)/length)))
        for a,b in zip(edges,edges[1:]):
            xs.extend((a+b+(b-a)*nodes)/2)
            ws.extend((b-a)*weights/2)
    return np.array(xs), np.array(ws)


def masses(q,rate,db):
    bot,top=np.array(q['bottom']),np.array(q['top'])
    n=len(bot);length=q['circumference'];s,t=np.array(q['levels'])*q['height']
    # A fixed winding and fixed pins fix total displacement, so a common drift
    # cancels from the conditional law. Center it on the selected sector.
    drift=(sum(top)-sum(bot)+length*q['winding'])/(n*q['height'])
    ranks=np.arange(n);sec=19
    roots=np.exp(2j*np.pi*np.arange(n+1)/(n+1))
    phases=np.exp(2j*np.pi*np.arange(sec)/sec)*(-1)**(n-1)
    early=np.exp(2j*np.pi*ranks/n)
    lifts=np.arange(-7,8)
    left_nodes,w1=camera_mesh(length,q['window']);right_nodes,w2=camera_mesh(length,q['count_window'])
    z1=np.zeros(len(w1),dtype=int);z2=np.zeros(len(w2),dtype=int)
    l=heat(bot,left_nodes,db,z1,s,rate,length,lifts,drift)*w1
    m=heat(left_nodes,right_nodes,z1,z2,t-s,rate,length,lifts,drift)*w2
    r=heat(right_nodes,top,z2,orders(top),q['height']-t,rate,length,lifts,drift)
    f=heat(bot,top,db,orders(top),q['height'],rate,length,lifts,drift)
    left=np.einsum('aji,ixy->ajxy',(phases[:,None,None]*early[None,:,None])**lifts,l)
    middle=np.einsum('ai,ixy->axy',phases[:,None]**lifts,m)
    right=np.einsum('ai,ixy->axy',phases[:,None]**lifts,r)
    full=np.einsum('ai,ixy->axy',phases[:,None]**lifts,f)
    lo,hi=q['count_window']; detected=(lo<=right_nodes)&(right_nodes<hi)
    d0=middle[:,:,~detected]@right[:,~detected,:]
    d1=middle[:,:,detected]@right[:,detected,:]
    winding=q['winding']%sec
    with np.errstate(divide="ignore", invalid="ignore"):
        norm=np.fft.fft(np.linalg.det(full))[winding].real/sec
    def cumulative(x):
        if x==0:return 0.
        flag=left_nodes<x
        kernel=np.empty((sec,n,n+1,n+1),complex)
        z,y=roots[:,None,None,None],roots[None,:,None,None]
        for j in range(sec):
            for k in range(n):
                c=left[j,k]
                mat=(c[:,~flag]@d0[j,~flag]+z*(c[:,flag]@d0[j,flag])
                     +y*(c[:,~flag]@d1[j,~flag])+z*y*(c[:,flag]@d1[j,flag]))
                # Some counting phases make the transfer matrix exactly singular.
                with np.errstate(divide="ignore", invalid="ignore"):
                    kernel[j,k]=np.linalg.det(mat)
        joint=(np.fft.fftn(kernel)/kernel.size)[winding,:,:,q['count']].real
        return sum(joint[k,(q['tag']+k)%n+1:].sum() for k in range(n))
    return cumulative(q['window'][1])-cumulative(q['window'][0]), norm

def image(q, rate):
    db = orders(q['bottom'])
    numerator, norm = masses(q, rate, db)
    probability = numerator/norm
    doublet = q.get('open_pair')
    if doublet is None:
        return float(probability)
    third, mixed = db.copy(), db.copy()
    third[doublet+1] = 3
    mixed[doublet:doublet+2] = [1,2]
    n3,z3 = masses(q,rate,third)
    n12,z12 = masses(q,rate,mixed)
    return float(q['circumference']**2*((n3-probability*z3)/24-(n12-probability*z12)/8)/norm)
