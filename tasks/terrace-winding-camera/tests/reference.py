"""Plane-wave propagation and exact interval overlap matrices."""
import numpy as np


def derivative_orders(coords):
    return [sum(v == x for v in coords[:i]) for i,x in enumerate(coords)]


def wave_grid(size, phase, modes):
    return (2*np.pi*np.arange(-modes,modes+1)-phase)/size


def window_overlap(p, q, lo, hi, size):
    delta = p[:,None]-q[None,:]
    return ((hi-lo)/size * np.exp(0.5j*(lo+hi)*delta)
            * np.sinc((hi-lo)*delta/(2*np.pi)))


def probability(ex, diffusion, modes=19, sectors=22):
    L=ex['circumference']; tag=ex['tag'];total=ex['height']
    s,t=np.array(ex['levels'])*total
    top=np.array(ex['top']);bottom=np.array(ex['bottom']);n=len(top)
    db=np.array(derivative_orders(bottom));dt=np.array(derivative_orders(top))
    theta=np.arange(sectors)*2*np.pi/sectors
    eta=np.arange(n)*2*np.pi/n
    z=np.exp(2j*np.pi*np.arange(n+1)/(n+1))
    signphase=(n-1)*np.pi
    normalization=[]
    result=np.zeros(2)
    for h,cut in enumerate(ex['window']):
        if cut==0: continue
        sums=np.zeros((sectors,n,n+1,n+1),complex)
        for b,angle in enumerate(theta):
            last=wave_grid(L,np.angle(np.exp(1j*(angle+signphase))),modes)
            lastpin=np.exp(1j*last[:,None]*top)*(1j*last[:,None])**dt/L
            bpin=np.exp(-1j*bottom[:,None]*last)*(-1j*last)**db[:,None]
            if not h or ex['window'][0]==0:
                normalization.append(np.linalg.det((bpin*np.exp(-diffusion*total*last**2))@lastpin))
            end=np.exp(-diffusion*(total-t)*last[:,None]**2)*lastpin
            late=window_overlap(last,last,*ex['count_window'],L)
            late_gate=end[None,:,:]+(z[:,None,None]-1)*(late@end)
            middle_phase=np.exp(-diffusion*(t-s)*last[:,None]**2)
            for j,stage in enumerate(eta):
                first=wave_grid(L,np.angle(np.exp(1j*(angle+signphase+stage))),modes)
                initial=np.exp(-1j*bottom[:,None]*first)*(-1j*first)**db[:,None]*np.exp(-diffusion*s*first**2)
                all_sites=window_overlap(first,last,0.,L,L)
                below=window_overlap(first,last,0.,cut,L)
                before=initial@all_sites+(z[:,None,None]-1)*(initial@below)
                matrix=(before[:,None,:,:]*middle_phase.T)@late_gate[None,:,:,:]
                with np.errstate(divide="ignore",invalid="ignore"):
                    sums[b,j]=np.linalg.det(matrix)
        weight=np.fft.fftn(sums)[ex['winding']%sectors,:,:,ex['count']].real / sums.size
        for crossings in range(n):
            result[h]+=weight[crossings,(tag+crossings)%n+1:].sum()
    mass=np.fft.fft(normalization)[ex['winding']%sectors].real/sectors
    return (result[1]-result[0])/mass

def image(ex, diffusion, step=.006):
    i=ex.get('open_pair')
    if i is None:
        return probability(ex,diffusion)
    def split(s):
        pins=list(ex['bottom'])
        half=ex['circumference']*s/2
        pins[i]-=half
        pins[i+1]+=half
        return probability(dict(ex,bottom=pins),diffusion)
    p1,p2,p4=[split(s) for s in (step,2*step,4*step)]
    fine=(p2-p1)/(3*step**2)
    coarse=(p4-p2)/(12*step**2)
    return (4*fine-coarse)/3


TRUE_PARAMETER = .927
