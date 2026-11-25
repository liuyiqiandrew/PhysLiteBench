"""Independent proposal reference: thermal Fock and singlet spin matrices.

No Bessel weights or scalar anomalous-sign kernels are used here.
"""
from functools import lru_cache
import numpy as np
from scipy.special import expit, gammaln, eval_genlaguerre
from scipy.optimize import root, least_squares
from numpy.polynomial.legendre import leggauss

KB=1.380649e-23
CHARGE=1.602176634e-19
RATE=2e-6*KB/CHARGE**2
HEAT=2e6*(KB/CHARGE)**2
NU=1e6
TAU=1e-5
OMEGA=24.
S=np.array([[0.,1.],[-1.,0.]])


def gaps(left,right):
    return tuple(1.764*tstar*np.tanh(1.74*np.sqrt(tstar/t-1)) for t,tstar in ((left,1.2),(right,1.9)))


@lru_cache(maxsize=256)
def fock_covariances(rho, temperature, count=24, states=96):
    """Direct D(i sqrt(rho)) and D(-i sqrt(rho)) matrix elements."""
    r=np.exp(-OMEGA/temperature)
    cov=[]
    for ell in range(-count,count+1):
        m=np.arange(max(0,-ell),states)
        j=m+ell
        low,high=np.minimum(m,j),np.maximum(m,j)
        factor=np.exp(-rho/2+.5*(gammaln(low+1)-gammaln(high+1)))
        plus=factor*(1j*np.sqrt(rho))**abs(ell)*eval_genlaguerre(low,abs(ell),rho)
        minus=factor*(-1j*np.sqrt(rho))**abs(ell)*eval_genlaguerre(low,abs(ell),rho)
        thermal=(1-r)*r**m
        cov.append((ell,float(thermal@abs(plus)**2),float(thermal@abs(minus)**2),complex(thermal@(plus*minus.conj()))))
    return cov


def matrices(amplitude):
    return {0:np.eye(2),1:amplitude*np.array([[0.,1.],[0.,0.]]),-1:amplitude*np.array([[0.,0.],[1.,0.]])}


def branch_matrix(energy,other,dl,dr,phase,harmonic,amplitude,covariance,creation=False):
    table=matrices(amplitude)
    electron=table[harmonic]
    hole=S@table[-harmonic].conj()@S.conj().T
    if creation:
        electron,hole=electron@S,hole@S
    _,pe,ph,cross=covariance
    result=np.zeros((2,2,len(np.atleast_1d(energy))))
    xil=np.sqrt(np.maximum(0.,energy*energy-dl*dl))/energy
    xir=np.sqrt(np.maximum(0.,other*other-dr*dr))/other
    for bl in (-1,1):
        ul,vl=np.sqrt((1+bl*xil)/2),np.sqrt((1-bl*xil)/2)
        for br in (-1,1):
            ur,vr=np.sqrt((1+br*xir)/2),np.sqrt((1-br*xir)/2)
            a,b=(ul*vr,vl*ur) if creation else (ul*ur,-vl*vr)
            result += (pe*abs(electron[:,:,None]*a)**2+ph*abs(hole[:,:,None]*b)**2+
                       2*np.real(cross*electron[:,:,None]*hole[:,:,None].conj()*np.exp(-1j*phase))*a*b)/4
    return result


@lru_cache(maxsize=128)
def prepare(left,right,phase,amplitude,drive,rho,temperature,nodes=288,count=24,normal=False):
    dl,dr=(0.,0.) if normal else gaps(left,right)
    x,wt=leggauss(nodes)
    cutoff=np.sqrt(100*max(left,right)+40)
    z=(x+1)*cutoff/2
    angle=(x+1)*np.pi/4
    covariances=fock_covariances(rho,temperature,count)
    reverse={ell:(pe,ph,cross) for ell,pe,ph,cross in covariances}
    rows=[]
    for n in (-1,0,1):
        if n and not amplitude: continue
        for covariance in covariances:
            ell,pe,ph,cross=covariance
            if max(pe,reverse[-ell][0])<1e-30: continue
            change=n*drive-ell*OMEGA
            edge=max(dl,dr-change)
            el=edge+z*z;er=el+change
            density=el*er/np.sqrt((el*el-dl*dl)*(er*er-dr*dr))
            forward=branch_matrix(el,er,dl,dr,phase,-n,amplitude,covariance)
            backward=branch_matrix(el,er,dl,dr,phase,-n,amplitude,(-ell,*reverse[-ell]))
            measure=wt*cutoff/2*2*z*density
            for i,sl in enumerate((1,-1)):
                for j,sr in enumerate((1,-1)):
                    if max(np.max(abs(forward[i,j])),np.max(abs(backward[i,j])))==0: continue
                    rows.append((el,er,measure*forward[i,j],measure*backward[i,j],sl,sr,False,n*drive,ell*OMEGA))
            if change>dl+dr:
                el=dl+(change-dl-dr)*np.sin(angle)**2;er=change-el
                density=2*el*er/np.sqrt((el+dl)*(er+dr))
                forward=branch_matrix(el,er,dl,dr,phase,n,amplitude,covariance,True)
                backward=branch_matrix(el,er,dl,dr,phase,n,amplitude,(-ell,*reverse[-ell]),True)
                measure=wt*np.pi/4*density
                for i,sl in enumerate((1,-1)):
                    for j,sr in enumerate((1,-1)):
                        if max(np.max(abs(forward[i,j])),np.max(abs(backward[i,j])))==0: continue
                        rows.append((el,er,measure*forward[i,j],measure*backward[i,j],sl,sr,True,n*drive,ell*OMEGA))
    return dl,dr,rows


def transitions(left,right,temperature,rows,mu):
    heat=np.zeros(4);spin=np.zeros(2);jac=np.zeros((2,2));entropy=0.
    for el,er,forward,backward,sl,sr,pair,work,recoil in rows:
        fl=expit((sl*mu[0]-el)/left);fr=expit((sr*mu[1]-er)/right)
        dfl=sl*fl*(1-fl)/left;dfr=sr*fr*(1-fr)/right
        if pair:
            net=forward*(1-fl)*(1-fr)-backward*fl*fr
            dmu=np.array([-dfl*(forward*(1-fr)+backward*fr),-dfr*(forward*(1-fl)+backward*fl)])
            ds=np.array([sl,sr]);ql,qr=-el,-er
            aff=(el-sl*mu[0])/left+(er-sr*mu[1])/right+recoil/temperature
        else:
            net=forward*fl*(1-fr)-backward*fr*(1-fl)
            dmu=np.array([dfl*(forward*(1-fr)+backward*fr),-dfr*(forward*fl+backward*(1-fl))])
            ds=np.array([-sl,sr]);ql,qr=el,-er
            aff=-(el-sl*mu[0])/left+(er-sr*mu[1])/right+recoil/temperature
        heat+=np.array([net@ql,net@qr,work*np.sum(net),recoil*np.sum(net)])
        spin+=ds*np.sum(net)
        jac+=np.outer(ds,np.sum(dmu,axis=1))
        entropy+=net@aff
    return heat,spin,jac,entropy


def loss(temperature,delta,mu):
    x=mu/temperature;a=np.exp(-delta/temperature)
    value=2*temperature*np.arctanh(a*np.sinh(x)/(1+a*np.cosh(x))) if abs(x)<30 else temperature*(np.logaddexp(0,(mu-delta)/temperature)-np.logaddexp(0,(-mu-delta)/temperature))
    slope=expit((mu-delta)/temperature)+expit((-mu-delta)/temperature)
    return value,slope


def stationary(left,right,phase,amplitude,drive,rho,temperature,conductance=45.,nodes=288,count=24,normal=False):
    dl,dr,rows=prepare(left,right,phase,amplitude,drive,rho,temperature,nodes,count,normal)
    ratio=NU/(TAU*RATE*conductance)
    def balance(mu):
        return transitions(left,right,temperature,rows,mu)[1]-ratio*np.array([loss(t,d,m)[0] for t,d,m in zip((left,right),(dl,dr),mu)])
    def derivative(mu):
        return transitions(left,right,temperature,rows,mu)[2]-ratio*np.diag([loss(t,d,m)[1] for t,d,m in zip((left,right),(dl,dr),mu)])
    answer=root(balance,[0.,0.],jac=derivative,options={'xtol':1e-11})
    mu=answer.x
    if max(abs(balance(mu)))>1e-9:
        answer=least_squares(balance,[0.,0.],jac=derivative,bounds=(-40.,40.),
                             x_scale='jac',ftol=1e-14,xtol=1e-14,gtol=1e-14)
        mu=answer.x
    for _ in range(8):
        value=balance(mu)
        if max(abs(value))<2e-12: break
        mu-=np.linalg.solve(derivative(mu),value)
    if max(abs(balance(mu)))>1e-9 or max(abs(mu))>32:
        raise RuntimeError('Independent spin balance failed or reference window insufficient')
    heat,spin,jac,entropy=transitions(left,right,temperature,rows,mu)
    flip=ratio*np.sum([m*loss(t,d,m)[0]/t for t,d,m in zip((left,right),(dl,dr),mu)])
    return dict(mu=mu,heat=HEAT*conductance*heat,spin=RATE*conductance*spin,
                residual=balance(mu),entropy=KB*RATE*conductance*(entropy+flip))
