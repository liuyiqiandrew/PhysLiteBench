import numpy as np
from scipy.optimize import brentq
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

MU=1.;LAM=2.


def stretch(s,p):
    return brentq(lambda t:MU*(t*t-1)+LAM*np.log(s*t)+p*s*t,.2,2.)


def modes(s,p,rho=1.,follower=True):
    t=stretch(s,p);J=s*t;c=MU-LAM*np.log(J);b=MU*t*t;chi=b/(b+LAM+c)
    eta=1. if follower else c/b
    def secular(x):
        return ((1+eta-x)**2-(1+eta)**2*np.sqrt((1-x)*(1-chi*x)))/x
    x=brentq(secular,1e-6,1-1e-10,xtol=1e-13)
    v2=(MU*(s*s-t*t)+b*x)/rho
    return np.sqrt(v2),dict(t=t,x=x,eta=eta,chi=chi,v2=v2)


def energy(F,p):
    j=np.linalg.det(F)
    return MU/2*(np.sum(F*F)-3-2*np.log(j))+LAM/2*np.log(j)**2+p*j


def hessian(s,p,follower):
    t=stretch(s,p);F=np.diag([s,1.,t]);J=np.linalg.det(F)
    maps=[]
    for i in [0,2]:
        for a in [0,2]:
            L=np.zeros((3,3));L[i,a]=1
            maps.append(L@F)
    def calc(h):
        A=np.empty((4,4))
        for i,ei in enumerate(maps):
            for j,ej in enumerate(maps):
                A[i,j]=(energy(F+h*(ei+ej),p if follower else 0)-energy(F+h*(ei-ej),p if follower else 0)-energy(F+h*(-ei+ej),p if follower else 0)+energy(F-h*(ei+ej),p if follower else 0))/(4*h*h*J)
        return A
    return (4*calc(.0002)-calc(.0004))/3


def finite_element(s,p,rho=1.,follower=True,n=400,depth=20.):
    h=depth/n;A=hessian(s,p,follower);J=s*stretch(s,p)
    size=2*(n+1);K=lil_matrix((size,size),dtype=complex);M=lil_matrix((size,size),dtype=complex)
    for element in range(n):
        inds=[2*element,2*element+1,2*element+2,2*element+3]
        ke=np.zeros((4,4),complex);me=np.zeros((4,4),complex)
        for z in [-1/np.sqrt(3),1/np.sqrt(3)]:
            N=np.array([(1-z)/2,(1+z)/2]);dN=np.array([-1,1])/h
            B=np.zeros((4,4),complex);Nmat=np.zeros((2,4),complex)
            for node in range(2):
                for i in range(2):
                    B[2*i,2*node+i]=1j*N[node];B[2*i+1,2*node+i]=dN[node]
                    Nmat[i,2*node+i]=N[node]
            ke+=h/2*B.conj().T@A@B
            me+=h/2*rho/J*Nmat.conj().T@Nmat
        for i in range(4):
            for j in range(4):K[inds[i],inds[j]]+=ke[i,j];M[inds[i],inds[j]]+=me[i,j]
    K=K[:-2,:-2].tocsc();M=M[:-2,:-2].tocsc()
    v2=eigsh(K,k=2,M=M,sigma=0.,which='LM',return_eigenvectors=False)
    return np.sqrt(min(v2.real))


if __name__=='__main__':
    for s in [.95,1.,1.1,1.2]:
        for p in [0.,.2,.4,.5]:
            try:
                v,info=modes(s,p);v0,_=modes(s,p,follower=False)
                print(s,p,v,v0,(v0/v-1),info)
            except ValueError as e:print('FAIL',s,p,str(e))
    for s,p in [(1.,0.),(1.1,.2),(.95,.4),(1.2,.5)]:
        for follower in [True,False]:
            exact=modes(s,p,follower=follower)[0];v=finite_element(s,p,follower=follower,n=300);v2=finite_element(s,p,follower=follower,n=600)
            print('REF',s,p,follower,exact,v,v2,(4*v2-v)/3-exact)
