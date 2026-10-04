import numpy as np
from scipy.integrate import solve_bvp
from numpy.polynomial.legendre import leggauss
G=.06; NU=.04

def modes(z,w,d,a,p):
 k=w*np.sin(a);s=np.roots([NU,G-1j*w-NU*w*w,-w*w*(G-1j*w)-1j*w*p*p]);q=np.sqrt(s-k*k+0j)
 z=np.atleast_1d(z);fields=np.zeros((len(z),4,4),complex)
 for j in range(2):
  for side in range(2):
   sign=1 if side==0 else -1;anchor=0 if side==0 else d
   E=np.exp(1j*sign*q[j]*(z-anchor));J=(s[j]-w*w)/(1j*w)
   fields[:,:,2*j+side]=E[:,None]*np.array([1,1j*sign*q[j],J,1j*sign*q[j]*J])
 return fields

def solve(w,d,a,p):
 f,b=modes([0,d],w,d,a,p);v=w*np.cos(a)
 bc=np.array([f[1]+1j*v*f[0],b[1]-1j*v*b[0],f[2],b[2]])
 A=np.linalg.solve(bc,[2j*v,0,0,0]);return A

def pred(w,d,a,p,window=(0,1),correct=True):
 A=solve(w,d,a,p);x,h=leggauss(64);lo,hi=np.array(window)*d;z=(lo+hi)/2+(hi-lo)*x/2
 E,Ep,J,Jp=(modes(z,w,d,a,p)@A).T;k=w*np.sin(a)
 Jpp=((G-1j*w+NU*k*k)*J-p*p*E)/NU
 q=G*abs(J)**2+NU*(abs(Jp)**2+k*k*abs(J)**2) if correct else G*abs(J)**2-NU*np.real(np.conj(J)*(Jpp-k*k*J))
 return ((hi-lo)/2*np.dot(h,q)/(p*p*np.cos(a))).real

def reference(w,d,a,p,window=(0,1)):
 k=w*np.sin(a);v=w*np.cos(a);M=np.array([[0,1,0,0],[-(w*w-k*k),0,-1j*w,0],[0,0,0,1],[-p*p/NU,0,(G-1j*w+NU*k*k)/NU,0]],complex)
 A=np.block([[M.real,-M.imag],[M.imag,M.real]])
 C0=np.array([[1j*v,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]],complex)
 C1=np.array([[0,0,0,0],[-1j*v,1,0,0],[0,0,0,0],[0,0,1,0]],complex)
 R0=np.block([[C0.real,-C0.imag],[C0.imag,C0.real]])
 R1=np.block([[C1.real,-C1.imag],[C1.imag,C1.real]])
 rhs=np.array([0,0,0,0,2*v,0,0,0])
 z=np.linspace(0,d,25);Y=np.zeros((8,len(z)))
 def fun(z,Y):return A@Y
 def jac(z,Y):return np.repeat(A[:,:,None],len(z),axis=2)
 b=solve_bvp(fun,lambda a,b:R0@a+R1@b-rhs,z,Y,tol=1e-8,max_nodes=2000,fun_jac=jac,bc_jac=lambda a,b:(R0,R1))
 if not b.success:raise RuntimeError(b.message)
 x,h=leggauss(96);lo,hi=np.array(window)*d;z=(lo+hi)/2+(hi-lo)*x/2;vals=b.sol(z);E,Ep,J,Jp=vals[:4]+1j*vals[4:]
 q=G*abs(J)**2+NU*(abs(Jp)**2+k*k*abs(J)**2)
 return float(((hi-lo)/2*np.dot(h,q)/(p*p*np.cos(a))).real)
if __name__=='__main__':
 for d in [.3,.5,.8,1.2]:
  for w in [.7,1.,1.4]:
   a=.65;p=1.03
   A=solve(w,d,a,p);f,b=modes([0,d],w,d,a,p)@A;bal=1-abs(f[0]-1)**2-abs(b[0])**2
   data=[]
   for window in [(0,1),(0,.25),(.3,.7),(.75,1)]:
    o=pred(w,d,a,p,window);s=pred(w,d,a,p,window,False)
    data.append([window,round(o,5),round(s,5),round(s/o-1,3)])
   print('d/w',d,w,'balance',pred(w,d,a,p)-bal,'vsBVP',reference(w,d,a,p)-pred(w,d,a,p),data)
