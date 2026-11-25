import json,numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
D=json.load(open('/app/data/calibration.json')); Ns=256;M=10
def lev(th,q,r,cols):
 ph=2*np.pi*np.arange(Ns)/Ns;I=np.eye(4);X=np.array([[0,1],[1,0]],complex);Z=np.diag([1.,-1.]);g=np.kron(np.eye(2),X);j=np.kron(np.cos(th)*Z+np.sin(th)*X,Z)
 f=np.cos(q*ph/2)[:,None,None]*I-1j*np.sin(q*ph/2)[:,None,None]*g;s=np.cos(r*ph/2)[:,None,None]*I-1j*np.sin(r*ph/2)[:,None,None]*j
 u=f@s;d=(-1j*q/2*g)@u+f@((-1j*r/2*j)@s);w=u[:,:,cols];d=d[:,:,cols]
 A=np.einsum('nka,nkb->nab',d.conj(),d);B=np.einsum('nka,nkb->nab',d.conj(),w);C=np.einsum('nka,nkb->nab',w.conj(),d)
 Ah=np.fft.fft(A,axis=0)/Ns;Bh=np.fft.fft(B,axis=0)/Ns;Ch=np.fft.fft(C,axis=0)/Ns
 ns=np.arange(-M,M+1);ks=ns-((q+r)%2)/2;H=np.zeros((2*len(ks),2*len(ks)),complex)
 for ii,k in enumerate(ks):
  for jj,l in enumerate(ks):
   m=(int(round(k-l)))%Ns;H[2*ii:2*ii+2,2*jj:2*jj+2]+=Ah[m]+1j*l*Bh[m]-1j*k*Ch[m]
   if ii==jj:H[2*ii:2*ii+2,2*jj:2*jj+2]+=k*l*np.eye(2)
 return eigh(H,eigvals_only=True)
for cols in [slice(0,2),slice(2,4)]:
 c={}
 for x in D[:1]: pass
 for x in D:
  e=x['input']; key=(e['theta'],e['q'],e['r'])
  if key not in c:c[key]=lev(*key,cols)
 def o(I):
  ss=0
  for a in D:
   e=a['input'];x=c[(e['theta'],e['q'],e['r'])]/I;z=np.exp(-(x-x.min())/e['temperature']); y=z@x/z.sum();ss+=((y-a['value'])/.0005)**2
  return ss
 res=minimize_scalar(o,bounds=(.8,1.2),method='bounded');print(cols,res.x,res.fun,res.fun/287,flush=True)
