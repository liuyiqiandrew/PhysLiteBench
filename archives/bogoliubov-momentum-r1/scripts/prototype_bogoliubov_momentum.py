import numpy as np,json
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal

def thermal(x):
 q=np.exp(-x)
 return q/(-np.expm1(-x))

def values(T,mu):
 def spectra(p):
  e=p*p/2;E=np.sqrt(e*(e+2*mu));n=thermal(E/T);mix=(e+mu)/E
  return E,n,mix
 def exact(p):
  E,n,mix=spectra(p);return p**4*n*(1+n)/(6*np.pi**2)
 def shortcut(p):
  E,n,mix=spectra(p);return T*p*p*mix*n/(2*np.pi**2)
 def wick(p):
  e=p*p/2;E,n,mix=spectra(p);v2=(mix-1)/2;na=mix*n+v2;an=-mu/(2*E)*(2*n+1)
  return p**4*(na*(na+1)-an*an)/(6*np.pi**2)
 limit=np.sqrt(2*(40*T+mu))
 return dict(correct=quad(exact,0,limit,epsabs=1e-12,epsrel=1e-10)[0],shortcut=quad(shortcut,0,limit,epsabs=1e-12,epsrel=1e-10)[0],wick=quad(wick,0,limit,epsabs=1e-12,epsrel=1e-10)[0])

def fock(e,mu,T,N):
 # Diagonalize exact conserved number-difference blocks of the two-mode Hamiltonian.
 energies=[];differences=[];weights=[]
 for d in range(N):
  n=np.arange(N-d);diag=(e+mu)*(2*n+d);off=mu*np.sqrt((n[:-1]+1)*(n[:-1]+d+1))
  eigenvalues=eigh_tridiagonal(diag,off,eigvals_only=True)
  energies.extend(eigenvalues);differences.extend(np.full(len(n),d*d));weights.extend(np.full(len(n),1 if d==0 else 2))
 energies=np.array(energies);boltz=np.array(weights)*np.exp(-(energies-energies.min())/T)
 return float(boltz@differences/boltz.sum())

report={'spectral_cases':[],'fock':[]}
for T in [.08,.2,.5,.9]:
 for mu in [0.,.1,.4,1.]:
  v=values(T,mu);report['spectral_cases'].append(dict(T=T,mu=mu,**v,shortcut_relative_error=abs(v['shortcut']/v['correct']-1),wick_error=abs(v['wick']-v['correct'])))
for e,mu,T in [(.25,.6,.3),(.08,.4,.1),(.5,1.,.5),(.3,0.,.6)]:
 E=np.sqrt(e*(e+2*mu));n=thermal(E/T);expected=2*n*(1+n)
 report['fock'].append(dict(e=e,mu=mu,T=T,expected=expected,cutoffs={str(N):fock(e,mu,T,N) for N in [24,48,80]}))
print(json.dumps(report,indent=2));open('results/bogoliubov-momentum-prototype.json','w').write(json.dumps(report,indent=2)+'\n')
