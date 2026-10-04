"""Prototype only: hard-core boson interference versus fermionic coherence."""
from pathlib import Path
from itertools import combinations
import json,time
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

SITES=8
OCCUPIED=(2,3,4)
POSITIONS=np.arange(SITES)-(SITES-1)/2


def single_hamiltonian(hopping,trap,tilt):
    H=np.diag(trap*POSITIONS**2+tilt*POSITIONS)
    H+=np.diag(np.full(SITES-1,-hopping),1)+np.diag(np.full(SITES-1,-hopping),-1)
    return H


def correlations(hopping,trap,tilt,duration):
    energy,U=eigh(single_hamiltonian(hopping,trap,tilt))
    propagated=(U*np.exp(-1j*energy*duration))@U.conj().T
    P=propagated[:,OCCUPIED]
    fermion=P.conj()@P.T  # matrix entry <f_i^dagger f_j>
    augmented=[]
    for site in range(SITES):
        Q=np.zeros((SITES,len(OCCUPIED)+1),complex);Q[:,:-1]=P
        Q[:site,:-1]*=-1
        Q[site,-1]=1
        augmented.append(Q)
    green=np.array([[np.linalg.det(A.conj().T@B) for B in augmented] for A in augmented])
    boson=green.T.copy()
    boson[np.diag_indices(SITES)]=1-green.diagonal()
    return fermion,boson


def exact_fock(hopping,trap,tilt,duration):
    basis=[sum(1<<j for j in sites) for sites in combinations(range(SITES),len(OCCUPIED))]
    index={b:i for i,b in enumerate(basis)}
    H=np.zeros((len(basis),len(basis)))
    onsite=trap*POSITIONS**2+tilt*POSITIONS
    for col,bits in enumerate(basis):
        H[col,col]=sum(onsite[j] for j in range(SITES) if bits&(1<<j))
        for j in range(SITES-1):
            if bool(bits&(1<<j)) != bool(bits&(1<<(j+1))):
                target=bits^(1<<j)^(1<<(j+1));H[index[target],col]=-hopping
    energies,U=eigh(H)
    initial=index[sum(1<<j for j in OCCUPIED)]
    psi=(U*np.exp(-1j*energies*duration))@U[initial].conj()
    rho=np.zeros((SITES,SITES),complex)
    for j in range(SITES):
        for i in range(SITES):
            for col,bits in enumerate(basis):
                if not bits&(1<<j):continue
                removed=bits^(1<<j)
                if removed&(1<<i):continue
                target=removed|(1<<i)
                rho[i,j]+=np.conj(psi[index[target]])*psi[col]
    return rho,psi,H


def momentum(rho,q):
    amplitude=np.exp(-1j*q*POSITIONS)
    return float(np.vdot(amplitude,rho@amplitude).real/SITES)


def main():
    start=time.monotonic();report={'status':'prototype_only_no_task_or_model_evaluations','sites':SITES,'initial_occupied_sites':OCCUPIED}
    checks=[]
    rng=np.random.default_rng(119181)
    for _ in range(18):
        J=rng.uniform(.8,1.2);trap=rng.uniform(0,.15);tilt=rng.uniform(-.3,.3);t=rng.uniform(.1,2.)
        f,b=correlations(J,trap,tilt,t);r,psi,H=exact_fock(J,trap,tilt,t)
        checks.append(dict(hopping=J,trap=trap,tilt=tilt,time=t,one_body_error=float(abs(b-r).max()),density_equivalence=float(abs(b.diagonal()-f.diagonal()).max()),nearest_bond_equivalence=float(abs(np.diag(b,1)-np.diag(f,1)).max()),minimum_boson_eigenvalue=float(np.linalg.eigvalsh(b).min()),minimum_fermion_eigenvalue=float(np.linalg.eigvalsh(f).min()),trace_error=float(abs(np.trace(b)-len(OCCUPIED))),number_error=float(abs(np.vdot(psi,psi)-1)),energy=float(np.vdot(psi,H@psi).real)))
    assert max(c['one_body_error'] for c in checks)<1e-12
    assert max(c['density_equivalence'] for c in checks)<1e-12
    assert max(c['nearest_bond_equivalence'] for c in checks)<1e-12
    cal=[]
    for trap,tilt in [(0.,0.),(.08,.15),(.15,-.3)]:
        for t in [.12,.2,.3,.4]:
            for site in [1,5]:
                cal.append(dict(trap=trap,tilt=tilt,time=t,site=site))
    def predict_cal(J):return np.array([correlations(J,e['trap'],e['tilt'],e['time'])[0][e['site'],e['site']].real for e in cal])
    values=predict_cal(1.)
    slopes=[(predict_cal(J+1e-5)-predict_cal(J-1e-5))/2e-5 for J in np.linspace(.8,1.2,41)]
    recoveries=[]
    for true in [.8,.9,1.,1.1,1.2]:
        truth=predict_cal(true)
        fit=minimize_scalar(lambda J:np.sum((predict_cal(J)-truth)**2),bounds=(.8,1.2),method='bounded',options={'xatol':1e-12}).x
        recoveries.append(dict(true=true,fitted=float(fit),error=float(abs(fit-true))))
    hidden=[]
    for trap,tilt in [(0.,0.),(.08,.15),(.15,-.3)]:
        for t in [.5,.8,1.1,1.4,1.8]:
            f,b=correlations(1.,trap,tilt,t)
            qs=np.linspace(-np.pi,np.pi,17,endpoint=False)
            correct=np.array([momentum(b,q) for q in qs]);short=np.array([momentum(f,q) for q in qs])
            hidden.append(dict(trap=trap,tilt=tilt,time=t,momenta=qs.tolist(),oracle=correct.tolist(),shortcut=short.tolist(),normalized_rms=float(np.linalg.norm(short-correct)/np.linalg.norm(correct)),minimum_correct=float(correct.min()),minimum_shortcut=float(short.min()),q_integral_error=float(abs(correct.mean()*SITES-len(OCCUPIED)))))
    report.update(independent_checks=checks,calibration=cal,calibration_values=values.tolist(),minimum_calibration_derivative=float(np.min(slopes)),maximum_noiseless_recovery_error=max(r['error'] for r in recoveries),parameter_recovery=recoveries,hidden=hidden,seconds=time.monotonic()-start)
    out=Path(__file__).with_name('assessment.json');out.write_text(json.dumps(report,indent=2)+'\n')
    print('independent',max(c['one_body_error'] for c in checks),'cal derivative',report['minimum_calibration_derivative'])
    print('gaps',[(r['time'],r['trap'],round(r['normalized_rms'],4),round(r['minimum_correct'],4)) for r in hidden])
if __name__=='__main__':main()
