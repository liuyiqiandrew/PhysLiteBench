"""Independent quantum-oscillator physics audit; no task edits or model runs."""
from pathlib import Path
import hashlib, importlib.util, json, time
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PROPOSAL=ROOT/'superconducting-heat/phase-oscillator-prototype'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

p=load(PROPOSAL/'prototype.py','oscillator_proposal')
r=load(PROPOSAL/'reference.py','oscillator_independent_reference')


def direct_displacement(rho,temperature,states=100,count=24):
    q=np.diag(np.sqrt(np.arange(1,states)),1)
    plus=expm(1j*np.sqrt(rho)*(q+q.T));minus=expm(-1j*np.sqrt(rho)*(q+q.T))
    thermal=(1-np.exp(-24/temperature))*np.exp(-24*np.arange(states)/temperature)
    ells=np.arange(-count,count+1);prob=[];cross=[]
    for ell in ells:
        initial=np.arange(max(0,-ell),min(states,states-ell));final=initial+ell
        a,b=plus[final,initial],minus[final,initial]
        prob.append(thermal[initial]@abs(a)**2)
        cross.append(thermal[initial]@(a*b.conj()))
    return ells,np.array(prob),np.array(cross)


def phase_space(x,t):
    if x==0:return t
    if x/t>40:return x
    if x/t< -40:return -x*np.exp(x/t)
    return x/(-np.expm1(-x/t))


def main():
    start=time.perf_counter();rng=np.random.default_rng(613106)
    out=dict(task='superconducting-heat',revision_proposed=5,reviewer='brownian_physics_review',scope='Independent read-only physics audit of proposal sources; no task mutations, paid jobs or resampling.',issues=[],proposal_physics_approved=True)
    errors=[];parity=[];moments=[];balance=[];positive=[]
    for rho,temp in [(0.,12.),(.1,12.),(.7,24.),(1.,12.),(1.,24.)]:
        ell,prob,cross=direct_displacement(rho,temp)
        _,bessel=p.sidebands(rho,temp)
        laguerre=r.fock_covariances(rho,temp)
        errors.extend(abs(prob-bessel));errors.extend(abs(prob-np.array([row[1] for row in laguerre])))
        parity.extend(abs(cross-(-1.)**ell*prob))
        nb=1/np.expm1(24/temp)
        moments.extend([abs(prob.sum()-1),abs(ell@prob-rho),abs((ell-rho)**2@prob-rho*(2*nb+1)),abs(cross.sum()-np.exp(-2*rho*(2*nb+1)))])
        for j in range(1,25):balance.append(abs(prob[24-j]-np.exp(-24*j/temp)*prob[24+j]))
        positive.append(float(prob.min()))
    assert max(errors+parity+moments+balance)<2e-12
    out['direct_matrix_exponential']=dict(fock_states=100,sideband_count=24,probability_error=max(errors),parity_error=max(parity),sum_mean_variance_dephasing_error=max(moments),detailed_balance_error=max(balance),minimum_probability=min(positive))
    matrix_errors=[]
    covariances=r.fock_covariances(.7,24.)
    for _ in range(128):
        el,er=rng.uniform(4,40,2);dl,dr=2.,3.;phi=rng.uniform(-np.pi,np.pi);amp=rng.uniform(0,1);n=int(rng.integers(-1,2));pair=bool(rng.integers(0,2));cov=covariances[int(rng.integers(0,len(covariances)))]
        harmonic=n if pair else -n
        actual=r.branch_matrix(np.array([el]),np.array([er]),dl,dr,phi,harmonic,amp,cov,pair)[:,:,0]
        odd=(-1.)**cov[0];sign=(1 if pair else -1)*(1 if n==0 else -1)
        kernel=.5*cov[1]*(1+sign*odd*dl*dr*np.cos(phi)/(el*er))
        expected=np.zeros((2,2))
        if n==0:
            if pair:expected[0,1]=expected[1,0]=kernel
            else:expected[0,0]=expected[1,1]=kernel
        elif pair:expected[0 if n==1 else 1,0 if n==1 else 1]=amp**2*kernel
        else:expected[1 if n==1 else 0,0 if n==1 else 1]=amp**2*kernel
        matrix_errors.extend(abs(actual-expected).ravel())
    assert max(matrix_errors)<2e-14
    out['singlet_spin_branch_check']=dict(random_cases=128,max_error=max(matrix_errors),channel_rule='Static transfer equal-spin; static creation opposite-spin; driven transfer (-n,n); driven creation (n,n), including n=-1 when oscillator absorption opens it.')
    phase_errors=[];static_mu=[]
    for rho,temp in [(.3,12.),(.7,24.)]:
        args=(.6,.8,0.,0.,.3,rho,temp)
        base=p.stationary(*args);opposite=p.stationary(*args[:2],np.pi,*args[3:]);contrast=(base['heat'][0]-opposite['heat'][0])/2
        for phase in [.8,np.pi/2,-.8]:
            a=list(args);a[2]=phase;b=list(a);b[2]=phase-np.pi
            first=p.stationary(*a);second=p.stationary(*b)
            phase_errors.append(abs((first['heat'][0]-second['heat'][0])/2-np.cos(phase)*contrast))
            static_mu.extend(abs(first['mu']));static_mu.extend(abs(second['mu']))
    assert max(phase_errors+static_mu)<2e-11
    equilibrium=[]
    for rho in [0.,.7,1.]:
        result=p.stationary(5.,5.,.3,0.,.3,rho,5.,normal=True)
        equilibrium.extend(abs(result['heat']));equilibrium.extend(abs(result['mu']))
    assert max(equilibrium)<2e-9
    integrals=[]
    for temp in [.25,1.05,5.]:
        for shift in [-24.,-1.,0.,1.,24.]:
            width=abs(shift)+80*temp
            literal=quad(lambda energy:expit(-energy/temp)*expit((energy+shift)/temp),-width,width,epsabs=2e-12,epsrel=2e-12)[0]
            integrals.append(abs(literal-phase_space(shift,temp)))
    assert max(integrals)<2e-10
    normal=[]
    for temp,drive in [(.25,.3),(1.05,9.)]:
        ell,prob,_=direct_displacement(.7,24.)
        amp=.8;gamma=p.r4.QUASIPARTICLE_DENSITY/(p.r4.NORMAL_POLARIZATION_TIME*p.r4.RATE_CONVERSION*45.)
        def current(v):return sum(y*(phase_space(v-l*24,temp)-phase_space(-v-l*24,temp)) for l,y in zip(ell,prob))
        closed=brentq(lambda mu:amp**2/2*current(drive-2*mu)-gamma*mu,0,drive/2,xtol=5e-15)
        actual=p.stationary(temp,temp,.7,amp,drive,.7,24.,normal=True)
        normal.extend(abs(actual['mu']-closed))
    assert max(normal)<2e-11
    out['independent_closed_anchors']=dict(static_phase_cosine_error=max(phase_errors),static_zero_spin_error=max(static_mu),common_temperature_normal_equilibrium_error=max(equilibrium),literal_signed_energy_phase_space_error=max(integrals),normal_scalar_spin_error=max(normal))
    args=(.6,.8,0.,1.,11.,.7,24.)
    first=p.stationary(*args);second_args=list(args);second_args[2]=np.pi
    second=p.stationary(*second_args);channels=p.prepare(*second_args)[2]
    held=p.transition(args[0],args[1],args[6],channels,first['mu'])[0]*p.r4.HEAT_CONVERSION*45.
    correct=(first['heat'][0]-second['heat'][0])/2
    frozen=(first['heat'][0]-held[0])/2
    roots=[]
    for initial in [(15.,-15.),(-15.,15.),(20.,20.),(-20.,-20.)]:
        roots.extend(abs(p.stationary(*args,initial=initial)['mu']-first['mu']))
    assert max(roots)<2e-10
    out['phase_re_equilibration']=dict(input=args,first_mu=first['mu'].tolist(),opposite_mu=second['mu'].tolist(),correct_contrast_pW=correct,frozen_opposite_spin_contrast_pW=frozen,omission_difference_pW=abs(correct-frozen),multiple_initial_root_error=max(roots))
    out['derivation']={
        'displacement':'D=exp(i*sqrt(rho)*(b+b†)); oscillator number parity Pi sends D to D†. Thus <m+ell|D†|m>=(−1)^ell<m+ell|D|m>, giving conditional cross covariance (−1)^ell*Pell. Thermal reversal weights are P−ell=exp(−ell*w0/Tph)*Pell.',
        'reverse_and_pair_channels':'Define delta=n*w−ell*w0. Transfer ER=EL+delta; creation EL+ER=delta>gapL+gapR. Negative ell opens static opposite-spin and n=−1 down/down creation. Hermitian reverse annihilation is included once through net occupations with P−ell, rather than counted as a separate positive-energy creation channel.',
        'spin_uniqueness':'Each transfer contributes columns (−AL,sL*sR*AL) and (sL*sR*AR,−AR); each creation changes off-diagonal signs but keeps negative diagonal and equal magnitude. AL,AR>=0 for both forward/reverse weights. Positive finite-temperature local relaxation adds strictly negative diagonals. The integrated Jacobian between two candidate roots is strictly column diagonally dominant, hence nonsingular, proving uniqueness. At large polarization the normal-state linear part plus relaxation is positive definite; bounded recoil moments and finite gaps give sublinear corrections, establishing existence/coercivity.',
        'many_weak_cells':'For N independent noninterfering contact cells take conductance per cell G/N and bath linewidth/restoring rate proportional to N^(−1/2). Then tunnel/bath rate ratio→0 and linewidth/oscillator frequency→0, while total G stays fixed. Backaction and any dissipative coherent-pair terms vanish; static phase-controlled coherent Josephson current supplies no dc power. This ordering must be public, not an instantaneous reset of one cell carrying full G.',
        'factorized_source':'Conditional covariance matrix Pell*[[1,DW],[DW,1]] is positive semidefinite, retains detailed balance and correct oscillator probabilities, but drops charge-displacement/energy correlation. It is a coherent complete approximation, exact at rho0 and in the normal state, not an omitted channel or negative-rate model.',
        'phase_protocol':'Stationary spin populations must be solved separately at phi and phi+pi; holding the first populations in the second device changes the observable. Proposal code uses two independent stationary solves.',
        'conservation':'Each transfer or creation satisfies QL+QR+n*w−ell*w0=0. Every driven event injects total spin2n; static creation injects zero total spin, so stationary gate work equals kB*w*(IL+IR)/2. Forward-minus-reverse times log-forward-over-reverse is nonnegative. Adding local flip entropy and spin balance leaves −QL/TL−QR/TR+Qenv/Tph.'}
    out['limitations']=['This approves the proposal physics, not a paid-model difficulty claim. Final actual task/public/source/reference/calibration consistency still requires a frozen-source review.', 'Sharp oscillator lines and equilibrium populations require the stated many-weak-cell ordering. A single oscillator carrying all conductance with zero linewidth would not justify this equilibrium bath approximation.', 'The reference uses independent Fock and 2x2 spin matrices but the same positive-energy integration changes as the prototype; direct exponentiation, closed signed-energy and thermodynamic anchors provide additional independent checks.', 'Production reference must be self-contained instead of importing immutable r4 prototype dependencies; phase-contrast groups must use nonzero RMS signals and individually re-equilibrated devices.']
    out['sources']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [PROPOSAL/'README.md',PROPOSAL/'prototype.py',PROPOSAL/'reference.py',PROPOSAL/'screen.py']}
    out['primary_sources']=['https://web.physics.ucsb.edu/~martinisgroup/papers/Martinis2009supp.pdf','https://arxiv.org/pdf/cond-mat/0508728','https://arxiv.org/pdf/1211.4745']
    out['seconds']=time.perf_counter()-start
    target=ROOT/'results/superconducting-heat-r5-brownian-physics-audit.json';target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['derivation','sources','primary_sources']},indent=2))

if __name__=='__main__':main()
