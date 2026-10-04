"""Validate thermal-bodies r11; regeneration is explicit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_continuous_lyapunov

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/thermal-bodies'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load(TASK/'solution/model.py', 'thermal_oracle')
shortcut = load(ROOT/'scripts/thermal_colored_baseline.py', 'thermal_shortcut')
reference = load(TASK/'tests/reference.py', 'thermal_reference')


def calibration_inputs():
    inputs = []
    controls = [([1., 1.2], .2, .3, [1.5, .6]),
                ([.8, 1.4], .5, .8, [.6, 1.8]),
                ([1.3, .9], .9, 1.7, [2., 1.]),
                ([1.1, 1.1], .35, 1.2, [1.2, 1.2])]
    for _ in range(2):
        for springs, coupling, memory, temperatures in controls:
            for mode in (0, 1):
                for lag in (.15, .35, .6, .9, 1.2, 1.6, 2., 2.5, 3., 3.7, 4.5, 5.5):
                    inputs.append(dict(springs=springs, coupling=coupling, memory=memory,
                        temperatures=temperatures, readout='mode_correlation', mode=mode, lag=lag))
    return inputs


def error(actual, truth):
    return float(np.sqrt(np.mean((actual-truth)**2) / np.mean(truth**2)))


def run(noise_trials=256):
    records = json.loads((TASK/'tests/data/calibration.json').read_text())
    inputs = [r['input'] for r in records]
    sigma = np.array([r['sigma'] for r in records])
    clean = reference.predict(inputs, reference.TRUE_PARAMETER)
    groups = reference.hidden_inputs()
    truths = {key: reference.predict(es, reference.TRUE_PARAMETER) for key, es in groups.items()}
    fitted = {}
    for label, module in [('oracle', oracle), ('shortcut', shortcut)]:
        model = module.Model().fit(records)
        residual = (model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        fitted[label] = dict(friction=model.friction, calibration_chi2=float(residual@residual/(len(records)-1)),
            hidden={key: error(model.predict(es), truths[key]) for key, es in groups.items()})
    gamma_grid = np.linspace(.3, 1.2, 31)
    loss_profile = [float(np.sum((reference.predict(inputs, gamma)-clean)**2)) for gamma in gamma_grid]
    # Full covariance versus independent transmission calculation, including
    # stationary energy balance and reservoir entropy production.
    comparison, balance, entropy_min, positive_min = 0., 0., np.inf, np.inf
    for gamma in (.3, .7, 1.2):
        for k1,k2,c,tau,t1,t2 in [(.8,1.4,.05,.15,2.,.5), (1.,1.,1.2,2.,.5,2.),
                                 (1.2,.9,.3,1.1,1.8,.6), (.8,1.4,.9,.15,1.,1.)]:
            drift,covariance,_=oracle.stationary_state(gamma,k1,k2,c,tau,t1,t2)
            j=np.array([covariance[2,4],covariance[3,5]])
            truth=reference.reservoir_current(gamma,k1,k2,c,tau,t1,t2)
            comparison=max(comparison,abs(j[0]-truth))
            balance=max(balance,abs(sum(j)))
            entropy_min=min(entropy_min,-j[0]/t1-j[1]/t2)
            positive_min=min(positive_min,float(np.min(np.linalg.eigvalsh(covariance))))
    # Calibration identity holds at arbitrary friction, not only the fit.
    equivalence = max(float(np.max(abs(oracle.predict_at(inputs,g)-shortcut.predict_at(inputs,g)))) for g in (.3,.53,.7,1.2))
    ref_cal = float(np.max(abs(oracle.predict_at(inputs,.7)-clean)))
    # Single-bath temperature equilibrium and disconnected systems are exact.
    zero_coupling=max(abs(oracle.stationary_state(.7,1.,1.2,0.,tau,2.,.5)[1][2,4]) for tau in (.15,.8,2.))
    equal_temperature=max(abs(oracle.stationary_state(.7,1.,1.2,.4,tau,1.,1.)[1][2,4]) for tau in (.15,.8,2.))
    # White-noise reference is a separate four-variable Langevin covariance.
    K=oracle.stiffness([1.,1.2],.4);A=np.block([[np.zeros((2,2)),np.eye(2)],[-K,-.7*np.eye(2)]])
    D=np.diag([0.,0.,2*.7*2.,2*.7*.5]);S=solve_continuous_lyapunov(A,-D);white=.7*(2.-S[2,2])
    white_errors=[abs(oracle.stationary_state(.7,1.,1.2,.4,tau,2.,.5)[1][2,4]-white) for tau in (1e-3,5e-4,1e-4)]
    # Independent integration of a modal velocity spectrum checks memory and
    # Fourier normalization as well as the calibration's initial ensemble.
    mode=np.linalg.eigh(K)[0][0];vec=np.linalg.eigh(K)[1][:,0];Teff=vec@np.diag([2.,.5])@vec
    def spectrum(w):
        drag=.7/(1-1j*w*.8)
        return 2*drag.real*Teff*w*w/abs(mode-w*w-1j*w*drag)**2
    spectral=[]
    for lag in (0.,.4,1.7):
        measured=(quad(spectrum,0,np.inf,epsabs=1e-10,limit=300)[0] if lag==0 else quad(spectrum,0,np.inf,weight='cos',wvar=lag,epsabs=1e-10,limit=300)[0])/np.pi
        e=dict(springs=[1.,1.2],coupling=.4,memory=.8,temperatures=[2.,.5],readout='mode_correlation',mode=0,lag=lag)
        spectral.append(abs(measured-reference.predict([e],.7)[0]))
    rng=np.random.default_rng(876034)
    noise=[]
    for i in range(noise_trials):
        noisy=[dict(input=e,value=float(y),sigma=float(s)) for e,y,s in zip(inputs,clean+rng.normal(size=len(inputs))*sigma,sigma)]
        # The two completed controls have the same calibration map to roundoff;
        # fit it once and explicitly check both predictions at every fit.
        model=shortcut.Model().fit(noisy); gamma=model.friction
        y=np.array([r['value'] for r in noisy]);res=(model.predict(inputs)-y)/sigma
        chi=float(res@res/(len(inputs)-1))
        cal_delta=float(np.max(abs(model.predict(inputs)-oracle.predict_at(inputs,gamma))))
        noise.append(dict(friction=gamma,chi2=chi,calibration_equivalence=cal_delta,
            oracle={key:error(oracle.predict_at(es,gamma),truths[key]) for key,es in groups.items()},
            shortcut={key:error(model.predict(es),truths[key]) for key,es in groups.items()}))
    report=dict(revision=11,seed_calibration=876031,seed_noise=876034,noise_trials=noise_trials,
        actual_controls=fitted,independent_current_error=comparison,independent_calibration_error=ref_cal,
        exact_calibration_equivalence=equivalence,stationary_first_law_error=balance,
        minimum_reservoir_entropy_production=entropy_min,minimum_covariance_eigenvalue=positive_min,
        disconnected_current=zero_coupling,equilibrium_current=equal_temperature,
        white_noise_errors=white_errors,modal_spectrum_errors=spectral,
        identification={'friction_grid':gamma_grid.tolist(),'noise_free_loss':loss_profile},
        noise={'calibration_passes':sum(n['chi2']<1.5 for n in noise),
               'parameter_passes':sum(abs(n['friction']/.7-1)<.03 for n in noise),
               'oracle_passes':sum(max(n['oracle'].values())<.05 for n in noise),
               'shortcut_passes':sum(max(n['shortcut'].values())<.05 for n in noise),
               'maximum_oracle_error':max(max(n['oracle'].values()) for n in noise),
               'minimum_shortcut_error':min(min(n['shortcut'].values()) for n in noise),
               'friction_range':[min(n['friction'] for n in noise),max(n['friction'] for n in noise)],
               'maximum_chi2':max(n['chi2'] for n in noise)},noise_realizations=noise)
    assert comparison<1e-8 and equivalence<1e-10 and ref_cal<1e-10
    assert balance<1e-10 and entropy_min>-1e-10 and positive_min>0
    assert max(spectral)<1e-7 and white_errors[-1]<white_errors[0]
    assert report['noise']['calibration_passes']==noise_trials and report['noise']['parameter_passes']==noise_trials
    assert report['noise']['oracle_passes']==noise_trials and report['noise']['shortcut_passes']==0
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--regenerate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    if args.regenerate:
        inputs=calibration_inputs();values=reference.predict(inputs,.7);rng=np.random.default_rng(876031)
        records=[dict(input=e,value=float(v+.003*rng.normal()),sigma=.003) for e,v in zip(inputs,values)]
        data=json.dumps(records,indent=2)+'\n'
        for location in ('environment/data/calibration.json','tests/data/calibration.json'):
            (TASK/location).write_text(data)
    report=run(args.noise_trials);path=ROOT/'results/thermal-r11-validation.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('noise_realizations','identification')},indent=2))
