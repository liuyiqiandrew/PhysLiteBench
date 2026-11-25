"""Validate joint winding/occupation; --generate replaces checked-in data."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln, hyp2f1

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/trapped-winding'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def gaussian_path_log(probe, radial, mobility, k, temperature, time, points):
    """Integer angular harmonic with a discretized exact Gaussian OU path.

    Radial occupation uses trapezoidal weights. The tilted path remains
    Gaussian; its endpoint correlation gives the integer angle harmonic.
    This check uses no radial oscillator kernel.
    """
    a = mobility*k
    clock = np.linspace(0.,time,points)
    covariance = (temperature/k)*np.exp(-a*abs(clock[:,None]-clock[None,:]))
    weight = np.full(points,time/(points-1))
    weight[[0,-1]] /= 2
    square_root = np.sqrt(2*radial*weight)
    matrix = np.eye(points)+square_root[:,None]*covariance*square_root[None,:]
    factor = cho_factor(matrix)
    endpoint = covariance[:,[0,-1]]
    tilted = covariance[np.ix_([0,-1],[0,-1])]
    tilted -= (endpoint*square_root[:,None]).T@cho_solve(factor,square_root[:,None]*endpoint)
    rho = tilted[0,1]/np.sqrt(tilted[0,0]*tilted[1,1])
    nu = abs(probe)
    angular = nu*np.log(rho)+2*gammaln(1+nu/2)-gammaln(1+nu)
    angular += np.log(hyp2f1(nu/2,nu/2,1+nu,rho*rho))
    return -2*float(np.log(np.diag(factor[0])).sum())+angular


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    ref = load('ref', TASK/'tests/reference.py')
    oracle = load('oracle', TASK/'solution/model.py')
    shortcut = load('shortcut', ROOT/'scripts/trapped_winding_baseline.py')
    mobility,sigma = ref.TRUE_PARAMETER,.001
    inputs = ref.calibration_inputs()*6
    means = ref.predict(inputs,mobility)
    assert max(abs(oracle.predict_at(inputs,mobility)-means)) < 1e-8
    assert max(abs(shortcut.predict_at(inputs,mobility)-means)) < 1e-8
    if args.generate:
        rng = np.random.default_rng(615101)
        records = [dict(input=e,value=float(y),sigma=sigma)
                   for e,y in zip(inputs,means+rng.normal(0,sigma,len(means)))]
        for part in ['environment','tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
        truth = {name:dict(inputs=rows,values=ref.predict(rows,mobility).tolist())
                 for name,rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    assert inputs == [r['input'] for r in records]
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    report = dict(revision=2,calibration_seed=615101,noise_seed=615109,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        chi = float(residual@residual)/(len(records)-1)
        report['controls'][name] = dict(mobility=model.mobility,chi2=chi)
        assert chi < 1.5 and abs(model.mobility/mobility-1) < .03
    rng = np.random.default_rng(615109)
    fits,chis = [],[]
    for _ in range(256):
        noisy = [dict(r,value=float(y)) for r,y in zip(records,means+rng.normal(0,sigma,len(means)))]
        model = oracle.Model().fit(noisy)
        fits.append(model.mobility)
        residual = (model.predict(inputs)-[r['value'] for r in noisy])/sigma
        chis.append(float(residual@residual)/(len(noisy)-1))
    assert max(chis) < 1.5 and max(abs(np.asarray(fits)/mobility-1)) < .03
    report.update(noise_draws=256,noise_parameter_range=[min(fits),max(fits)],noise_max_chi2=max(chis))
    for name,rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        assert rows == truth[name]['inputs']
        fresh = ref.predict(rows,mobility)
        refined = ref.predict(rows,mobility,times=(14.,18.),order=128)
        assert max(abs(fresh-expected)) < 1e-12
        assert max(abs(fresh-refined)) < 1e-8
        assert max(abs(oracle.predict_at(rows,mobility)-refined)) < 1e-10
        errors = {key:error(m.predict(rows),expected) for key,m in models.items()}
        noise_oracle = max(error(oracle.predict_at(rows,p),expected) for p in fits)
        noise_shortcut = min(error(shortcut.predict_at(rows,p),expected) for p in fits)
        assert errors['oracle'] < .04 and noise_oracle < .04
        if name != 'shared_anchors':
            assert errors['shortcut'] > .04 and noise_shortcut > .04
        report['groups'][name] = dict(errors=errors,noise_oracle_max=noise_oracle,
                                     noise_shortcut_min=noise_shortcut,
                                     reference_refinement_max_error=float(max(abs(fresh-refined))))
    recovery = 0.
    for mu in np.linspace(.7,1.4,41):
        noiseless = [dict(r,value=float(y)) for r,y in zip(records,oracle.predict_at(inputs,mu))]
        recovery = max(recovery,abs(oracle.Model().fit(noiseless).mobility-mu))
    assert recovery < 1e-7
    rng = np.random.default_rng(615113)
    cases = [(ref.experiment(float(rng.uniform(.6,1.8)),float(rng.uniform(.5,1.8)),
                            float(rng.uniform(-2.5,2.5)),float(rng.uniform(0,1.8))),
              float(rng.uniform(.7,1.4))) for _ in range(96)]
    cases += [(ref.experiment(k,t,p,s),mu) for k in [.6,1.8] for t in [.5,1.8]
              for p in [-2.5,0.,2.5] for s in [0.,1.8] for mu in [.7,1.4]]
    domain_error, kernel_error = 0.,0.
    for e,mu in cases:
        value = oracle.predict_at([e],mu)[0]
        numeric = ref.predict([e],mu)[0]
        domain_error = max(domain_error,abs(value-numeric))
        assert value >= 0.
        assert oracle.predict_at([dict(e,probe=-e['probe'])],mu)[0] == value
        assert oracle.predict_at([dict(e,probe=0.,radial_probe=0.)],mu)[0] == 0.
        assert value >= shortcut.predict_at([e],mu)[0]-1e-14
        # Integrate the same kernel by its convergent positive Bessel series.
        a,D,nu = mu*e['stiffness'],mu*e['temperature'],abs(e['probe'])
        omega = np.sqrt(a*a+4*D*e['radial_probe'])
        time = 1.2/omega
        h = a*np.sinh(omega*time)+omega*np.cosh(omega*time)
        fraction = omega/h
        analytic = np.log(2*a*omega*np.sinh(omega*time)/h**2)+a*time
        analytic += nu*np.log(fraction)+2*gammaln(1+nu/2)-gammaln(1+nu)
        analytic += np.log(hyp2f1(1+nu/2,1+nu/2,1+nu,fraction**2))
        integrated = ref.log_characteristic(e['probe'],e['radial_probe'],mu,e['stiffness'],e['temperature'],time)
        kernel_error = max(kernel_error,abs(analytic-integrated))
    assert domain_error < 1e-8 and kernel_error < 1e-10
    checks=[]
    for mu,k,t,s,p,tau in [(.7,.6,1.8,1.8,1,1.),(1.4,1.8,.5,.3,2,1.5),
                           (1.09,1.,1.,.8,0,2.),(1.09,.8,1.4,1.2,2,2.)]:
        omega=np.sqrt((mu*k)**2+4*mu*t*s);time=tau/omega
        exact=ref.log_characteristic(p,s,mu,k,t,time,128)
        coarse=gaussian_path_log(p,s,mu,k,t,time,257)
        fine=gaussian_path_log(p,s,mu,k,t,time,513)
        richardson=(4*fine-coarse)/3
        assert abs(richardson-exact) < 1e-8
        checks.append(dict(mobility=mu,stiffness=k,temperature=t,radial_probe=s,probe=p,time=time,
                           reference_log_characteristic=exact,coarse_log_error=coarse-exact,
                           fine_log_error=fine-exact,richardson_log_error=richardson-exact))
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_error=recovery,
                  domain_cases=len(cases),domain_reference_max_error=domain_error,
                  heat_kernel_series_max_log_error=kernel_error,
                  cartesian_gaussian_path_checks=checks,
                  invariants=['nonnegative rate','even angular probe','zero joint probe',
                              'exact shared marginals','positive joint correlation correction'])
    path = ROOT/'results/expansion-trapped-winding-r2-validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
