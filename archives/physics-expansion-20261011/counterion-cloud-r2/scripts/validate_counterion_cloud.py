"""Validate the counterion cell; --generate intentionally replaces data."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/counterion-cloud'


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual,expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    args = parser.parse_args()
    ref = load('reference',TASK/'tests/reference.py')
    oracle = load('oracle',TASK/'solution/model.py')
    shortcut = load('shortcut',ROOT/'scripts/counterion_cloud_baseline.py')
    density,sigma = ref.TRUE_PARAMETER,.001
    inputs = ref.calibration_inputs()*5
    means = ref.predict(inputs,density)
    assert max(abs(oracle.predict_at(inputs,density)-means)) < 1e-12
    assert max(abs(shortcut.predict_at(inputs,density)-means)) < 1e-12
    if args.generate:
        rng = np.random.default_rng(619001)
        records = [dict(input=e,value=float(y),sigma=sigma)
                   for e,y in zip(inputs,means+rng.normal(0,sigma,len(means)))]
        for part in ['environment','tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
        truth = {name:dict(inputs=rows,values=ref.predict(rows,density).tolist())
                 for name,rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    assert inputs == [r['input'] for r in records]
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    report = dict(revision=2,calibration_seed=619001,noise_seed=619009,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        chi = float(residual@residual)/(len(records)-1)
        report['controls'][name] = dict(line_density=model.line_density,chi2=chi)
        assert chi < 1.5 and abs(model.line_density/density-1) < .03
    rng = np.random.default_rng(619009)
    fits,chis = [],[]
    for _ in range(256):
        noisy = [dict(r,value=float(y)) for r,y in zip(records,means+rng.normal(0,sigma,len(means)))]
        model = oracle.Model().fit(noisy)
        fits.append(model.line_density)
        residual = (model.predict(inputs)-[r['value'] for r in noisy])/sigma
        chis.append(float(residual@residual)/(len(noisy)-1))
    assert max(chis) < 1.5 and max(abs(np.asarray(fits)/density-1)) < .03
    report.update(noise_draws=256,noise_parameter_range=[min(fits),max(fits)],noise_max_chi2=max(chis))
    for name,rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        assert rows == truth[name]['inputs']
        fresh = ref.predict(rows,density)
        analytical = oracle.predict_at(rows,density)
        assert max(abs(fresh-expected)) < 1e-12
        assert max(abs(fresh-analytical)) < 1e-10
        errors = {key:error(m.predict(rows),expected) for key,m in models.items()}
        noise_oracle = max(error(oracle.predict_at(rows,p),expected) for p in fits)
        noise_shortcut = min(error(shortcut.predict_at(rows,p),expected) for p in fits)
        assert errors['oracle'] < .04 and noise_oracle < .04
        if name != 'shared_anchors':
            assert errors['shortcut'] > .04 and noise_shortcut > .04
        report['groups'][name] = dict(errors=errors,noise_oracle_max=noise_oracle,
                                     noise_shortcut_min=noise_shortcut,
                                     oracle_reference_max_error=float(max(abs(fresh-analytical))))
    recovery = 0.
    for tau in np.linspace(.7,1.4,41):
        noiseless = [dict(r,value=float(y)) for r,y in zip(records,oracle.predict_at(inputs,tau))]
        recovery = max(recovery,abs(oracle.Model().fit(noiseless).line_density-tau))
    assert recovery < 1e-12
    rng = np.random.default_rng(619013)
    cases = [(ref.experiment(float(rng.uniform(.1,1)),float(rng.uniform(.6,1.25)),
                            float(rng.uniform(.5,2)),float(rng.uniform(0,1))),
              float(rng.uniform(.7,1.4))) for _ in range(256)]
    cases += [(ref.experiment(u,b,a,f),tau) for u in [.1,1.] for b in [.6,1.25]
              for a in [.5,2.] for f in [0.,1.] for tau in [.7,1.4]]
    domain_error = max(abs(oracle.predict_at([e],tau)[0]-ref.predict([e],tau)[0]) for e,tau in cases)
    assert domain_error < 1e-9
    # Check the exact boundaries and nearby points of all condensation regimes.
    transitions = []
    for f in [0.,.1,.4,.7,.85,1.]:
        boundaries = [1/3]
        if f<1:
            boundaries += [1/(3*(1-f)),1/(1-f)]
        for boundary in boundaries:
            for offset in [-1e-6,0.,1e-6]:
                xi = boundary+offset
                e = ref.experiment(1.,xi,1.,f)
                discrepancy = abs(oracle.predict_at([e],1.)[0]-ref.predict([e],1.)[0])
                assert discrepancy < 1e-9
                transitions.append(discrepancy)
    convergence = []
    selected = [(.2,.5),(.6,.85),(1.1,.85),(1.625,.85),(.7,.4),(1.2,.4),
                (1.625,.4),(1.43,.2),(1.625,.2)]
    for xi,f in selected:
        e = ref.experiment(1.,xi,1.,f)
        exact = oracle.predict_at([e],1.)[0]
        pressures = [ref.solve(xi,f,3.,L) for L in [80.,160.,320.]]
        errors = [abs(v-exact) for v in pressures]
        assert errors[-1] < 1e-4
        assert errors[-1] < max(errors[0],1e-8)
        # Charge and species numbers are directly imposed as BVP boundary data.
        # A separate dense quadrature checks both reconstructed populations.
        sol = None
        for L in np.geomspace(.1,12.,100):
            pressure,sol = ref.finite(xi,f,3.,L,sol)
        from scipy.integrate import simpson
        s = np.linspace(0,1,10001)
        n = sol.sol(s)
        populations = simpson(np.exp(n[:2]),x=s,axis=1)*12/2
        population_error = float(max(abs(populations-np.array([xi*(1-f),xi*f/3]))))
        # Exact finite-cell contact identity includes total inner density.
        inner = (np.exp(sol.y[0,0])+np.exp(sol.y[1,0]))/(4*xi*(1-f+f/3))
        contact = (-np.expm1(-24))*(1-xi/(2*(1-f+f/3))+inner)
        assert population_error < 1e-8 and abs(contact-pressure) < 1e-8
        convergence.append(dict(xi=xi,trivalent_charge_fraction=f,
                                finite_log_radii=[80.,160.,320.],pressures=pressures,
                                limit_errors=errors,species_normalization_error=population_error,
                                finite_contact_identity_error=float(abs(contact-pressure))))
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_error=recovery,
                  domain_cases=len(cases),domain_reference_max_error=float(domain_error),
                  transition_cases=len(transitions),transition_max_error=max(transitions),
                  convergence=convergence,
                  reference='Convex dilute free-energy minimization; independent species-conserving finite-cell PB BVP at log(R/a)=80,160,320',
                  invariants=['separate species conservation','neutral cell','zero outer electric field',
                              'finite-cell contact identity','pure-valence limits',
                              'radius-independent dilution limit','continuous condensation transitions'])
    path = ROOT/'results/expansion-counterion-cloud-r2-validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
