"""Validate finite-bath gravity; --generate intentionally replaces data."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/finite-bath-gravitating-gas'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    ref = load('reference',TASK/'tests/reference.py')
    oracle = load('oracle',TASK/'solution/model.py')
    shortcut = load('shortcut',ROOT/'scripts/finite_bath_gravitating_gas_baseline.py')
    gravity,sigma = ref.TRUE_PARAMETER,.001
    inputs = ref.calibration_inputs()*8
    means = ref.predict(inputs,gravity)
    assert max(abs(oracle.predict_at(inputs,gravity)-means)) < 1e-8
    assert max(abs(shortcut.predict_at(inputs,gravity)-means)) < 1e-8
    if args.generate:
        rng = np.random.default_rng(616001)
        records = [dict(input=e,value=float(y),sigma=sigma)
                   for e,y in zip(inputs,means+rng.normal(0,sigma,len(means)))]
        for part in ['environment','tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
        truth = {name:dict(inputs=rows,values=ref.predict(rows,gravity).tolist())
                 for name,rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    assert inputs == [r['input'] for r in records]
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    report = dict(revision=1,calibration_seed=616001,noise_seed=616009,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        chi = float(residual@residual)/(len(records)-1)
        report['controls'][name] = dict(gravity=model.gravity,chi2=chi)
        assert chi < 1.5 and abs(model.gravity/gravity-1) < .03
    rng = np.random.default_rng(616009)
    fits,chis = [],[]
    for _ in range(256):
        noisy = [dict(r,value=float(y)) for r,y in zip(records,means+rng.normal(0,sigma,len(means)))]
        model = oracle.Model().fit(noisy)
        fits.append(model.gravity)
        residual = (model.predict(inputs)-[r['value'] for r in noisy])/sigma
        chis.append(float(residual@residual)/(len(noisy)-1))
    assert max(chis) < 1.5 and max(abs(np.asarray(fits)/gravity-1)) < .03
    report.update(noise_draws=256,noise_parameter_range=[min(fits),max(fits)],noise_max_chi2=max(chis))
    for name,rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        assert rows == truth[name]['inputs']
        fresh = ref.predict(rows,gravity)
        refined = ref.predict(rows,gravity,modes=128,points=384)
        assert max(abs(fresh-expected)) < 1e-12
        assert max(abs(fresh-refined)) < 1e-8
        assert max(abs(oracle.predict_at(rows,gravity)-refined)) < 1e-8
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
    for g in np.linspace(.7,1.4,41):
        noiseless = [dict(r,value=float(y)) for r,y in zip(records,oracle.predict_at(inputs,g))]
        recovery = max(recovery,abs(oracle.Model().fit(noiseless).gravity-g))
    assert recovery < 1e-12
    rng = np.random.default_rng(616013)
    cases = [(ref.experiment(mass=float(rng.uniform(.8,1.6)),radius=float(rng.uniform(.7,1.5)),
                            bath=float(rng.uniform(0,2))),float(rng.uniform(.7,1.4))) for _ in range(24)]
    cases += [(ref.experiment(mass=m,radius=r,bath=b),g)
              for m in [.8,1.6] for r in [.7,1.5] for b in [0.,2.,'infinite'] for g in [.7,1.4]]
    cases += [(ref.experiment('temperature',m,r,b,c),g)
              for m,r,g in [(.8,1.5,.7),(1.6,.7,1.4)]
              for b in [0.,2.,'infinite'] for c in [1.5,25.]]
    domain_error = max(abs(oracle.predict_at([e],g)[0]-ref.predict([e],g)[0]) for e,g in cases)
    assert domain_error < 1e-8
    stability = []
    for bath in [0.,.25,.5,1.,1.5,2.,'infinite']:
        temperature,alpha = ref.critical(bath)
        left = ref.curvature(.98*alpha,bath)
        right = ref.curvature(1.02*alpha,bath)
        assert left > 0 > right
        # Check no earlier loss on the branch; the first zero is selected.
        assert min(ref.curvature(a,bath) for a in np.linspace(1.,.95*alpha,12)) > 0
        assert abs(ref.equilibrium(alpha)[0]*ref.equilibrium(alpha)[1]-
                   alpha**2*np.exp(-ref.equilibrium(alpha)[2])) < 1e-8
        stability.append(dict(bath=bath,alpha=alpha,contrast=float(np.exp(ref.equilibrium(alpha)[2])),
                              temperature_coefficient=temperature,curvature_before=left,curvature_after=right))
    finite = [r['temperature_coefficient'] for r in stability[:-1]]
    assert np.all(np.diff(finite)<0) and finite[-1]>stability[-1]['temperature_coefficient']
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_error=recovery,
                  domain_cases=len(cases),domain_reference_max_error=float(domain_error),
                  stability_checks=stability,
                  reference='96-mode radial entropy Hessian; 128-mode/384-point refinement',
                  invariants=['mass-conserving regular perturbations','first eigenvalue crossing',
                              'temperature scales as G M/R','finite-bath threshold approaches canonical limit'])
    path = ROOT/'results/expansion-finite-bath-gravitating-gas-r1-validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
