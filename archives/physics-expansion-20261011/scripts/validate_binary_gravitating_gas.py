"""Validate binary gravity; --generate intentionally replaces data."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/binary-gravitating-gas'


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
    shortcut = load('shortcut',ROOT/'scripts/binary_gravitating_gas_baseline.py')
    gravity,sigma = ref.TRUE_PARAMETER,.001
    inputs = ref.calibration_inputs()*8
    means = ref.predict(inputs,gravity)
    assert max(abs(oracle.predict_at(inputs,gravity)-means)) < 1e-8
    assert max(abs(shortcut.predict_at(inputs,gravity)-means)) < 1e-8
    if args.generate:
        rng = np.random.default_rng(617001)
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
    report = dict(revision=1,calibration_seed=617001,noise_seed=617009,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        chi = float(residual@residual)/(len(records)-1)
        report['controls'][name] = dict(gravity=model.gravity,chi2=chi)
        assert chi < 1.5 and abs(model.gravity/gravity-1) < .03
    rng = np.random.default_rng(617009)
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
        refined = ref.predict(rows,gravity,modes=96,points=384)
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
    rng = np.random.default_rng(617013)
    cases = [(ref.experiment(float(rng.uniform(.8,1.6)),float(rng.uniform(.7,1.5)),
                            float(rng.uniform(1,5)),float(rng.uniform(0,.5))),
              float(rng.uniform(.7,1.4))) for _ in range(16)]
    cases += [(ref.experiment(m,r,ratio,q),g)
              for m,r,g in [(.8,1.5,.7),(1.6,.7,1.4)]
              for ratio,q in [(1.,0.),(1.,.5),(5.,0.),(5.,.5),(5.,.001),(5.,.02)]]
    domain_error = 0.
    for e,g in cases:
        direct = oracle.predict_at([e],g)[0]
        numeric = ref.predict([e],g)[0]
        domain_error = max(domain_error,abs(direct-numeric))
        assert direct > 0
    assert domain_error < 1e-7
    stability = []
    for ratio,q in [(1.,0.),(2.,.3),(3.,.1),(3.,.5),(5.,.02),(5.,.1),(5.,.5)]:
        temperature,alpha = ref.critical(ratio,q)
        left = ref.curvature(.98*alpha,ratio,q)
        right = ref.curvature(1.02*alpha,ratio,q)
        assert left > 0 > right
        assert min(ref.curvature(a,ratio,q) for a in np.linspace(.5,.95*alpha,12)) > 0
        eta,central,profile = ref.equilibrium(alpha,ratio,q)
        mass_light,mass_heavy = profile.y[2:,-1]*temperature/alpha
        assert abs(mass_light-(1-q)) < 1e-8 and abs(mass_heavy-q) < 1e-8
        if q>0 and ratio>1:
            assert central>q
        stability.append(dict(mass_ratio=ratio,heavy_fraction=q,alpha=alpha,
                              central_heavy_fraction=central,temperature_coefficient=temperature,
                              curvature_before=left,curvature_after=right,
                              mass_normalization_error=max(abs(mass_light-(1-q)),abs(mass_heavy-q))))
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_error=recovery,
                  domain_cases=len(cases),domain_reference_max_error=float(domain_error),
                  stability_checks=stability,
                  reference='64-mode-per-species free-energy Hessian; 96-mode/384-point refinement',
                  invariants=['each species mass conserved','first eigenvalue crossing',
                              'temperature scales as G M/R','heavy species concentrates centrally',
                              'equal-mass and absent-heavy-species limits'])
    path = ROOT/'results/expansion-binary-gravitating-gas-r1-validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
