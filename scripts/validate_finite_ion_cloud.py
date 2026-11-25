"""Validate the finite ionic cloud; --generate intentionally replaces data."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/finite-ion-cloud'


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
    shortcut = load('shortcut',ROOT/'scripts/finite_ion_cloud_baseline.py')
    coupling,sigma = ref.TRUE_PARAMETER,.001
    inputs = ref.calibration_inputs()*5
    means = ref.predict(inputs,coupling)
    assert max(abs(oracle.predict_at(inputs,coupling)-means)) < 1e-9
    assert max(abs(shortcut.predict_at(inputs,coupling)-means)) < 1e-9
    if args.generate:
        rng = np.random.default_rng(622001)
        records = [dict(input=e,value=float(y),sigma=sigma)
                   for e,y in zip(inputs,means+rng.normal(0,sigma,len(means)))]
        for part in ['environment','tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
        truth = {name:dict(inputs=rows,values=ref.predict(rows,coupling).tolist())
                 for name,rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    assert inputs == [r['input'] for r in records]
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    report = dict(revision=3,calibration_seed=622001,noise_seed=622009,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        chi = float(residual@residual)/(len(records)-1)
        report['controls'][name] = dict(coupling=model.coupling,chi2=chi)
        assert chi < 1.5 and abs(model.coupling/coupling-1) < .03
    rng = np.random.default_rng(622009)
    fits,chis = [],[]
    for _ in range(256):
        noisy = [dict(r,value=float(y)) for r,y in zip(records,means+rng.normal(0,sigma,len(means)))]
        model = oracle.Model().fit(noisy)
        fits.append(model.coupling)
        residual = (model.predict(inputs)-[r['value'] for r in noisy])/sigma
        chis.append(float(residual@residual)/(len(noisy)-1))
    assert max(chis) < 1.5 and max(abs(np.asarray(fits)/coupling-1)) < .03
    report.update(noise_draws=256,noise_parameter_range=[min(fits),max(fits)],noise_max_chi2=max(chis))
    for name,rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        assert rows == truth[name]['inputs']
        fresh = ref.predict(rows,coupling)
        refined = ref.predict(rows,coupling,order=64,angles=256)
        assert max(abs(fresh-expected)) < 1e-9
        assert max(abs(fresh-refined)) < 1e-7
        assert max(abs(oracle.predict_at(rows,coupling)-refined)) < 1e-7
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
        recovery = max(recovery,abs(oracle.Model().fit(noiseless).coupling-g))
    assert recovery < 1e-7
    rng = np.random.default_rng(622013)
    cases = [(ref.experiment(float(rng.uniform(.1,3.)),int(rng.integers(1,4)),
                            float(rng.uniform(1.25,12)),float(rng.uniform(.5,2))),
              float(rng.uniform(.7,1.4))) for _ in range(48)]
    cases += [(ref.experiment(u,n,h,a),g) for u in [.1,3.] for n in [1,2,3]
              for h in [1.25,12.] for a in [.5,2.] for g in [.7,1.4]]
    domain_error = max(abs(oracle.predict_at([e],g)[0]-ref.predict([e],g)[0]) for e,g in cases)
    assert domain_error < 1e-4
    convergence = []
    for gamma in [.07,.3,.8,1.,1.5,2.,3.,4.2]:
        for h in [1.25,3.,12.]:
            coarse = ref.pressure(gamma,3,h)
            fine = ref.pressure(gamma,3,h,64,256)
            direct = oracle.coefficient(gamma,3,h)
            refined = oracle.three_ion_coefficient(gamma,h,64,512)
            assert abs(coarse-fine) < 1e-4 and abs(direct-refined) < 2e-5
            assert abs(fine-refined) < 2e-5
            convergence.append(dict(gamma=gamma,ratio=h,reference_value=coarse,
                                    reference_refinement_error=float(abs(coarse-fine)),
                                    oracle_refinement_error=float(abs(direct-refined)),
                                    refined_disagreement=float(abs(fine-refined))))
    determinant=[]
    for n in [1,2,3]:
        for h in [1.25,1.5,2.,3.,5.,12.]:
            length=np.log(h)
            slopes=2*np.arange(n)+2-2*n
            terms=[1/length if a==0 else a/(-np.expm1(-a*length)) for a in slopes]
            exact=-np.expm1(-2*length)*sum(terms)/(2*n)
            result=oracle.coefficient(2.,n,h)
            assert abs(result-exact) < 1e-10
            assert abs(ref.pressure(2.,n,h)-exact) < 1e-10
            determinant.append(dict(count=n,ratio=h,exact=exact,oracle_error=float(abs(result-exact))))
    ideal_error=max(abs(ref.pressure(0.,n,h)-1.) for n in [1,2,3] for h in [1.25,2.,12.])
    assert ideal_error < 1e-10
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_error=recovery,
                  domain_cases=len(cases),domain_reference_max_error=float(domain_error),
                  convergence=convergence,ideal_gas_max_error=float(ideal_error),
                  determinant_gamma_two=determinant,
                  reference='Positive Gibbs quadrature over two linked angles, two radial gaps and common radius; 40 radial and 160 angular nodes, refined to 64 and 256; no Fourier factorization or hypergeometric angular formula for three ions',
                  invariants=['finite neutral canonical ensemble','absolute-radius scale invariance at fixed ratio',
                              'ideal pressure as coupling vanishes','exact Gamma=2 determinantal partition product'])
    path = ROOT/'results/expansion-finite-ion-cloud-r3-validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
