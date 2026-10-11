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
    inputs = ref.calibration_inputs()*10
    means = ref.predict(inputs,coupling)
    assert max(abs(oracle.predict_at(inputs,coupling)-means)) < 1e-12
    assert max(abs(shortcut.predict_at(inputs,coupling)-means)) < 1e-12
    if args.generate:
        rng = np.random.default_rng(620001)
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
    report = dict(revision=1,calibration_seed=620001,noise_seed=620009,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        chi = float(residual@residual)/(len(records)-1)
        report['controls'][name] = dict(coupling=model.coupling,chi2=chi)
        assert chi < 1.5 and abs(model.coupling/coupling-1) < .03
    rng = np.random.default_rng(620009)
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
        refined = ref.predict(rows,coupling,length=6400.)
        assert max(abs(fresh-expected)) < 1e-12
        assert max(abs(fresh-refined)) < 1e-10
        assert max(abs(oracle.predict_at(rows,coupling)-refined)) < 1e-10
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
    for tau in np.linspace(.7,1.4,41):
        noiseless = [dict(r,value=float(y)) for r,y in zip(records,oracle.predict_at(inputs,tau))]
        recovery = max(recovery,abs(oracle.Model().fit(noiseless).coupling-tau))
    assert recovery < 1e-12
    rng = np.random.default_rng(620013)
    cases = [(ref.experiment(float(rng.uniform(.1,1.8)),int(rng.integers(1,5)),
                            float(rng.uniform(.5,2))),float(rng.uniform(.7,1.4))) for _ in range(128)]
    cases += [(ref.experiment(u,n,a),g) for u in [.1,1.8] for n in [1,2,3,4]
              for a in [.5,2.] for g in [.7,1.4]]
    domain_error = max(abs(oracle.predict_at([e],g)[0]-ref.predict([e],g)[0]) for e,g in cases)
    assert domain_error < 1e-6
    convergence = []
    for n in [1,2,3,4]:
        for gamma in [.07,.42,2.52]+[2/j+d for j in range(1,n+1) for d in [-.001,0.,.001]]:
            exact = oracle.coefficient(gamma,n)
            coarse,fine = ref.limiting(gamma,n),ref.limiting(gamma,n,6400.)
            assert abs(coarse-exact) < 1e-5 and abs(fine-exact) < 1e-5
            convergence.append(dict(count=n,gamma=gamma,limit_error=float(abs(coarse-exact)),
                                    refined_limit_error=float(abs(fine-exact))))
    one_ion_error = 0.
    for gamma in [.1,1.,1.99,2.,2.01,2.5]:
        for length in [1.,10.,80.]:
            slope=2-gamma
            exact=(1/length if slope == 0 else slope/(-np.expm1(-slope*length)))/2
            one_ion_error=max(one_ion_error,abs(exact-ref.radial(gamma,1,length)))
    assert one_ion_error < 1e-12
    gibbs=[]
    for n,gamma in [(4,.3),(2,1.2),(3,1.6),(4,1.8),(4,2.2),(4,1.)]:
        estimates=[]
        for seed in range(720001,720005):
            logs=[ref.log_partition(gamma,n,length,18,seed) for length in [40.,80.]]
            estimates.append((logs[1]-logs[0])/(80*n))
        radial=(ref.radial(gamma,n,80.,True)-ref.radial(gamma,n,40.,True))/(80*n)
        mean=float(np.mean(estimates));stderr=float(np.std(estimates,ddof=1)/2)
        assert abs(mean-radial) < 6*stderr+2e-4
        gibbs.append(dict(count=n,gamma=gamma,lengths=[40,80],sobol_power=18,
                          seeds=list(range(720001,720005)),estimates=estimates,
                          mean=mean,standard_error=stderr,radial_finite_ratio=radial,
                          limiting_value=oracle.coefficient(gamma,n)))
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_error=recovery,
                  domain_cases=len(cases),domain_reference_max_error=float(domain_error),
                  convergence=convergence,one_ion_finite_integral_max_error=one_ion_error,
                  full_angular_gibbs=gibbs,
                  reference='Ordered-radial finite partition convolution at log(R/a)=3200,6400,12800,25600; inverse-log extrapolation, factor-two refinement, one-ion integral and full angular Gibbs Sobol checks',
                  invariants=['finite neutral canonical ensemble','radius-independent dilation limit',
                              'ideal pressure as coupling vanishes','continuous pressure at integer-sector ties'])
    path = ROOT/'results/expansion-finite-ion-cloud-r1-validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
