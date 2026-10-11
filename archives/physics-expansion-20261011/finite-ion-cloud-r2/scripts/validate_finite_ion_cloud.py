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
        rng = np.random.default_rng(621001)
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
    report = dict(revision=2,calibration_seed=621001,noise_seed=621009,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        chi = float(residual@residual)/(len(records)-1)
        report['controls'][name] = dict(coupling=model.coupling,chi2=chi)
        assert chi < 1.5 and abs(model.coupling/coupling-1) < .03
    rng = np.random.default_rng(621009)
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
        refined = ref.predict(rows,coupling,order=160)
        assert max(abs(fresh-expected)) < 1e-12
        assert max(abs(fresh-refined)) < 1e-9
        assert max(abs(oracle.predict_at(rows,coupling)-refined)) < 1e-9
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
    rng = np.random.default_rng(621013)
    cases = [(ref.experiment(float(rng.uniform(.1,3.)),int(rng.integers(1,3)),
                            float(rng.uniform(1.25,12)),float(rng.uniform(.5,2))),
              float(rng.uniform(.7,1.4))) for _ in range(96)]
    cases += [(ref.experiment(u,n,h,a),g) for u in [.1,3.] for n in [1,2]
              for h in [1.25,12.] for a in [.5,2.] for g in [.7,1.4]]
    domain_error = max(abs(oracle.predict_at([e],g)[0]-ref.predict([e],g)[0]) for e,g in cases)
    assert domain_error < 1e-8
    convergence = []
    for n in [1,2]:
        for gamma in [.07,.3,1.,4/3,2.,3.,4.2]:
            for h in [1.25,2.,5.,12.]:
                exact = oracle.coefficient(gamma,n,h)
                coarse,fine = ref.pressure(gamma,n,h,96),ref.pressure(gamma,n,h,160)
                assert abs(coarse-exact) < 1e-8 and abs(fine-exact) < 1e-9
                convergence.append(dict(count=n,gamma=gamma,ratio=h,
                                        reference_error=float(abs(coarse-exact)),
                                        refined_reference_error=float(abs(fine-exact))))
    ideal_error=max(abs(ref.pressure(0.,n,h)-1.) for n in [1,2] for h in [1.25,2.,12.])
    assert ideal_error < 1e-12
    contact=[]
    x,w=ref.nodes(96)
    for gamma in [.3,1.,4/3,2.,3.,4.2]:
        for h in [1.25,3.,12.]:
            length=np.log(h);gap=length*x;angle=np.pi*x
            distance=(1-np.exp(-gap))[:,None]**2+4*np.exp(-gap)[:,None]*np.sin(angle[None,:]/2)**2
            angular=distance**(gamma/2)@w
            s=(length-gap)[:,None]*x[None,:]
            z=length*np.dot(w,angular*(length-gap)*np.exp((2-gamma)*gap)*(np.exp((4-3*gamma)*s)@w))
            outer=length*np.dot(w,angular*np.exp((4-3*gamma)*(length-gap)+(2-gamma)*gap))
            inner=length*np.dot(w,angular*np.exp((2-gamma)*gap))
            residual=abs((outer-inner)/z-(4-3*gamma))
            assert residual < 1e-10
            contact.append(dict(gamma=gamma,ratio=h,virial_contact_residual=float(residual)))
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_error=recovery,
                  domain_cases=len(cases),domain_reference_max_error=float(domain_error),
                  convergence=convergence,ideal_gas_max_error=float(ideal_error),
                  virial_contact=contact,
                  reference='Direct positive quadrature over common logarithmic radius, radial gap and relative angle; 96 nodes per dimension with 160-node refinement, without the oracle hypergeometric reduction',
                  invariants=['finite neutral canonical ensemble','absolute-radius scale invariance at fixed ratio',
                              'ideal pressure as coupling vanishes','exact finite-annulus outer and inner contact relation'])
    path = ROOT/'results/expansion-finite-ion-cloud-r2-validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
