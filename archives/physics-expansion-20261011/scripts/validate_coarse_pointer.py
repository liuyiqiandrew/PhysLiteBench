"""Validate the pointer apparatus; --generate replaces checked-in measurements."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/coarse-pointer'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    ref = load('ref', TASK/'tests/reference.py')
    oracle = load('oracle', TASK/'solution/model.py')
    shortcut = load('shortcut', ROOT/'scripts/coarse_pointer_baseline.py')
    width, sigma = ref.TRUE_PARAMETER, .0005
    inputs = ref.calibration_inputs()*12
    means = ref.predict(inputs,width)
    assert max(abs(oracle.predict_at(inputs,width)-means)) < 1e-12
    assert max(abs(shortcut.predict_at(inputs,width)-means)) < 1e-12
    if args.generate:
        rng = np.random.default_rng(614001)
        records = [dict(input=e,value=float(y),sigma=sigma)
                   for e,y in zip(inputs,means+rng.normal(0,sigma,len(means)))]
        for part in ['environment','tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
        truth = {name:dict(inputs=rows,values=ref.predict(rows,width).tolist())
                 for name,rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    assert inputs == [r['input'] for r in records]
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name:module.Model().fit(records) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    report = dict(revision=1,calibration_seed=614001,noise_seed=614009,controls={},groups={})
    for name,model in models.items():
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        chi = float(residual@residual)/(len(records)-1)
        report['controls'][name] = dict(width=model.width,chi2=chi)
        assert chi < 1.5 and abs(model.width/width-1) < .03
    rng = np.random.default_rng(614009)
    fits,chis = [],[]
    for _ in range(256):
        noisy = [dict(r,value=float(y)) for r,y in zip(records,means+rng.normal(0,sigma,len(means)))]
        model = oracle.Model().fit(noisy)
        fits.append(model.width)
        residual = (model.predict(inputs)-[r['value'] for r in noisy])/sigma
        chis.append(float(residual@residual)/(len(noisy)-1))
    assert max(chis) < 1.5 and max(abs(np.asarray(fits)/width-1)) < .03
    report.update(noise_draws=256,noise_parameter_range=[min(fits),max(fits)],noise_max_chi2=max(chis))
    for name,rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        assert rows == truth[name]['inputs']
        fresh = ref.predict(rows,width)
        refined = ref.predict(rows,width,eps=2e-14)
        assert max(abs(fresh-expected)) < 1e-12
        assert max(abs(fresh-refined)) < 1e-12
        assert max(abs(oracle.predict_at(rows,width)-refined)) < 1e-12
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
    for w in np.linspace(.6,1.,41):
        noiseless = [dict(r,value=float(y)) for r,y in zip(records,oracle.predict_at(inputs,w))]
        recovery = max(recovery,abs(oracle.Model().fit(noiseless).width-w))
    assert recovery < 1e-7
    rng = np.random.default_rng(614013)
    cases = [(ref.experiment(float(rng.uniform(0,1.8)),float(rng.uniform(-2,2)),
                            float(rng.uniform(0,np.pi)),float(rng.uniform(-np.pi,np.pi))),
              float(rng.uniform(.6,1.))) for _ in range(96)]
    cases += [(ref.experiment(d,h,t,p),w) for d in [0.,1.8] for h in [-2.,2.]
              for t in [0.,np.pi/2,np.pi] for p in [0.,np.pi] for w in [.6,1.]]
    direct_error, grid_error, trace_error, complement_error = 0.,0.,0.,0.
    for e,w in cases:
        value = oracle.predict_at([e],w)[0]
        direct = ref.predict([e],w)[0]
        direct_error = max(direct_error,abs(value-direct))
        d,h = e['displacement'],e['threshold']
        plus,minus = ndtr((d-h)/w),ndtr((-d-h)/w)
        c = np.exp(-d*d/(2*w*w))*ndtr(-h/w)
        cr = np.exp(-d*d/(2*w*w))*ndtr(h/w)
        assert c*c <= plus*minus+1e-15
        assert cr*cr <= (1-plus)*(1-minus)+1e-15
        trace_error = max(trace_error,abs(c+cr-np.exp(-d*d/(2*w*w))))
        acceptance = oracle.predict_at([dict(e,observable='acceptance')],w)[0]
        opposite = dict(e,theta=np.pi-e['theta'],phi=e['phi']+np.pi)
        complement_error = max(complement_error,abs(value+oracle.predict_at([opposite],w)[0]-acceptance))
        assert -1e-15 <= value <= acceptance+1e-15
        q = np.linspace(h,d+10*w,4097)
        grid = simpson([ref.density(x,e,w) for x in q],x=q)
        grid_error = max(grid_error,abs(grid-direct))
    assert direct_error < 1e-12 and grid_error < 1e-10
    assert trace_error < 1e-14 and complement_error < 1e-14
    report.update(noiseless_recovery_cases=41,noiseless_recovery_max_error=recovery,
                  domain_cases=len(cases),direct_integral_max_error=direct_error,
                  position_grid_max_error=grid_error,complementary_analyzer_max_error=complement_error,
                  total_coherence_max_error=trace_error,
                  invariants=['probability bounds','positive accepted/rejected maps',
                              'complementary analyzer probabilities','total pointer overlap'])
    path = ROOT/'results/expansion-coarse-pointer-r1-validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
