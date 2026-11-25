"""Scientific controls for coupled-device quantum calorimetry."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/quantum-calorimetry'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    args = parser.parse_args()
    start = time.time()
    good = load('quantum_good', TASK/'solution/model.py')
    bad = load('quantum_bad', ROOT/'scripts/quantum_calorimetry_baseline.py')
    ref = load('quantum_reference', TASK/'tests/reference.py')
    prototype = load('quantum_prototype', ROOT/'scripts/prototype_quantum_calorimetry.py')
    meta = json.loads((TASK/'tests/metadata.json').read_text())
    true = ref.TRUE_PARAMETER
    sigma = meta['measurement_sigma']
    inputs = ref.calibration_inputs()
    clean = ref.predict(inputs)
    if args.generate:
        values = clean+sigma*np.random.default_rng(meta['calibration_seed']).normal(size=len(inputs))
        records = [dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        for directory in ['environment','tests']:
            (TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes() == (TASK/'tests/data/calibration.json').read_bytes()
    assert [r['input'] for r in records] == inputs and {r['sigma'] for r in records} == {sigma}
    hidden = ref.hidden_inputs()
    truth = {k:ref.predict(es) for k,es in hidden.items()}
    discriminating = [k for k in hidden if k != 'uncoupled_anchors']
    def score(model):
        return {k:float(np.sqrt(np.mean((model.predict(es)-truth[k])**2)/np.mean(truth[k]**2))) for k,es in hidden.items()}
    def chi(model, rs):
        residual = (model.predict(inputs)-np.array([r['value'] for r in rs]))/sigma
        return float(residual@residual)/(len(rs)-1)
    report = dict(task='quantum-calorimetry',revision=1,noise_trials=args.noise_trials,seeds=dict(calibration=meta['calibration_seed'],noise=meta['noise_seed']),controls={})
    for name,source in [('oracle',good),('shortcut',bad)]:
        model = source.Model().fit(records)
        result = dict(parameter=model.natural_frequency,parameter_relative_error=abs(model.natural_frequency/true-1),calibration_chi2=chi(model,records),hidden=score(model))
        assert result['calibration_chi2'] < 1.5 and result['parameter_relative_error'] < .03
        assert max(result['hidden'].values()) < .04 if name == 'oracle' else min(result['hidden'][k] for k in discriminating) > .04
        assert result['hidden']['uncoupled_anchors'] < .04
        report['controls'][name] = result
    rng = np.random.default_rng(meta['noise_seed'])
    parameters=[];chi2=[];correct=[];incorrect=[]
    for _ in range(args.noise_trials):
        values = clean+sigma*rng.normal(size=len(inputs))
        sample = [dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        a=good.Model().fit(sample);b=bad.Model().fit(sample)
        assert a.natural_frequency == b.natural_frequency
        parameters.append(a.natural_frequency);chi2.append(chi(a,sample))
        correct.append(max(score(a).values()))
        incorrect.append(min(score(b)[k] for k in discriminating))
    assert max(chi2)<1.5 and max(abs(np.array(parameters)/true-1))<.03
    assert max(correct)<.04 and min(incorrect)>.04
    report['noise']=dict(calibration_passes=args.noise_trials,parameter_passes=args.noise_trials,oracle_passes=args.noise_trials,shortcut_rejections=args.noise_trials,maximum_chi2=max(chi2),parameter_min=min(parameters),parameter_max=max(parameters),oracle_hidden_max=max(correct),shortcut_discriminating_min=min(incorrect))
    # All scored independent finite-bath references, with separate mode and high-frequency refinements.
    comparisons=[]
    for key,es in hidden.items():
        for e in es:
            exact=good.heat_capacity(e,true)
            base=ref.heat_capacity(e,true)
            refined=ref.heat_capacity(e,true,28,64)
            extended=ref.heat_capacity(e,true,28,128)
            comparisons.append(dict(input=e,oracle=exact,reference=base,oracle_reference_error=abs(exact-base),basis_refinement=abs(refined-base),tail_refinement=abs(extended-refined),extended_reference_error=abs(exact-extended)))
    assert max(r['oracle_reference_error'] for r in comparisons)<2e-6
    assert max(r['basis_refinement'] for r in comparisons)<2e-7
    assert max(r['extended_reference_error'] for r in comparisons)<2e-7
    corners=[];uncertainty=[];passivity=[];density_integral=[]
    for frequency in [.8,1.2]:
        for damping in [1.,3.]:
            for cutoff in [1.,4.]:
                for temperature in [.1,1.]:
                    e=ref.experiment(temperature,damping,cutoff)
                    exact=good.heat_capacity(e,frequency);independent=ref.heat_capacity(e,frequency,24,128)
                    corners.append(abs(exact-independent))
                    q2,p2,_,_=bad.stationary_moments(temperature,damping,cutoff,frequency)
                    uncertainty.append(q2*p2)
                    assert exact>0
                omega=np.geomspace(1e-6,1e4,1000)
                passivity.append(float(np.min(bad.susceptibility(omega,frequency,damping,cutoff).imag)))
                density_integral.append(abs(quad(lambda w:prototype.spectral_shift(w,frequency,damping,cutoff),0,np.inf,epsabs=1e-10)[0]-1))
    assert max(corners)<3e-6 and min(uncertainty)>=.25 and min(passivity)>0 and max(density_integral)<1e-8
    # Direct finite-bath bare-oscillator energy derivatives check the completed shortcut itself.
    local_checks=[]
    for temperature,frequency,damping,cutoff in [(.35,1.04,3.,1.),(.6,1.04,2.,2.),(1.,1.04,3.,4.)]:
        e=ref.experiment(temperature,damping,cutoff)
        direct=prototype.finite_bath(temperature,frequency,damping,cutoff,24,64,True)[3]
        local_checks.append(abs(bad.heat_capacity(e,frequency)-direct))
    assert max(local_checks)<2e-7
    low=good.heat_capacity(ref.experiment(1e-4,2.,1.),true)
    low_relative=abs(low/(np.pi*2*1e-4/(3*true**2))-1)
    high=good.heat_capacity(ref.experiment(100.,3.,1.),true)
    assert low_relative<1e-5 and abs(high-1)<1e-4
    recovery=[]
    for frequency in [.8,.9,1.04,1.1,1.2]:
        values=ref.predict(inputs,frequency)
        fitted=good.Model().fit([dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,values)])
        recovery.append(abs(fitted.natural_frequency-frequency))
    assert max(recovery)<1e-6
    report['physical_checks']=dict(hidden_comparisons=comparisons,corner_reference_error=max(corners),minimum_covariance_uncertainty_product=min(uncertainty),minimum_susceptibility_imaginary_part=min(passivity),spectral_shift_integral_error=max(density_integral),shortcut_finite_bath_local_energy_error=max(local_checks),low_temperature_asymptotic_relative_error=low_relative,high_temperature_capacity=high,noiseless_frequency_recovery_max=max(recovery),exact_calibration_equivalence=float(np.max(abs(good.predict_at(inputs,true)-bad.predict_at(inputs,true)))))
    report['seconds']=time.time()-start
    files=[TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/quantum_calorimetry_baseline.py',Path(__file__)]
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    output=ROOT/'results/quantum-calorimetry-validation.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(seconds=report['seconds'],controls=report['controls'],noise=report['noise'],max_ref=max(r['oracle_reference_error'] for r in comparisons),corner_ref=max(corners)),indent=2))


if __name__ == '__main__':
    main()
