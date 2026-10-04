"""Controls for interacting-Gibbs qubit preparation (revision 9)."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import os
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/qubit-control'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


good=load(TASK/'solution/model.py','good')
bad=load(ROOT/'scripts/qubit_control_baseline.py','bad')
ref=load(TASK/'tests/reference.py','reference')


def run(count):
    records=json.loads((TASK/'tests/data/calibration.json').read_text())
    inputs=[r['experiment'] for r in records];sigma=np.array([r['sigma'] for r in records]);truth_cal=good.signal(good.signal_terms(inputs),.12)
    hidden=ref.hidden_inputs();truth={name:ref.predict(es,.12) for name,es in hidden.items()}
    hidden_cutoff=max(float(np.max(abs(truth[name]-ref.predict(es,.12,80)))) for name,es in hidden.items())
    def scores(model):return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2))) for name,es in hidden.items()}
    controls={}
    for name,module in [('oracle',good),('shortcut',bad)]:
        m=module.QubitModel().fit(records);res=(m.predict(inputs)-np.array([r['probability'] for r in records]))/sigma
        controls[name]=dict(gamma=m.gamma,calibration_chi2=float(res@res/(len(records)-1)),hidden=scores(m))
    rng=np.random.default_rng(56294);samples=[];fit_differences=[]
    for _ in range(count):
        noise=truth_cal+rng.normal(size=len(inputs))*sigma
        runs=[dict(experiment=e,probability=float(y),sigma=float(s)) for e,y,s in zip(inputs,noise,sigma)]
        a=good.QubitModel().fit(runs);b=bad.QubitModel().fit(runs);fit_differences.append(abs(a.gamma-b.gamma));assert fit_differences[-1]<1e-7
        residual=(a.predict(inputs)-noise)/sigma
        samples.append(dict(gamma=a.gamma,chi2=float(residual@residual/(len(inputs)-1)),oracle=scores(a),shortcut=scores(b)))
    # Check the full Fock Gibbs partition and spin populations, independently
    # of the completed-square analytic energy shift in the prediction models.
    check_cases=[]
    for gamma in (.04,.12,.25):
        for T in (.2,1.2):
            for w in ([1.2,1.2],[1.2,-1.1]):
                e=ref.experiment(.73,w,T,[[.2,1.3],[-.6,1.7]],[[.4,-1.4],[-.9,.8]])
                check_cases.append((gamma,e))
    for time in (np.pi, 6.7, 10.):
        check_cases.append((.25, ref.experiment(time,[1.2,1.2],1.2,[[0.,np.pi/2],[.7,1.7]],[[1.4,np.pi/2],[-.3,.9]])))
    max_reference=0.;max_cutoff=0.;min_eig=1.;max_trace=0.;stationarity=0.;partition=0.;marginal=0.;gibbs_energy=0.;joint_stationarity=0.
    for gamma,e in check_cases:
        density=ref.reduced_density(e,gamma,64);fine=ref.predict([e],gamma,80)[0]
        value=good.signal(good.signal_terms([e]),gamma)[0]
        max_reference=max(max_reference,abs(value-fine));max_cutoff=max(max_cutoff,abs(ref.predict([e],gamma,64)[0]-fine))
        min_eig=min(min_eig,float(np.min(np.linalg.eigvalsh(density))));max_trace=max(max_trace,abs(np.trace(density)-1))
        s=ref.Z@e['weights'];logs=np.zeros(4);energies=np.zeros(4)
        for omega,a in zip(ref.FREQUENCIES,ref.STRENGTHS):
            for j,v in enumerate(s):
                values,vectors=ref.oscillator(omega,a,gamma,float(v),64);logs[j]+=logsumexp(-values/e['temperature']);energies[j]+=softmax(-values/e['temperature'])@values
                conditional=(vectors*softmax(-values/e['temperature']))@vectors.T
                unitary=(vectors*np.exp(-1j*values*e['wait']))@vectors.T
                joint_stationarity=max(joint_stationarity,float(np.max(abs(unitary@conditional@unitary.conj().T-conditional))))
        exactlogs=gamma*s*s*np.sum(np.array(ref.STRENGTHS)/ref.FREQUENCIES)/e['temperature']-sum(np.log1p(-np.exp(-np.array(ref.FREQUENCIES)/e['temperature'])))
        partition=max(partition,float(max(abs(logs-exactlogs))))
        analytic_energy=sum(np.array(ref.FREQUENCIES)/np.expm1(np.array(ref.FREQUENCIES)/e['temperature']))-gamma*s*s*np.sum(np.array(ref.STRENGTHS)/ref.FREQUENCIES)
        gibbs_energy=max(gibbs_energy,float(max(abs(energies-analytic_energy))))
        p=softmax(logs);marginal=max(marginal,abs(p[0]+p[1]-.5),abs(p[0]+p[2]-.5))
        no_pulse=dict(e,preparation=[[0.,0.],[0.,0.]])
        stationary=ref.reduced_density(no_pulse,gamma,64)
        stationarity=max(stationarity,float(np.max(abs(stationary-np.diag(p)))))
    calibration_equivalence=max(float(np.max(abs(good.signal(good.signal_terms(inputs),g)-bad.signal(bad.signal_terms(inputs),g)))) for g in (.04,.12,.25))
    # Finite-Fock calibration checks a subset spanning every control type.
    subset=inputs[::7];cal_ref=float(np.max(abs(good.signal(good.signal_terms(subset),.12)-ref.predict(subset,.12))))
    zero_wait=[];no_coupling=[];recovery=[];probabilities=[]
    for g in (.04,.09,.12,.19,.25):
        clean=good.signal(good.signal_terms(inputs),g);m=good.QubitModel().fit([dict(experiment=e,probability=float(v),sigma=.002) for e,v in zip(inputs,clean)]);recovery.append(abs(m.gamma-g))
        for es in hidden.values():
            transformed=[dict(e,wait=0.) for e in es]
            zero_wait.append(float(max(abs(good.signal(good.signal_terms(transformed),g)-bad.signal(bad.signal_terms(transformed),g)))))
            disconnected=[dict(e,weights=[0.,0.]) for e in es]
            no_coupling.append(float(max(abs(good.signal(good.signal_terms(disconnected),g)-bad.signal(bad.signal_terms(disconnected),g)))))
            for module in (good,bad):probabilities.extend(module.signal(module.signal_terms(es),g).tolist())
    local={}
    for name,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/qubit_control_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='qubit-r9-'+name+'-') as directory:
            r=Path(directory);shutil.copytree(TASK/'environment',r/'app');shutil.copytree(TASK/'tests',r/'tests');shutil.copy2(path,r/'app/model.py')
            result=subprocess.run([sys.executable,'-m','pytest','-q',str(r/'app/test_public.py'),str(r/'tests/test_hidden.py')],cwd=r/'app',env=dict(os.environ,PYTHONPATH=str(r/'app'),OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
            local[name]=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    report=dict(revision=9,noise_trials=count,controls=controls,independent_reference_error=max_reference,
        fock64_to80_error=max_cutoff,all_hidden_fock64_to80_error=hidden_cutoff,hidden_reference_error=max(float(np.max(abs(good.signal(good.signal_terms(es),.12)-truth[name]))) for name,es in hidden.items()),
        calibration_reference_error=cal_ref,calibration_equivalence=calibration_equivalence,
        gibbs_partition_log_error=partition,gibbs_conditional_energy_error=gibbs_energy,
        full_joint_no_pulse_stationarity_error=joint_stationarity,
        individual_spin_maximally_mixed_error=marginal,no_pulse_stationarity_error=stationarity,
        minimum_spin_density_eigenvalue=min_eig,trace_error=float(max_trace),zero_wait_equivalence=max(zero_wait),
        zero_coupling_equivalence=max(no_coupling),noiseless_parameter_recovery_error=max(recovery),
        probability_range=[min(probabilities),max(probabilities)],
        noise=dict(maximum_control_fit_difference=max(fit_differences),calibration_passes=sum(s['chi2']<1.5 for s in samples),parameter_passes=sum(abs(s['gamma']/.12-1)<.03 for s in samples),
            oracle_passes=sum(max(s['oracle'].values())<.03 for s in samples),shortcut_passes=sum(max(s['shortcut'].values())<.03 for s in samples),
            maximum_oracle_error=max(max(s['oracle'].values()) for s in samples),minimum_shortcut_error=min(min(s['shortcut'].values()) for s in samples),
            maximum_chi2=max(s['chi2'] for s in samples),parameter_range=[min(s['gamma'] for s in samples),max(s['gamma'] for s in samples)]),local_pytest=local,noise_realizations=samples)
    assert calibration_equivalence<1e-12 and cal_ref<1e-10 and max_reference<1e-8 and max_cutoff<1e-8
    assert partition<1e-10 and gibbs_energy<1e-10 and stationarity<1e-10 and min_eig>-1e-12 and max_trace<1e-10
    assert min(probabilities)>-1e-12 and max(probabilities)<1+1e-12
    assert all(report['noise'][k]==count for k in ('calibration_passes','parameter_passes','oracle_passes'))
    assert report['noise']['shortcut_passes']==0 and report['noise']['minimum_shortcut_error']>.03
    assert local['oracle']['returncode']==0 and local['shortcut']['returncode']==1
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    if args.generate:
        inputs=ref.calibration_inputs();values=good.signal(good.signal_terms(inputs),.12);rng=np.random.default_rng(56291)
        # Full Fock generation is redundant for unrotated populations, checked
        # independently against the canonical partition function below.
        records=[dict(experiment=e,probability=float(v+.002*rng.normal()),sigma=.002) for e,v in zip(inputs,values)]
        for area in ('environment','tests'):(TASK/area/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    report=run(args.noise_trials);path=ROOT/'results/qubit-neutral-r4-validation.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('noise_realizations','local_pytest')},indent=2))
