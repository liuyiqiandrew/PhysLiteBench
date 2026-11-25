import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('p','/private/tmp/prototype_paired_spin_noise.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
path=Path('results/paired-spin-noise-prototype.json');report=json.loads(path.read_text())
selected=[]
for t,g,T,site,c in [(.75,.55,.55,0,2.5),(.75,.7,.55,1,3.),(1.,.7,.55,2,2.5),(.65,.55,.7,2,2.5),(.9,.9,.55,1,3.5),(.8,.8,.45,0,3.)]:
 e=dict(hopping=t,pairing=g,temperature=T,site=site,center=c,width=.6,phase=0.,offset=0.)
 y=p.at(e);z=p.at(e,False);r=p.ref(e);selected.append({'input':e,'correct':y,'shortcut':z,'reference':r,'relative_error':abs(z-y)/y,'absolute_gap':abs(z-y),'global_spin_correct':p.signal(p.gaussian_lines(e,True,[.5,-.5]*3),e),'global_spin_shortcut':p.signal(p.gaussian_lines(e,False,[.5,-.5]*3),e)})
cal=[]
for t in [.55,.8,1.]:
 for site in [0,1,2]:
  for T in [.3,.6]:
   e=dict(hopping=t,pairing=0.,temperature=T,site=site,center=2.,width=.7,phase=0.,offset=0.)
   cal.append(p.at(e))
x=np.array(cal);gain=1.1;sigma=.0005;rng=np.random.default_rng(739391);y=gain*x[None,:]+rng.normal(size=(256,len(x)))*sigma;fitted=y@x/(x@x)
kms={};pos={};sumrule=0.;normal_propagator=0.
for complete in [False,True]:
 errors=[];neg=[]
 for item in selected:
  e=item['input'];f,w=p.gaussian_lines(e,complete);bins={}
  for a,b in zip(f.ravel(),w.ravel()):
   key=round(float(a),10);bins[key]=bins.get(key,0.)+float(b)
  errors.extend(abs(bins.get(round(-a,10),0.)-np.exp(-a/e['temperature'])*b) for a,b in bins.items() if a>0)
  neg.append(min(bins.values()))
  if complete:
   F,W=p.fock_lines(e);sumrule=max(sumrule,abs(w.sum()-W.sum()))
  # Check exact normal propagators independently in Fock space at three times.
  h,pair=p.matrices(e);H=np.zeros((64,64),complex)
  for a in range(p.N):
   for b in range(p.N):
    H+=h[a,b]*p.ANN[a].conj().T@p.ANN[b]
    if a<b:
     z=pair[a,b]*p.ANN[a].conj().T@p.ANN[b].conj().T;H+=z+z.conj().T
  en,V=p.eigh(H);pop=np.exp(-(en-en[0])/e['temperature']);pop/=pop.sum();B=np.block([[h,pair],[-pair.conj(),-h.T]]);ev,Q=p.eigh(B)
  for tt in [0.,.3,1.1]:
   K=(Q*(np.exp(-1j*ev*tt)*p.expit(ev/e['temperature'])))@Q.conj().T
   a=2*e['site'];op=V.conj().T@p.ANN[a]@V
   direct=np.sum(pop[:,None]*np.exp(1j*(en[:,None]-en[None,:])*tt)*abs(op)**2)
   normal_propagator=max(normal_propagator,abs(direct-K[a,a]))
 kms[str(complete)]=max(errors);pos[str(complete)]=min(neg)
report.update({'selected_examples':selected,'calibration':{'normal_pairing':0,'cases':18,'minimum_response':float(x.min()),'maximum_response':float(x.max()),'gain_identifiable':'Strictly positive scalar normal-state response makes the least-squares objective strictly convex in the sole gain.','gain_truth':gain,'fixed_instrument_sigma':sigma,'noise_draws':256,'largest_relative_gain_error':float(np.max(abs(fitted-gain))/gain),'fisher_information':float(x@x/sigma**2)},'additional_checks':{'kms_positive_frequency_max_absolute':kms,'minimum_grouped_spectral_weight':pos,'full_equal_time_sum_rule_vs_fock_error':sumrule,'exact_normal_propagator_vs_fock_error':float(normal_propagator)},'apparatus':{'system':'Three sites with two spin orbitals each, six fermionic modes. The given quadratic Hamiltonian is exact; pairing is imposed by rigid classical proximity sources. No charging energy or parity restriction. Each repetition begins in exp(-H/T)/Z over the complete 64-state Fock space.','hamiltonian':'H=sum_ab h_ab c_a^dagger c_b + sum_i[Delta_i c_iup^dagger c_idown^dagger+h.c.]; h_spin=kron(h_site,I2), site energies[-.35,.15,.7]+offset, hoppings -hopping*[1,.8,.3] on links01,12,02; Delta_i=pairing*[1,.85,1.15]*exp(i*phase*[0,1,-.4]).','observable':'A weak detector couples to only S_z at the specified single site, (n_up-n_down)/2. Excited two-state detector probes of positive gap omega lose excitation into the sample. Their known spectral weight is omega^2*exp[-(omega-center)^2/(2*width^2)] for omega>0. Readout is the summed leading-order downward detector rate, with 2*pi and detector density absorbed into the sole unknown gain. Positive omega means energy absorbed by the sample. No sample lifetime broadening is introduced.','shortcut':'Exact full BdG normal propagators, then normal density Wick bubble only. Two spin sectors are modeled as statistically independent Gaussian auxiliary systems with the exact normal one-particle spectral functions.','oracle':'The same complete Gaussian state including anomalous pair contractions in the connected local spin-density correlation.','reference':'Independently construct fermionic creation/annihilation matrices on the 64-state occupation basis, diagonalize H, and sum exact squared local-Sz transition matrix elements weighted by Gibbs probabilities and detector bandpass.'},'limits_and_caveats':{'global_spin':'The full model conserves total Sz and gives zero finite-frequency total-spin spectrum. The independent-spin approximation violates that Ward identity; selected examples record its spurious global response explicitly. The proposed measured observable is strictly local and has healthy nonzero response.','overlap':'Related to retained superconducting-heat through BCS electron-hole coherence. There the closure freezes an energy-dependent tunneling coherence ratio; here it replaces an exact paired two-particle covariance by a positive independent-spin spectral convolution after retaining all normal one-particle dynamics. This is a different observable and approximation but not an independent broad physical family. Missing cross-covariance also overlaps several screened candidates.','difficulty':'Physical validity and large response separation do not establish model difficulty. A competent solver may implement the small exact Fock Hamiltonian directly; no task build or evaluation has been performed.','recommendation':'Keep as a documented prototype fallback rather than count it as a clearly distinct benchmark family without root approval.'},'primary_sources':[{'url':'https://journals.aps.org/pr/abstract/10.1103/PhysRev.113.1504','title':'Hebel and Slichter, Nuclear Spin Relaxation in Normal and Superconducting Aluminum (1959)','supports':'Measured nuclear spin relaxation and the role of opposite-spin correlations in superconducting response.'},{'url':'https://arxiv.org/abs/1605.08164','title':'Nagai and Ota, Nuclear magnetic relaxation rates of unconventional superconductivity in doped topological insulators (2016)','supports':'Normal and anomalous spectral contributions to the spin-relaxation coherence response; background rather than the finite-system implementation.'}],'prototype_source_sha256':{str(z):hashlib.sha256(z.read_bytes()).hexdigest() for z in [Path('/private/tmp/prototype_paired_spin_noise.py'),Path('/private/tmp/finish_paired_spin_noise_prototype.py')]}})
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'selected':selected,'calibration':report['calibration'],'checks':report['additional_checks']},indent=2))
