"""Runnable scientific feasibility screen; no task edits or model trials."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
import prototype as p
import reference as ref

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def contrast(args,factorized=False,reference=False):
    a=list(args);a[2]=(a[2]+np.pi+np.pi)%(2*np.pi)-np.pi
    function=ref.stationary if reference else p.stationary
    kw={} if reference else dict(factorized=factorized)
    return (function(*args,**kw)['heat'][0]-function(*a,**kw)['heat'][0])/2


def groups():
    result={}
    for label,driven in (('static',0),('low_drive',1),('pair_drive',2)):
        cases=[]
        for j,(tl,tr) in enumerate(((.3,.9),(.6,.8),(.95,.4))):
            for rho,tph in ((.3,12.),(.7,24.)):
                for phi in (0.,.8):
                    amplitude=0. if not driven else (.6 if j%2==0 else 1.)
                    drive=.3 if driven<2 else (8. if j%2==0 else 11.)
                    cases.append((tl,tr,phi,amplitude,drive,rho,tph))
        result[label]=cases
    return result


def phase_space(value,temperature):
    if value==0: return temperature
    if value/temperature>40: return value
    if value/temperature < -40: return -value*np.exp(value/temperature)
    return value/(-np.expm1(-value/temperature))


def normal_closed(temperature,amplitude,drive,rho,tph,conductance=45.,count=24):
    ells,probs=p.sidebands(rho,tph,count)
    def current(bias):
        return sum(prob*(phase_space(bias-ell*p.OSCILLATOR_ENERGY,temperature)-phase_space(-bias-ell*p.OSCILLATOR_ENERGY,temperature)) for ell,prob in zip(ells,probs))
    ratio=p.r4.QUASIPARTICLE_DENSITY/(p.r4.NORMAL_POLARIZATION_TIME*p.r4.RATE_CONVERSION*conductance)
    mu=brentq(lambda m:amplitude**2/2*current(drive-2*m)-ratio*m,0.,drive/2,xtol=5e-15)
    return mu


def main():
    start=time.time();out={'proposal_only':True,'groups':{},'domain':[],'limits':{},'sources':{}}
    for name,cases in groups().items():
        mean=[];source_mean=[];target=[];source=[];independent=[]
        for args in cases:
            mean.append(p.stationary(*args)['heat'][0]);source_mean.append(p.stationary(*args,factorized=True)['heat'][0])
            target.append(contrast(args));source.append(contrast(args,True));independent.append(contrast(args,reference=True))
        target,source=np.array(target),np.array(source)
        out['groups'][name]=dict(inputs=cases,mean=mean,source_mean=source_mean,
                                mean_relative_rmse=float(np.sqrt(np.mean((np.array(mean)-source_mean)**2)/np.mean(np.array(mean)**2))),
                                contrast=target.tolist(),source_contrast=source.tolist(),
                                contrast_relative_rmse=float(np.sqrt(np.mean((target-source)**2)/np.mean(target**2))),
                                independent_contrast_max_pW=float(max(abs(target-np.array(independent)))),
                                contrast_signal_min_pW=float(min(abs(target))),contrast_signal_rms_pW=float(np.sqrt(np.mean(target**2))))
        print('Group',name,flush=True)
    rng=np.random.default_rng(931006)
    corner=[]
    for j in range(64):
        tl=(.25,1.05)[j&1];tr=(.25,1.05)[(j>>1)&1]
        a=(0.,1.)[(j>>2)&1];drive=(.1,.5,7.,12.)[(j>>3)&3]
        rho=(0.,1.)[(j>>5)&1];tph=(12.,24.)[(j>>4)&1]
        corner.append((tl,tr,(0.,np.pi)[j%2],a,drive,rho,tph))
    for _ in range(32):
        tl,tr=rng.uniform(.25,1.05,2);phase=rng.uniform(-np.pi,np.pi);a=rng.uniform(0,1)
        drive=rng.uniform(.1,.5) if rng.uniform()<.5 else rng.uniform(7,12)
        corner.append((tl,tr,phase,a,drive,rng.uniform(0,1),rng.uniform(12,24)))
    for j,args in enumerate(corner):
        exact=p.stationary(*args);source=p.stationary(*args,factorized=True)
        independent=ref.stationary(*args)
        refined=p.stationary(*args,nodes=288,count=32)
        mu=exact['mu'];heat=exact['heat'];spin=exact['spin'];jac=exact['jacobian']
        ells,probs=p.sidebands(args[5],args[6],24)
        _,refined_probs=p.sidebands(args[5],args[6],32)
        fock=ref.fock_covariances(args[5],args[6],24)
        expected_dw=np.exp(-2*args[5]*(2/np.expm1(24/args[6])+1))
        entropy=exact['entropy']+exact['flip_entropy']
        thermo=-heat[0]/args[0]-heat[1]/args[1]+heat[3]/args[6]
        row=dict(input=args,polarization_K=mu.tolist(),heat_pW=heat.tolist(),source_heat_pW=source['heat'].tolist(),
                 source_entropy_W_per_K=float(source['entropy']+source['flip_entropy']),
                 independent_heat_error_pW=float(max(abs(heat-independent['heat']))),
                 independent_mu_error_K=float(max(abs(mu-independent['mu']))),
                 refined_heat_error_pW=float(max(abs(heat-refined['heat']))),
                 refined_mu_error_K=float(max(abs(mu-refined['mu']))),
                 energy_identity_pW=float(heat[0]+heat[1]+heat[2]-heat[3]),
                 spin_work_identity_pW=float(heat[2]-p.r4.KB*args[4]*sum(spin)/2*1e12),
                 entropy_W_per_K=float(entropy),entropy_identity_W_per_K=float(entropy-thermo*1e-12),
                 minimum_rate=exact['minimum_rate'],balance_residual_normalized=float(max(abs(exact['residual']))),
                 jacobian_column_dominance_min=float(min(-np.diag(jac)-np.sum(abs(jac),axis=0)+abs(np.diag(jac)))),
                 probabilities_total=float(sum(probs)),probability_tail=float(sum(refined_probs)-sum(probs)),
                 dephasing_sum_error=float(sum((-1.)**ells*probs)-expected_dw),
                 fock_probability_error=float(max(abs(probs-np.array([c[1] for c in fock])))),
                 fock_cross_error=float(max(abs((-1.)**ells*probs-np.array([c[3] for c in fock])))),
                 search_bound=exact['search_bound'],search_expansions=exact['expansions'])
        out['domain'].append(row)
        if (j+1)%16==0: print('Domain',j+1,flush=True)
    old=[]
    for args in [(a,b,ph,c,w,0.,12.) for a,b in ((.25,1.05),(.65,.9),(1.05,.25)) for ph in (0.,2.) for c,w in ((0.,.3),(.7,.3),(.7,9.))]:
        actual=p.stationary(*args);prior=p.r4.stationary_state(*args[:5],45.)
        old.append(float(max(abs(actual['heat'][:3]-prior['heat']))))
    normal=[]
    for temp in (.25,1.05):
        for drive in (.3,9.):
            args=(temp,temp,.7,.8,drive,.7,24.)
            actual=p.stationary(*args,normal=True);approx=p.stationary(*args,normal=True,factorized=True)
            independent=ref.stationary(*args,normal=True)
            closed=normal_closed(temp,.8,drive,.7,24.)
            normal.append(dict(input=args,mu=actual['mu'].tolist(),closed_mu=closed,
                               closed_mu_error=float(max(abs(actual['mu']-closed))),
                               independent_heat_error_pW=float(max(abs(actual['heat']-independent['heat']))),
                               closure_equality_pW=float(max(abs(actual['heat']-approx['heat'])))))
    out['limits']=dict(rho_zero_r4_heat_error_pW=max(old),normal_state=normal)
    for filename in ('prototype.py','reference.py','screen.py'):
        out['sources'][filename]=hashlib.sha256((HERE/filename).read_bytes()).hexdigest()
    out['seconds']=time.time()-start
    target=ROOT/'results/superconducting-heat-r5-oscillator-proposal-screen.json'
    target.write_text(json.dumps(out,indent=2)+'\n')
    concise=dict(seconds=out['seconds'],groups=out['groups'],domain_summary={
        key:(min(r[key] for r in out['domain']) if key in ('entropy_W_per_K','source_entropy_W_per_K','minimum_rate','jacobian_column_dominance_min') else max(abs(r[key]) for r in out['domain']))
        for key in ('independent_heat_error_pW','independent_mu_error_K','refined_heat_error_pW','refined_mu_error_K','energy_identity_pW','spin_work_identity_pW','entropy_W_per_K','source_entropy_W_per_K','entropy_identity_W_per_K','minimum_rate','balance_residual_normalized','jacobian_column_dominance_min','probability_tail','dephasing_sum_error','fock_probability_error','fock_cross_error')},limits=out['limits'])
    print(json.dumps(concise,indent=2))


if __name__=='__main__':main()
