"""Read-only independent controls for the finite-bias rotor proposal."""
from pathlib import Path
import hashlib
import importlib.util
import itertools
import json
import time
import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[3]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


spectral = module(Path(__file__).with_name('spectral.py'), 'independent_spectral')
fv = module(ROOT/'results/prototypes/rotating-reservoir-r7-root/renewal_screen.py', 'positive_fv')
moments = module(ROOT/'archives/rotating-reservoir-r6/tasks/rotating-reservoir/solution/model.py', 'immutable_r6')
reference = module(ROOT/'archives/rotating-reservoir-r6/tasks/rotating-reservoir/tests/reference.py', 'independent_r6')


def laguerre_renewal(chi, ta, tb, gamma=.67, rate=3.6, points=96):
    nodes, weights = laggauss(points)
    coefficient = chi*(ta-tb)-chi*chi*ta*tb
    def residual(theta):
        times = nodes/(rate+theta)
        transform = (1+coefficient*(-np.expm1(-2*gamma*times)))**-.5
        return rate/(rate+theta)*(weights@transform)-1
    return float(brentq(residual, -rate+1e-4, 2., xtol=2e-13))


def direct_gaussian_cycle(chi, ta, tb, gamma, duration):
    nodes, weights = hermgauss(48)
    gas = np.sqrt(2*tb)*nodes[:, None]
    after = np.exp(-gamma*duration)*gas + np.sqrt(2*ta*(-np.expm1(-2*gamma*duration)))*nodes[None, :]
    heat = (after*after-gas*gas)/2
    return float(weights@np.exp(-chi*heat)@weights/np.pi)


def parameters(e):
    return (e['heat_bias'],e['temperature_a'],e['temperature_b'],
            e['angular_speed'],e['flux_fraction'],e['flip_rate'],e['drag'])


def run():
    start = time.perf_counter()
    source = json.loads((ROOT/'results/prototypes/rotating-reservoir-r7-root/renewal_screen.json').read_text())
    groups = []
    for root_row in source['cases']:
        e = dict(root_row['input'], drag=.67)
        args = parameters(e)
        coarse = spectral.scgf(*args, points=91, quadrature=100, extent=7.)
        normal = spectral.scgf(*args)
        fine = spectral.scgf(*args, points=151, quadrature=180, extent=11.)
        tail = spectral.scgf(*args, points=181, quadrature=220, extent=13.)
        robin = spectral.scgf(*args, gauge='full', boundary='reflecting')
        gauge_neumann = spectral.scgf(*args, boundary='gauge_neumann')
        positive = fv.extrapolated(*args, points=161, extent=9.)
        positive_fine = fv.extrapolated(*args, points=241, extent=11.)
        state = moments.physical_statistics(e,.67)
        gaussian = -e['heat_bias']*state['mean'] + e['heat_bias']**2*state['noise']/2
        row = dict(input=e, coarse=coarse, normal=normal, fine=fine, tail=tail,
                   alternate_full_gauge_robin=robin, alternate_gauge_neumann=gauge_neumann,
                   positive_fv=positive, positive_fv_refined=positive_fine,
                   gaussian=gaussian, source_relative_error=abs(gaussian-fine['value'])/abs(fine['value']))
        groups.append(row)
        print(json.dumps(dict(group=e,scgf=fine['value'],relative_gap=row['source_relative_error'],
                              independent_error=abs(positive_fine-fine['value']))),flush=True)
    corners = []
    for gamma,ta,tb,omega,flip,eta,chi in itertools.product(
            [.4,1.1],[.6,1.6],[.6,1.4],[-.65,.65],[.8,2.],[0.,1.],[-.35,.35]):
        args = (chi,ta,tb,omega,eta,flip,gamma)
        normal = spectral.scgf(*args)
        fine = spectral.scgf(*args, points=151, quadrature=180, extent=11.)
        corners.append(dict(input=dict(drag=gamma,temperature_a=ta,temperature_b=tb,
            angular_speed=omega,flip_rate=flip,flux_fraction=eta,heat_bias=chi),
            normal=normal,fine=fine))
    print('completed full-box corners',len(corners),flush=True)
    random = []
    rng = np.random.default_rng(713106)
    for i in range(24):
        gamma=rng.uniform(.4,1.1);ta=rng.uniform(.6,1.6);tb=rng.uniform(.6,1.4)
        omega=rng.uniform(-.65,.65);flip=rng.uniform(.8,2.);eta=rng.uniform(0,1);chi=rng.uniform(-.35,.35)
        args=(chi,ta,tb,omega,eta,flip,gamma)
        a=spectral.scgf(*args);b=spectral.scgf(*args,points=151,extent=11.,quadrature=180)
        exactfv=fv.extrapolated(*args,points=161,extent=9.)
        random.append(dict(input=dict(drag=gamma,temperature_a=ta,temperature_b=tb,
            angular_speed=omega,flip_rate=flip,flux_fraction=eta,heat_bias=chi),
            normal=a,fine=b,positive_fv=exactfv))
    renewal = []
    for ta,tb,chi in itertools.product([.6,1.,1.6],[.6,1.,1.4],[-.35,.35]):
        independent=laguerre_renewal(chi,ta,tb)
        continuous=spectral.scgf(chi,ta,tb,points=151,extent=11.,quadrature=180)['value']
        root_quad=fv.renewal(chi,ta,tb)
        renewal.append(dict(ta=ta,tb=tb,chi=chi,independent=independent,
            continuous=continuous,root_quad=root_quad))
    gaussian_cycles=[]
    for ta,tb,chi,duration in itertools.product([.6,1.6],[.6,1.4],[-.35,.35],[.1,1.,4.]):
        exact=(1+(-np.expm1(-2*.67*duration))*(chi*(ta-tb)-chi*chi*ta*tb))**-.5
        direct=direct_gaussian_cycle(chi,ta,tb,.67,duration)
        gaussian_cycles.append(dict(ta=ta,tb=tb,chi=chi,time=duration,exact=exact,direct=direct))
    gc=[]
    for ta,tb,eta,chi in itertools.product([1.,1.2],[1.,1.1],[0.,1.],[-.15,.15]):
        reflection=1/tb-1/ta-chi
        assert abs(reflection)<1/max(ta,tb)
        a=spectral.scgf(chi,ta,tb,eta=eta)['value']
        b=spectral.scgf(reflection,ta,tb,eta=eta)['value']
        gc.append(dict(ta=ta,tb=tb,eta=eta,chi=chi,reflected_bias=reflection,
            both_inside_public_bias_range=abs(reflection)<=.35,value=a,reflected_value=b,error=abs(a-b)))
    cumulants=[]
    for ta,tb,omega,eta,flip,gamma in [(1.,1.,0.,0.,1.,.67),(.6,1.4,.65,1.,.8,.4),
            (1.6,.6,-.65,1.,2.,1.1),(1.4,1.4,.65,1.,1.,.67),(.8,1.2,-.5,.4,1.6,.8)]:
        base=(ta,tb,omega,eta,flip,gamma)
        zero=spectral.scgf(0.,*base)['value']
        d1=.002;d2=.001
        estimates=[]
        for delta in [d1,d2]:
            plus=spectral.scgf(delta,*base)['value'];minus=spectral.scgf(-delta,*base)['value']
            estimates.append(dict(mean=-(plus-minus)/(2*delta),noise=(plus+minus-2*zero)/delta**2))
        extrapolated={key:(4*estimates[1][key]-estimates[0][key])/3 for key in ['mean','noise']}
        exact=reference.state(gamma,ta,tb,omega,flip,eta)
        cumulants.append(dict(input=dict(drag=gamma,temperature_a=ta,temperature_b=tb,
            angular_speed=omega,flip_rate=flip,flux_fraction=eta),zero=zero,estimates=estimates,
            extrapolated=extrapolated,independent_mean=exact['mean'],independent_noise=exact['noise']))
    summary=dict(group_refinement_max=max(abs(r['fine']['value']-r['normal']['value']) for r in groups),
        group_tail_max=max(abs(r['tail']['value']-r['fine']['value']) for r in groups),
        group_robin_gauge_max=max(abs(r['alternate_full_gauge_robin']['value']-r['normal']['value']) for r in groups),
        group_gauge_neumann_max=max(abs(r['alternate_gauge_neumann']['value']-r['normal']['value']) for r in groups),
        independent_refined_fv_max=max(abs(r['positive_fv_refined']-r['fine']['value']) for r in groups),
        corners_refinement_max=max(abs(r['normal']['value']-r['fine']['value']) for r in corners),
        random_refinement_max=max(abs(r['normal']['value']-r['fine']['value']) for r in random),
        random_independent_fv_max=max(abs(r['positive_fv']-r['fine']['value']) for r in random),
        renewal_independent_max=max(abs(r['independent']-r['continuous']) for r in renewal),
        renewal_quadrature_max=max(abs(r['independent']-r['root_quad']) for r in renewal),
        direct_gaussian_cycle_max=max(abs(r['direct']-r['exact']) for r in gaussian_cycles),
        gc_max=max(r['error'] for r in gc),
        zero_bias_max=max(abs(r['zero']) for r in cumulants),
        mean_cumulant_max=max(abs(r['extrapolated']['mean']-r['independent_mean']) for r in cumulants),
        noise_cumulant_max=max(abs(r['extrapolated']['noise']-r['independent_noise']) for r in cumulants),
        minimum_spectral_gap=min(r['fine']['spectral_gap'] for r in groups+corners+random),
        maximum_principal_imaginary=max(r['fine']['imaginary'] for r in groups+corners+random))
    assert summary['group_refinement_max']<1e-8
    assert summary['group_tail_max']<1e-8
    assert summary['corners_refinement_max']<1e-8
    assert summary['random_refinement_max']<1e-8
    assert summary['independent_refined_fv_max']<2e-6
    assert summary['renewal_independent_max']<1e-9
    assert summary['renewal_quadrature_max']<1e-10
    assert summary['direct_gaussian_cycle_max']<1e-10
    assert summary['gc_max']<1e-9
    assert summary['noise_cumulant_max']<2e-6
    assert summary['minimum_spectral_gap']>0.5
    assert summary['maximum_principal_imaginary']<1e-10
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path(__file__),Path(__file__).with_name('spectral.py'),
        ROOT/'results/prototypes/rotating-reservoir-r7-root/renewal_screen.py',
        ROOT/'archives/rotating-reservoir-r6/preservation-manifest.json']}
    result=dict(status='passed_independent_physics_proposal',revision=7,summary=summary,
        groups=groups,corners=corners,random_cases=random,renewal=renewal,
        direct_gaussian_cycles=gaussian_cycles,gc=gc,cumulants=cumulants,source_sha256=hashes,
        domain_proof='For every stated preparation and |chi|<=.35, use f(v)=exp[a v^2/2], a=.4-.5*chi. Then a>=.225, a<.625<=1/Ta, a+chi>=.225 and a+chi<.7142857<=1/Tb. The tilted jump gain divided by f has negative quadratic v exponent and finite Gaussian gas integral; the OU term has negative quadratic v coefficient. Motor flips leave f fixed. A uniform finite upper bound L_chi f/f, together with finite stationary E[f], proves finite exponential growth on the full public box. This is a conservative valid window, not a claim of maximal heat-transform analyticity.',
        physical_statement='Finite exponential bias probes the joint tilted path ensemble and all current cumulants. The completed Gaussian long-time current closure preserves exact collision dynamics, stationary mean and zero-frequency noise but discards higher heat cumulants. Bulk mean/noise calibration and fitted drag are unchanged.',
        constraints='No task edits, grading changes, Docker runs, paid model evaluations or refits were performed.',
        seconds=time.perf_counter()-start)
    target=ROOT/'results/rotating-reservoir-r7-brownian-proposal-audit.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(saved=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),seconds=result['seconds'],summary=summary)),flush=True)


if __name__=='__main__':
    run()
