"""Proposal only: thermal quantum phase oscillator on the frozen r4 contact.

Energies are kelvin. ell>0 denotes oscillator gain. This file is not a task
oracle or verifier and imports r4 only for constants, gap and spin loss.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit, ive
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('frozen_r4', ROOT/'archives/superconducting-heat-r4/tasks/superconducting-heat/solution/model.py')
r4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r4)
OSCILLATOR_ENERGY = 24.


def sidebands(rho, temperature, count=24):
    ell = np.arange(-count, count+1)
    if rho == 0:
        return ell, (ell == 0).astype(float)
    nb = 1/np.expm1(OSCILLATOR_ENERGY/temperature)
    argument = 2*rho*np.sqrt(nb*(nb+1))
    probabilities = ive(abs(ell), argument)*np.exp(
        -rho*(2*nb+1)+argument+ell*OSCILLATOR_ENERGY/(2*temperature))
    return ell, probabilities


@lru_cache(maxsize=512)
def prepare(left, right, phase, amplitude, drive, rho, oscillator_temperature,
            nodes=192, count=24, window=40., normal=False, factorized=False):
    dl, dr = (0., 0.) if normal else (r4.gap(left, 1.2), r4.gap(right, 1.9))
    x, weights = leggauss(nodes)
    ells, probabilities = sidebands(rho, oscillator_temperature, count)
    nb = 1/np.expm1(OSCILLATOR_ENERGY/oscillator_temperature)
    dephasing = np.exp(-2*rho*(2*nb+1))
    upper = np.sqrt(80*max(left, right)+window)
    z = (x+1)*upper/2
    angle = (x+1)*np.pi/4
    channels = []
    for n in (-1, 0, 1):
        if n and amplitude == 0:
            continue
        transfer_spins = ((1, 1, .5), (-1, -1, .5)) if n == 0 else ((-n, n, amplitude**2/2),)
        pair_spins = ((1, -1, .5), (-1, 1, .5)) if n == 0 else ((n, n, amplitude**2/2),)
        for ell, probability, reverse_probability in zip(ells, probabilities, probabilities[::-1]):
            if probability == reverse_probability == 0:
                continue
            shift = n*drive-ell*OSCILLATOR_ENERGY
            parity = dephasing if factorized else (-1.)**int(ell)
            sign = (-1 if n == 0 else 1)*parity
            edge = max(dl, dr-shift)
            energy, other = edge+z*z, edge+z*z+shift
            denominator = np.sqrt((energy-dl)*(energy+dl)*(other-dr)*(other+dr))
            measure = weights*upper/2*2*z*(energy*other+sign*dl*dr*np.cos(phase))/denominator
            for sl, sr, weight in transfer_spins:
                channels.append((energy, other, measure*weight, sl, sr, False,
                                 n*drive, ell*OSCILLATOR_ENERGY, probability, reverse_probability))
            if shift > dl+dr:
                energy = dl+(shift-dl-dr)*np.sin(angle)**2
                other = shift-energy
                sign = (1 if n == 0 else -1)*parity
                measure = weights*np.pi/4*2*(energy*other+sign*dl*dr*np.cos(phase))/np.sqrt((energy+dl)*(other+dr))
                for sl, sr, weight in pair_spins:
                    channels.append((energy, other, measure*weight, sl, sr, True,
                                     n*drive, ell*OSCILLATOR_ENERGY, probability, reverse_probability))
    return dl, dr, channels


def transition(left, right, oscillator_temperature, channels, mu):
    heat, spin, jacobian = np.zeros(4), np.zeros(2), np.zeros((2, 2))
    entropy, minimum_rate = 0., np.inf
    for energy, other, measure, sl, sr, creation, work, recoil, probability, reverse_probability in channels:
        fl = expit((sl*mu[0]-energy)/left)
        fr = expit((sr*mu[1]-other)/right)
        dl, dr = fl*(1-fl)/left, fr*(1-fr)/right
        if creation:
            forward, reverse = probability*(1-fl)*(1-fr), reverse_probability*fl*fr
            injection = np.array([sl, sr])
            derivatives = np.array([-sl*dl*(probability*(1-fr)+reverse_probability*fr),
                                    -sr*dr*(probability*(1-fl)+reverse_probability*fl)])
            loss_l, loss_r = -energy, -other
            affinity = (energy-sl*mu[0])/left+(other-sr*mu[1])/right+recoil/oscillator_temperature
        else:
            forward, reverse = probability*fl*(1-fr), reverse_probability*fr*(1-fl)
            injection = np.array([-sl, sr])
            derivatives = np.array([sl*dl*(probability*(1-fr)+reverse_probability*fr),
                                    -sr*dr*(probability*fl+reverse_probability*(1-fl))])
            loss_l, loss_r = energy, -other
            affinity = -(energy-sl*mu[0])/left+(other-sr*mu[1])/right+recoil/oscillator_temperature
        occupation = forward-reverse
        net = measure@occupation
        heat += [measure@(occupation*loss_l), measure@(occupation*loss_r), work*net, recoil*net]
        spin += injection*net
        jacobian += np.outer(injection, derivatives@measure)
        entropy += measure@(occupation*affinity)
        minimum_rate = min(minimum_rate, float(np.min(measure)), float(np.min(occupation*affinity)))
    return heat, spin, jacobian, entropy, minimum_rate


def stationary(left, right, phase, amplitude, drive, rho, oscillator_temperature,
               conductance=45., nodes=192, count=24, normal=False, factorized=False,
               initial=(0., 0.)):
    ratio = r4.QUASIPARTICLE_DENSITY/(r4.NORMAL_POLARIZATION_TIME*r4.RATE_CONVERSION*conductance)
    bound, guess = 40., np.array(initial, dtype=float)
    for expansion in range(8):
        dl, dr, channels = prepare(left, right, phase, amplitude, drive, rho, oscillator_temperature,
                                   nodes, count, bound, normal, factorized)
        def residual(mu):
            spin = transition(left, right, oscillator_temperature, channels, mu)[1]
            losses = [r4.spin_loss(t, d, m)[0] for t,d,m in zip((left,right),(dl,dr),mu)]
            return spin-ratio*np.array(losses)
        def derivative(mu):
            jac = transition(left, right, oscillator_temperature, channels, mu)[2]
            losses = [r4.spin_loss(t,d,m)[1] for t,d,m in zip((left,right),(dl,dr),mu)]
            return jac-ratio*np.diag(losses)
        answer = least_squares(residual, guess, jac=derivative, bounds=(-bound,bound),
                               x_scale='jac',ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=200)
        mu = answer.x
        for _ in range(15):
            value=residual(mu)
            if max(abs(value)) < 1e-12: break
            correction=np.linalg.solve(derivative(mu),value)
            step=1.
            while step>1e-8:
                candidate=mu-step*correction
                if max(abs(candidate)) < bound and max(abs(residual(candidate))) < max(abs(value)):
                    mu=candidate
                    break
                step/=2
            if step<=1e-8: break
        if max(abs(residual(mu)))<1e-9 and max(abs(mu))<.8*bound: break
        bound*=2
        guess=mu
    else: raise RuntimeError('Spin balance failed')
    heat,spin,jac,entropy,minrate=transition(left,right,oscillator_temperature,channels,mu)
    losses=np.array([r4.spin_loss(t,d,m)[0] for t,d,m in zip((left,right),(dl,dr),mu)])
    scale=r4.RATE_CONVERSION*conductance*r4.KB
    return dict(mu=mu,heat=r4.HEAT_CONVERSION*conductance*heat,
                spin=r4.RATE_CONVERSION*conductance*spin,
                residual=residual(mu),jacobian=derivative(mu),
                entropy=scale*entropy,flip_entropy=scale*ratio*np.sum(mu*losses/np.array([left,right])),
                minimum_rate=minrate,search_bound=bound,expansions=expansion)


if __name__=='__main__':
    import json,time
    start=time.time()
    cases=[]
    for rho in (.1,.3,.7,1.,2.):
        for t_ph in (12.,24.):
            for amplitude,drive in ((0.,.3),(.7,.3),(.7,9.)):
                args=(.65,.9,0.,amplitude,drive,rho,t_ph)
                exact=stationary(*args)
                source=stationary(*args,factorized=True)
                cases.append(dict(rho=rho,oscillator_temperature=t_ph,amplitude=amplitude,drive=drive,
                                  exact_heat=exact['heat'].tolist(),source_heat=source['heat'].tolist(),
                                  exact_mu=exact['mu'].tolist(),source_mu=source['mu'].tolist(),
                                  relative_left_difference=float(abs(exact['heat'][0]-source['heat'][0])/abs(exact['heat'][0])),
                                  energy_residual=float(exact['heat'][0]+exact['heat'][1]+exact['heat'][2]-exact['heat'][3]),
                                  entropy=float(exact['entropy']+exact['flip_entropy']),minimum_rate=exact['minimum_rate']))
    print(json.dumps(dict(seconds=time.time()-start,cases=cases),indent=2))
