# Heating of a compressible electron fluid

A uniform planar slab occupies 0<z<thickness, with vacuum on both sides and infinite lateral extent. A neutral stationary ionic background confines a charged electron fluid. Use reduced units with epsilon0=mu0=c=1. In the small-amplitude regime the induced charge density rho, electric current J and electromagnetic fields obey

    partial_t rho + div J = 0
    partial_t J + gamma J = omega_p² E - beta² grad rho
    curl E = -partial_t H
    curl H = J + partial_t E
    div E = rho.

Outside the slab J=rho=0. The known collision rate is gamma=.06, and the known conservative pressure-wave speed is beta=.1. The unknown plasma frequency omega_p is common to all preparations, lies in [.85,1.15], and is stored as `plasma_frequency`. Its definition is omega_p²=n0*q²/m_e in these units, where n0, q and m_e are equilibrium carrier density, carrier charge and mass; J=q*n0*v to linear order. Only their stated plasma-frequency combination is needed. The term -gamma*v is local mechanical friction against the stationary ionic bath. The pressure force is conservative. There is no other damping or heat source.

This is the specified ideal linear continuum fluid, valid for all stated settings. No viscosity, density diffusion, spill-out, tunneling, bound-electron dielectric response, extra surface energy, or quantum correction is included. The background relative permittivity and magnetic permeability are one. The slab is mechanically fixed. Its hard reflecting walls impose J_z=0 at both surfaces, with no additional surface charge sheet or surface current. Tangential E and H are continuous at each vacuum interface.

A monochromatic p-polarized plane wave arrives from z<0, with E in the x-z plane and H along y. It has positive angular frequency `frequency` in [.7,1.5], signed incidence angle `angle` in [-1,1] radians, and dependence exp(i*frequency*sin(angle)*x - i*frequency*t). There is no wave incident from the other side. The known slab `thickness` lies in [.2,.8]. The fields reach their periodic steady state before measurement.

A nonperturbing depth-resolved calorimeter records irreversible energy delivered by electron friction to the ionic bath between z=window[0]*thickness and z=window[1]*thickness. Both window endpoints lie in [0,1], and their difference is at least .15. The readout is the mean heating power per unit surface area in that region, averaged over one optical period and divided by the incident normal electromagnetic energy flux. It is dimensionless. The measurement labels the bath location at which friction transfers energy; subsequent thermal conduction is not part of this readout. Background thermal heating is subtracted. Take the incident-amplitude limit to zero while retaining this power ratio, so the linear field equations determine the response and no temperature feedback changes the material.

Each experiment has `frequency`, `thickness`, `angle` and `window` (a two-element list). `data/calibration.json` contains normal-incidence heating measurements at several frequencies, thicknesses and depth windows. Records have `input`, measured `value`, and independent Gaussian uncertainty `sigma`. The fixed instrument uncertainty is .00001 in the dimensionless readout, independent of the unknown plasma frequency and of the response. All controls are exact.

Implement `Model.fit(records)`, returning self and storing `plasma_frequency`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Use the same fitted plasma frequency in all preparations.
