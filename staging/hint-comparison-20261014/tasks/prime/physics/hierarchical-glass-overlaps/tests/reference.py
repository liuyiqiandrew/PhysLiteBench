"""Independent marked-Poisson Laplace integrals for quenched coincidences.

The Gaussian-tree limit is theorem-backed. These positive integrals check its
normalized cascade probabilities; they do not simulate a finite Gaussian tree.
No oracle participation formula is called.
"""
import math
from functools import lru_cache
from scipy.integrate import quad


@lru_cache(maxsize=None)
def gamma_integral(shape, tolerance=2e-11):
    if shape <= 0:
        raise ValueError('Positive gamma-integral shape required.')
    left, error_left = quad(lambda x: math.exp(-x), 0., 1.,
                            weight='alg', wvar=(shape-1.,0.),
                            epsabs=tolerance, epsrel=tolerance)
    right, error_right = quad(lambda x: math.exp((shape-1)*math.log(x)-x),
                              1., math.inf, epsabs=tolerance, epsrel=tolerance)
    return left+right, error_left+error_right


@lru_cache(maxsize=None)
def poisson_exponent(m, tolerance=2e-11):
    # m int v^(-1-m)(1-exp(-v))dv, split at1 and invert the upper tail.
    def left_kernel(v):
        return -math.expm1(-v)/v if v else 1.
    def right_kernel(w):
        return -math.expm1(-1/w) if w else 1.
    left, e1 = quad(left_kernel,0.,1.,weight='alg',wvar=(-m,0.),
                    epsabs=tolerance,epsrel=tolerance)
    right, e2 = quad(right_kernel,0.,1.,weight='alg',wvar=(m-1.,0.),
                     epsabs=tolerance,epsrel=tolerance)
    return m*(left+right), m*(e1+e2)


def parent_moment(m, order, tolerance=2e-11):
    B, eB = poisson_exponent(m,tolerance)
    numerator, en = gamma_integral(order-m,tolerance)
    denominator, ed = gamma_integral(float(order),tolerance)
    normalizer, ez = quad(lambda z: math.exp(-z),0.,math.inf,
                          epsabs=tolerance,epsrel=tolerance)
    return numerator*normalizer/(denominator*B), {
        'm':m,'B':B,'gamma_numerator':numerator,'gamma_denominator':denominator,
        'normalizer_integral':normalizer,'quadrature_errors':[eB,en,ed,ez]}


def marked_leaf_partition(q, alpha, blocks, tolerance=2e-11):
    order = sum(blocks)
    k = len(blocks)
    if not 0 < q < alpha < 1:
        raise ValueError('The marked stable moment requires 0<q<alpha<1.')
    parent, parent_evidence = parent_moment(q,order,tolerance)
    B, eB = poisson_exponent(alpha,tolerance)
    # Scale tau=B^(1/alpha)*t. Integrate the fractional-moment kernel
    # directly, retaining the S^q mark in both numerator and denominator.
    def left_kernel(tau):
        power = tau**alpha
        return -math.expm1(-power)/power if power else 1.
    def tail_kernel(w):
        return -math.expm1(-w**(-alpha)) if w else 1.
    jl, ejl = quad(left_kernel,0.,1.,weight='alg',wvar=(alpha-q-1.,0.),
                   epsabs=tolerance,epsrel=tolerance,limit=300)
    jr, ejr = quad(tail_kernel,0.,1.,weight='alg',wvar=(q-1.,0.),
                   epsabs=tolerance,epsrel=tolerance,limit=300)
    gamma_one_q, eg1 = gamma_integral(1-q,tolerance)
    M = q/gamma_one_q * B**(q/alpha) * (jl+jr)
    # For the Campbell numerator, z=tau^alpha leaves a positive gamma
    # integral. This variable change avoids a slowly decaying tau tail.
    Hnumerator, eH = gamma_integral(k-q/alpha,tolerance)
    H = Hnumerator/alpha
    terms = [gamma_integral(size-alpha,tolerance) for size in blocks]
    gn = math.prod(z[0] for z in terms)
    egn = sum(z[1] for z in terms)
    gd, egd = gamma_integral(order-q,tolerance)
    A = alpha**k*gn/gd * B**(q/alpha-k) * H
    return parent*A/M, {
        'q':q,'alpha':alpha,'order':order,'blocks':list(blocks),'parent':parent,
        'parent_evidence':parent_evidence,'B':B,'fractional_integral':[jl,jr],
        'M':M,'A':A,'H':H,'marked_ratio':A/M,
        'quadrature_errors':[eB,ejl,ejr,eg1,eH,egn,egd]}


def prediction(row, width, tolerance=2e-11):
    beta=float(row['beta']); order=int(row['replicas'])
    b1=math.sqrt(.6/.7)/width; b2=math.sqrt(1.4/.3)/width
    if beta <= b1:
        return 0., {'phase':'unfrozen'}
    q=b1/beta
    if row['observable']=='parent_collision':
        value,evidence=parent_moment(q,order,tolerance)
        return value, {'phase':'parent_frozen','integrals':evidence}
    blocks = row['partition'] if row['observable']=='leaf_partition' else [order]
    if beta <= b2:
        if all(size==1 for size in blocks):
            value,evidence=parent_moment(q,order,tolerance)
            return value, {'phase':'descendants_unfrozen_singletons','integrals':evidence}
        return 0., {'phase':'descendants_unfrozen_repeated_leaf'}
    value,evidence=marked_leaf_partition(q,b2/beta,blocks,tolerance)
    return value, {'phase':'both_frozen','integrals':evidence}


def predict(rows,width,tolerance=2e-11,return_evidence=False):
    values=[];evidence=[]
    for row in rows:
        value,detail=prediction(row,width,tolerance)
        values.append(value);evidence.append(detail)
    return (values,evidence) if return_evidence else values
