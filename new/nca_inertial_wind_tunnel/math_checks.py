"""Exact linear checks for an inertial cellular recurrence. No neural training.

Run: python math_checks.py --out results/linear_checks.json
The reported spectral e-fold/1% times are asymptotic modal estimates, NOT
measured task-solving times or finite-time norm convergence guarantees.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def roots(beta, q):
    """q is an eigenvalue of K = eta*J_R - D*lambda."""
    z = np.asarray(1.0 + beta + q, dtype=np.complex128)
    discriminant = np.lib.scimath.sqrt(z*z - 4.0*beta)
    return (z + discriminant)/2.0, (z - discriminant)/2.0


def radius(beta, q):
    u, v = roots(beta, q)
    return np.maximum(np.abs(u), np.abs(v))


def block_matrix(beta, K):
    K = np.atleast_2d(K)
    n = K.shape[0]
    I = np.eye(n)
    B = float(beta)*I
    return np.block([[I+K, B], [K, B]])


def run():
    rng = np.random.default_rng(20260930)
    beta = rng.uniform(0.0, 0.999, 20000)
    a = rng.uniform(-0.2, 4.2, 20000)
    expected = (a > 0) & (a < 2*(1+beta))
    observed = radius(beta, -a) < 1
    real_mismatch = int(np.count_nonzero(expected != observed))
    assert real_mismatch == 0

    q = rng.uniform(-4.2, 0.2, 20000) + 1j*rng.uniform(-1.2, 1.2, 20000)
    expr = ((1+beta+q.real)/(1+beta))**2 + (q.imag/(1-beta))**2
    complex_mismatch = int(np.count_nonzero((expr < 1) != (radius(beta, q) < 1)))
    assert complex_mismatch == 0

    matrix_errors = []
    for _ in range(50):
        b = float(rng.uniform(.01, .98))
        K = rng.normal(size=(4,4))*.12 - np.eye(4)*.2
        full_rho = max(abs(np.linalg.eigvals(block_matrix(b, K))))
        predicted_rho = max(radius(b, np.linalg.eigvals(K)))
        matrix_errors.append(abs(full_rho-predicted_rho))
        assert abs(np.linalg.det(block_matrix(b, K))-b**4) < 1e-11
    assert max(matrix_errors) < 1e-11

    # Locality and modal formula, 1D periodic ring, unnormalized Laplacian.
    n, t, d, b = 129, 24, .2, .9
    h = np.zeros(n); h[n//2] = 1.0
    v = np.zeros(n)
    h0 = h.copy()
    k = 2*np.pi*np.arange(n)/n
    lam = 2-2*np.cos(k)
    hf = np.fft.fft(h); vf = np.zeros(n, dtype=np.complex128)
    for _ in range(t):
        v = b*v-d*(2*h-np.roll(h,1)-np.roll(h,-1))
        h = h+v
        vf = b*vf-d*lam*hf
        hf = hf+vf
    fourier_error = float(np.max(np.abs(h-np.fft.ifft(hf).real)))
    dist = np.abs(np.arange(n)-n//2)
    outside_cone = float(np.max(np.abs(h[dist>t])))
    assert fourier_error < 1e-12 and outside_cone == 0
    # An unstructured first-order local CA can already carry a signal ballistically.
    shift = h0.copy()
    for _ in range(t):
        shift = shift + (np.roll(shift,1)-shift)
    assert shift[(n//2+t)%n] == 1

    scaling = []
    for n in (32,64,128,256,512):
        mu = 4*np.sin(np.pi/n)**2  # smallest NONZERO eigenvalue of periodic ring
        row = {'grid_size':n, 'lambda_min_nonzero':float(mu)}
        for name,b,d in [('diffusive_fixed',0.,.2),('inertial_fixed',.9,.2)]:
            r = float(radius(b,-d*mu))
            row[name] = {'beta':b,'d':d,'slow_mode_radius':r,
                         'asymptotic_1pct_steps':float(np.log(.01)/np.log(r))}
        # Spectrum-tuned classical quadratic heavy-ball reference: not a learned architecture.
        b = ((2-np.sqrt(mu))/(2+np.sqrt(mu)))**2
        d = 4/(2+np.sqrt(mu))**2
        row['spectrum_tuned_reference'] = {'beta':float(b),'d':float(d),
            'slow_mode_radius':float(np.sqrt(b)),
            'asymptotic_1pct_steps':float(np.log(.01)/np.log(np.sqrt(b)))}
        scaling.append(row)

    q = -.2+.1j
    complex_counterexample = {
        'q_real':q.real,'q_imag':q.imag,
        'rho_beta_0':float(radius(0.,q)), 'rho_beta_09':float(radius(.9,q))}
    assert complex_counterexample['rho_beta_0'] < 1 < complex_counterexample['rho_beta_09']

    A = block_matrix(.9,np.array([[-.1]]))
    B = block_matrix(.9,np.array([[-3.7]]))
    switching = {'a_sequence':[.1,3.7], 'beta':.9,
                 'rho_A':float(max(abs(np.linalg.eigvals(A)))),
                 'rho_B':float(max(abs(np.linalg.eigvals(B)))),
                 'rho_two_step_product':float(max(abs(np.linalg.eigvals(B@A))))}
    assert switching['rho_A'] < 1 and switching['rho_B'] < 1
    assert switching['rho_two_step_product'] > 1

    # Constructed Turing example, NOT a trained NCA and NOT a regeneration experiment.
    J = np.array([[1.,-2.],[2.,-3.]])
    D = np.diag([.1,1.])
    eta,beta = .01,.8
    values = np.linspace(0.,8.,801)
    rates = []
    flat_rates = []
    for value in values:
        rates.append(float(max(abs(np.linalg.eigvals(block_matrix(beta,eta*(J-value*D)))))))
        flat_rates.append(float(max(abs(np.linalg.eigvals(
            block_matrix(beta,eta*(J-value*np.eye(2)*np.trace(D)/2)))))))
    idx = int(np.argmax(rates))
    turing = {'reaction_J':J.tolist(),'continuous_D':[.1,1.], 'eta':eta,'beta':beta,
              'unstable_lambda_interval_continuous':[2.,5.],
              'homogeneous_radius':rates[0], 'max_radius':rates[idx],
              'lambda_at_sampled_max':float(values[idx]),
              'max_radius_equal_diffusion':max(flat_rates),
              'note':'Analytically chosen linear system. No nonlinear saturation or recovery tested.'}
    assert rates[0] < 1 and max(rates) > 1 and max(flat_rates) < 1

    # Zero spatial mode is neutral (not a strictly stable mode).
    zero_mode = {'roots_beta_09':[1.,.9],
                 'h_infinity_from_h0_0_v0_1':.9/(1-.9)}

    return {
        'status':'COMPLETED_LINEAR_CHECKS_ONLY',
        'seed':20260930,
        'real_stability_tests':{'n':20000,'mismatches':real_mismatch},
        'complex_stability_tests':{'n':20000,'mismatches':complex_mismatch},
        'full_matrix_spectral_mapping_max_error':float(max(matrix_errors)),
        'real_space_vs_fourier_max_error':fourier_error,
        'outside_local_causality_cone_max':outside_cone,
        'first_order_shift_baseline_is_ballistic':True,
        'zero_mode':zero_mode,
        'complex_reaction_counterexample':complex_counterexample,
        'switching_counterexample':switching,
        'scaling':scaling,
        'constructed_turing_example':turing,
        'not_established':['better training','better task performance','regeneration',
                           'GPU speedup','a new architecture contribution'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('results/linear_checks.json'))
    args = parser.parse_args()
    report = run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
