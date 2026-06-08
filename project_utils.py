import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker

""" 
Some functions were adapted from a Constitutive Model MATLAB author by Prof. Mahdi Taiebat, Drs. Ming Yang & Andres Reyes, and Mr. Sheng Zeng.
"""


def stress_invariants(sigm):
    """
    Returns first invariant J1, second invariant J2D, and Lode angle theta (MC) - used in YS
    """
    J1 = sigm[0] + sigm[1] + sigm[2]

    p = J1 / 3.0
    I = np.array([1., 1., 1., 0., 0., 0.])
    s = sigm - p * I     #shear/deviatoric

    J2D = (0.5 * (s[0]**2 + s[1]**2 + s[2]**2)
           + s[3]**2 + s[4]**2 + s[5]**2)

    # Lode angle from J3D
    if J2D < 1e-12:
        theta = 0.0
    else:
        J3D = (s[0]*s[1]*s[2]
               - s[0]*s[5]**2
               - s[1]*s[4]**2
               - s[2]*s[3]**2
               + 2*s[3]*s[4]*s[5])
        sin3theta = np.clip(-3*np.sqrt(3)/2 * J3D / (J2D**1.5), -1., 1.)
        theta     = 1./3. * np.arcsin(sin3theta)

    return J1, J2D, theta

def C_elastic(K, G):
    """
    6x6 isotropic elastic stiffness matrix
    K, G (kPa)
    """
    t1 = K + 4./3.*G
    t2 = K - 2./3.*G
    t3 = 2.*G

    Cel = np.array([
        [t1, t2, t2,  0,  0,  0],
        [t2, t1, t2,  0,  0,  0],
        [t2, t2, t1,  0,  0,  0],
        [ 0,  0,  0, t3,  0,  0],
        [ 0,  0,  0,  0, t3,  0],
        [ 0,  0,  0,  0,  0, t3]], dtype=float)

    return Cel

def yield_surface_MC(sigm, phi, c):
    """
    Mohr-Coulomb yield surface
    phi : friction angle (radians)
    c   : cohesion (kPa)

    Returns:
        f < 0 : elastic (inside YS)
        f = 0 : on YS
        f > 0 : not possible (outside YS)
    """
    J1, J2D, theta = stress_invariants(sigm)

    f = (-1./3. * J1 * np.sin(phi)
         + np.sqrt(J2D) * (np.cos(theta)
         + 1./np.sqrt(3.) * np.sin(theta) * np.sin(phi))
         - c * np.cos(phi))

    return f

def crosspoint(sigm, dsigm, phi, c):
    """
    Find crosspoint - beta is portion of step to reach YS
    brentq (equivalent to MATLAB fzero) 
    """
    def f(beta):
        stress_at_beta = sigm + beta * dsigm
        return yield_surface_MC(stress_at_beta, phi, c)

    f0 = f(0.0)
    f1 = f(1.0)

    if f0 >= 0.0:
        return 0.0      # on or outside YS
    if f1 <= 0.0:
        return 1.0      # inside YS

    beta = brentq(f, 0.0, 1.0, xtol=1e-10)
    return beta

def fnormal_MC(sigm, phi, c):
    """
    Gradient of YS wrt stress
    Returns 6-component gradient vector (dfds)
    """
    J1, J2D, theta = stress_invariants(sigm)

    p = J1 / 3.0
    I = np.array([1., 1., 1., 0., 0., 0.])
    s = sigm - p * I

    a1 = I.copy()

    sqrt_J2D = max(np.sqrt(J2D), 1e-12)
    a2       = s / (2. * sqrt_J2D)
    a2[3:]  *= 2.

    a3 = np.array([
        s[1]*s[2] - s[5]**2 + J2D/3.,
        s[0]*s[2] - s[4]**2 + J2D/3.,
        s[0]*s[1] - s[3]**2 + J2D/3.,
        2.*(s[4]*s[5] - s[2]*s[3]),
        2.*(s[3]*s[5] - s[1]*s[4]),
        2.*(s[3]*s[4] - s[0]*s[5])])

    C1 = -1./3. * np.sin(phi)

    if abs(theta + np.pi/6.) < 1e-6:
        C2 = 0.5 * (np.sqrt(3.) - np.sin(phi)/np.sqrt(3.))
        C3 = 0.0
    elif abs(theta - np.pi/6.) < 1e-6:
        C2 = 0.5 * (np.sqrt(3.) + np.sin(phi)/np.sqrt(3.))
        C3 = 0.0
    else:
        C2 = (np.cos(theta) * (1.
              + np.tan(theta)*np.tan(3.*theta)
              + np.sin(phi)*(-np.tan(3.*theta)
              + np.tan(theta))/np.sqrt(3.)))
        C3 = ((np.sqrt(3.)*np.sin(theta)
               - np.cos(theta)*np.sin(phi))
              / (2.*J2D*np.cos(3.*theta)))

    dfds = C1*a1 + C2*a2 + C3*a3
    return dfds

def gnormal_MC(sigm, psi):
    """
    Gradient of plastic potential wrt stress
    """
    return fnormal_MC(sigm, psi, c=0.0)

def f_constraint(load_tag, dX):
    S = np.zeros((6, 6))
    E = np.zeros((6, 6))
    dY = np.zeros(6)
    dY = np.array([0,  0,  0,  0,  0,  dX])

    if load_tag == 110:
        # drained triaxial
        S = np.array([
            [1,  0,  0,  0,  0,  0],
            [0,  1,  0,  0,  0,  0],
            [0,  0,  0,  1,  0,  0],
            [0,  0,  0,  0,  1,  0],
            [0,  0,  0,  0,  0,  1],
            [0,  0,  0,  0,  0,  0]], dtype=float)
        E = np.array([
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  1,  0,  0,  0]], dtype=float)

    elif load_tag == 100:
        # undrained triaxial
        S = np.array([
            [1,  -1,  0,  0,  0,  0],
            [0,  0,  0,  1,  0,  0],
            [0,  0,  0,  0,  1,  0],
            [0,  0,  0,  0,  0,  1],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0]], dtype=float)
        E = np.array([
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [1,  1,  1,  0,  0,  0],
            [0,  0,  1,  0,  0,  0]], dtype=float)

    elif load_tag == 10:
        # isotropic consolidation
        S = np.array([
            [1,  -1,  0,  0,  0,  0],
            [0,  1,  -1,  0,  0,  0],
            [0,  0,  0,  1,  0,  0],
            [0,  0,  0,  0,  1,  0],
            [0,  0,  0,  0,  0,  1],
            [0,  0,  0,  0,  0,  0]], dtype=float)
        E = np.array([
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0],
            [0,  0,  1,  0,  0,  0]], dtype=float)
        
    return S, E, dY

def Bardet_triaxial(sigm, C, deps_axial, inc_frac, load_tag):
    """
    Return the trial stress and strain increment

    Returns:
        deps  : 6-component strain increment
        dsigm : 6-component stress increment
    """
    dX = inc_frac * deps_axial

    S, E, dY = f_constraint(load_tag, dX)

    A = S @ C + E
    deps = np.linalg.solve(A, dY)
    dsigm = C @ deps

    return deps, dsigm

def MC_forward(params, c, sigma3, eps_max=0.25, n_steps=1000, load_tag=110, obs_eps1=None):

    K, G, phi_deg, psi_deg = params

    phi = np.radians(phi_deg)
    psi = np.radians(psi_deg)

    Cel = C_elastic(K, G)

    deps_axial = eps_max / n_steps

    # initial stress state
    sigm = np.array([sigma3, sigma3, sigma3, 0, 0, 0]) #confining pressure
    eps  = np.zeros(6)  #

    q_out     = np.zeros(n_steps)
    eps_q_out = np.zeros(n_steps)
    eps_v_out = np.zeros(n_steps)
    p_out     = np.zeros(n_steps)

    for i in range(n_steps):

        # elastic trial step
        deps_trial, dsigm_trial = Bardet_triaxial(sigm, Cel, deps_axial, inc_frac=1.0, load_tag=load_tag)

        sigm_trial = sigm + dsigm_trial
        f_pred     = yield_surface_MC(sigm_trial, phi, c)

        if f_pred <= 0:
            # elastic (inside YS)
            sigm = sigm_trial
            eps  = eps + deps_trial

        else:
            # elastoplastic (on YS)

            # check if already on yield surface
            f_start = yield_surface_MC(sigm, phi, c)

            if f_start >= 0.:
                beta = 0.0          # already on surface
            else:
                beta = crosspoint(sigm, dsigm_trial, phi, c)

            # apply elastic portion
            sigm = sigm + beta * dsigm_trial
            eps  = eps  + beta * deps_trial

            # gradients at yield point
            dfds = fnormal_MC(sigm, phi, c)
            dgds = gnormal_MC(sigm, psi)

            # plastic modulus — perfectly plastic for MC
            Kp   = 1e-9     # small not zero

            # elastoplastic stiffness Cepl
            temp1 = Cel @ dgds
            temp2 = Cel @ dfds
            denom = Kp + dfds @ temp1
            Cepl  = Cel - np.outer(temp1, temp2) / denom

            # plastic portion
            inc_frac           = 1.0 - beta
            deps, dsigm_pl     = Bardet_triaxial(sigm, Cepl, deps_axial, inc_frac, load_tag=load_tag)

            sigm = sigm + dsigm_pl
            eps  = eps  + deps

        q     = sigm[2] - sigm[0]                  
        p     = (sigm[0] + sigm[1] + sigm[2]) / 3.
        eps_q = 2./3. * (eps[2] - eps[0])            # deviatoric strain
        eps_v = eps[0] + eps[1] + eps[2]             # volumetric strain

        q_out[i]     = q
        eps_q_out[i] = eps_q
        eps_v_out[i] = eps_v
        p_out[i]     = p

    if obs_eps1 is not None:
        eps1_pred = np.linspace(0., eps_max, n_steps)
        q_out     = interp1d(eps1_pred, q_out, bounds_error=False, fill_value='extrapolate')(obs_eps1)
        eps_v_out = interp1d(eps1_pred, eps_v_out, bounds_error=False, fill_value='extrapolate')(obs_eps1)
        eps_q_out = interp1d(eps1_pred, eps_q_out, bounds_error=False, fill_value='extrapolate')(obs_eps1)
        p_out     = interp1d(eps1_pred, p_out, bounds_error=False, fill_value='extrapolate')(obs_eps1)

    return q_out, eps_q_out, eps_v_out, p_out

def plot_MC(q, eps_q, eps_v, p, label='MC', color='b'):
    """
    Reproduce the four plots from the MATLAB f_plot function
    for drained triaxial (load_tag = 110)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # q vs eps_q
    axes[0,0].plot(eps_q*100, q, c="b", lw=2, label=label)
    axes[0,0].set_xlabel('ε_q (%)')
    axes[0,0].set_ylabel('q = σ_a - σ_r (kPa)')
    axes[0,0].set_title('Deviatoric Stress vs Deviatoric Strain')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # q vs p
    axes[0,1].plot(p, q, c="g", lw=2, label=label)
    axes[0,1].set_xlabel("p' (kPa)")
    axes[0,1].set_ylabel('q (kPa)')
    axes[0,1].set_title("Stress Path (q vs p')")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # eps_v vs eps_q
    axes[1,0].plot(eps_q*100, eps_v*100, c="k", lw=2, label=label)
    axes[1,0].set_xlabel('ε_q (%)')
    axes[1,0].set_ylabel('ε_v (%)')
    axes[1,0].set_title('Volumetric vs Deviatoric Strain')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # eps_v vs p
    axes[1,1].plot(p, eps_v*100, c="r", lw=2, label=label)
    axes[1,1].set_xlabel("p' (kPa)")
    axes[1,1].set_ylabel('ε_v (%)')
    axes[1,1].set_title("Volumetric Strain vs p'")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def loader_drained(filename, data_dir=None):
    #for TMD format
    
    filepath    = os.path.join(data_dir, filename) if data_dir else filename
    data        = np.loadtxt(filepath, skiprows=3)
   
    epsa    = data[:, 0] / 100
    epsv    = data[:, 1] / 100
    e       = data[:, 4]
    q       = data[:, 5] #kPa
    p       = data[:, 6] #kPa

    sigma3  = p[0] - q[0] / 3
    epsr    = (epsv - epsa) / 2
    epsq    = 2/3 * (epsa - epsr)
    
    return {
        'epsa'  : epsa,
        'epsv'  : epsv,
        'epsq' : epsq,
        'e'     : e,
        'p'     : p,
        'q'     : q,
        'sigma3': sigma3}

def loader_undrained(filename, data_dir=None):
    #for TMU format
    
    filepath    = os.path.join(data_dir, filename) if data_dir else filename
    data        = np.loadtxt(filepath, skiprows=3)
   
    epsa    = data[:, 0] / 100
    u       = data[:, 1] #kPa
    p       = data[:, 6] #kPa
    q       = data[:, 7] #kPa
    epsv   = np.zeros(len(epsa))  # undrained
    e      = np.full(len(q), np.nan)

    sigma3      = p[0] - q[0] / 3
    epsr   = (epsv - epsa) / 2
    epsq   = 2/3 * (epsa - epsr)
    
    return {
        'epsa'  : epsa,
        'epsv'  : epsv,
        'epsq' : epsq,
        'e'     : e,
        'u'     : u,        
        'p'     : p,
        'q'     : q,
        'sigma3': sigma3}

def plot_convergence(test, phi_history, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phi_history, 'b-o', markersize=4)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Objective φ')
    ax.set_title(f'{test} — Convergence')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{test}_iterations.png'), dpi=150, bbox_inches='tight')
    plt.close()