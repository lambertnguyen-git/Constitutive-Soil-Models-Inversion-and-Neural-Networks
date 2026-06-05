import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
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

def grad_hess_function(params, dobs, c=0, ws=1.0, wphys=1.0, soiltype = 'sand'):
# Taylor Series approx for 2nd derivative: [ f(x+h) - 2f(x) + f(x-h) ] / h^2
    
    grad = np.zeros(len(params))
    hess = np.zeros(len(params))
    sigma3 = dobs['sigma3']
    obs_eps1=dobs['epsa']

    q_now, _, eps_v_now, _  =   MC_forward(params, c, sigma3, eps_max=obs_eps1.max(), n_steps=1000, load_tag=110, obs_eps1=obs_eps1)
    dpred_now = {'q': q_now, 'epsv': eps_v_now}
    phi_now, *_ = objective_function(params, dobs, dpred_now, ws=ws, wphys=wphys, soiltype=soiltype)
    

    for i in range(len(params)):

        dx_frac = 0.01  # scales pertubation size of 1% of parameter value
        dx = dx_frac * abs(params[i]) if abs(params[i]) > 1e-10 else dx_frac
        params_up = params.copy()
        params_up[i] += dx
        params_down = params.copy()
        params_down[i] -= dx

        try:    
            q_up, _, eps_v_up, _ = MC_forward(params_up, c, sigma3, eps_max=obs_eps1.max(), n_steps=1000, load_tag=110, obs_eps1=obs_eps1)
            dpred_up = {'q': q_up, 'epsv': eps_v_up}
            phi_up, *_ = objective_function(params_up, dobs, dpred_up, ws=ws, wphys=wphys, soiltype=soiltype)

            q_down, _, eps_v_down, _ = MC_forward(params_down, c, sigma3, eps_max=obs_eps1.max(), n_steps=1000, load_tag=110, obs_eps1=obs_eps1)
            dpred_down = {'q': q_down, 'epsv': eps_v_down}
            phi_down, *_ = objective_function(params_down, dobs, dpred_down, ws=ws, wphys=wphys, soiltype=soiltype)
            
            grad[i] = (phi_up - phi_down) / (2 * dx)
            hess[i] = (phi_up - 2*phi_now + phi_down) / (dx**2)
        
        except Exception:
            hess[i] = 0.0
            grad[i] = 0.0
    
    return grad, hess

def newton_inversion(params_init, dobs, c=0, ws=1, wphys=1, soiltype='sand', max_iter=100, tol_phi=1e-4, tol_hess=1e-4):

    params_now  = np.array(params_init, dtype=float)
    sigma3_obs  = dobs['sigma3']
    epsa_obs    = dobs['epsa']

    q_now, _, eps_v_now, _ = MC_forward(params_now, c, sigma3_obs, eps_max=epsa_obs.max(), n_steps=1000, load_tag=110, obs_eps1=epsa_obs)
    
    dpred_now = {'q': q_now, 'epsv': eps_v_now}
    phi_now, *_        = objective_function(params_now, dobs, dpred_now, ws, wphys, soiltype)
    grad_now, hess_now = grad_hess_function(params_now, dobs, c, ws, wphys, soiltype)

    phi_diff = 9999
    phi_history = [phi_now]
    i = 0

    while (phi_diff > tol_phi) and (i < max_iter):
        i += 1
        n_half = 0
        damping = 1.0 #reset at each iteration - damping used to halve the newton step

        step = np.array([-grad_now[j] / hess_now[j] if abs(hess_now[j]) > tol_hess else 0
                         for j in range(len(params_now))])

        while n_half < 20:
            params_update    = params_now + damping * step
            params_update[0] = max(params_update[0], 1000)
            params_update[1] = max(params_update[1], 500)
            params_update[2] = max(params_update[2], 20)
            params_update[3] = max(params_update[3], 0)

            try: 
                q_update, _, eps_v_update, _ = MC_forward(params_update, c, sigma3_obs, eps_max=epsa_obs.max(), n_steps=1000, load_tag=110, obs_eps1=epsa_obs)
                dpred_update = {'q': q_update, 'epsv': eps_v_update}
                phi_update, *_ = objective_function(params_update, dobs, dpred_update, ws, wphys, soiltype)
            except Exception:
                damping *= 0.5
                n_half += 1
                continue      

            if phi_update < phi_now:
                phi_diff   = abs(phi_update - phi_now)
                params_now = params_update
                phi_now    = phi_update
                grad_now, hess_now = grad_hess_function(params_now, dobs, c, ws, wphys, soiltype)
                phi_history.append(phi_update)
                break
            damping *= 0.5
            n_half += 1

        if n_half == 20:
            print(f"Line search failed at iteration {i} — stopping")
            break

    q_final, _, eps_v_final, p_final = MC_forward(params_now, c, sigma3_obs, eps_max=epsa_obs.max(), n_steps=1000, load_tag=110, obs_eps1=epsa_obs)
    dpred_final = {'q': q_final, 'epsv': eps_v_final, 'p': p_final}
    phi_final, phi_q, phi_epsv, phi_s, phi_phys = objective_function(params_now, dobs, dpred_final, ws, wphys, soiltype)

    K_opt, G_opt, phi_opt, psi_opt = params_now
    print(f"Converged in {i} iterations")
    print(f"K   = {K_opt/1000:.1f} MPa")
    print(f"G   = {G_opt/1000:.1f} MPa")
    print(f"phi = {phi_opt:.2f}°")
    print(f"psi = {psi_opt:.2f}°")
    print(f"phi_total = {phi_final:.4f}")

    return params_now, phi_history, dpred_final