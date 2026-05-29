# These are copied directly from main code and have not been adapted to be more general yet.

def objective_function(beta, dpred, dobs, vs):
    phi_d = np.sum((dpred - dobs)**2)
    phi_m = np.sum(np.diff(vs)**2)
    phi = phi_d + beta * phi_m

    return phi, phi_d, phi_m

def step_function(grad_vs1, grad_vs2, step_size, vs):
    #for 2 shear wave velocity system
    vs_new = [
        max(vs_first[0] - step * grad_vs1, 0.05),
        max(vs_first[1] - step * grad_vs2, 0.05),
        vs_first[2]
    ]
    vp_new = [2*v for v in vs_new]
    return vs_new, vp_new

def plot_tikhonov_curves(beta_values, phid, phim, n_kernels=n_kernels, beta_index=0, ax=None): 
    if ax is None: 
        fig, ax = plt.subplots(1, 3, figsize=(12, 4)) 
    
    ax[0].semilogx(beta_values, phid)
    ax[1].semilogx(beta_values, phim)
    ax[2].loglog(phim, phid) 

    # plot beta at the beta index
    ax[0].semilogx(beta_values[beta_index], phid[beta_index], "ro")
    ax[1].semilogx(beta_values[beta_index], phim[beta_index], "ro")
    ax[2].loglog(phim[beta_index], phid[beta_index], "ro")
    
    ax[0].set_ylabel("$\\phi_d$")
    ax[1].set_ylabel("$\\phi_m$")
    
    ax[2].set_xlabel("$\\phi_m$")
    ax[2].set_ylabel("$\\phi_d$")
    
    # add our target misfit 
    ax[0].semilogx(beta_values, np.ones(len(beta_values))*n_kernels, "--k")
    ax[2].loglog(phim, np.ones(len(phim))*n_kernels, "--k")
    
    for a in ax[:2]:
        a.invert_xaxis()
        a.set_xlabel("$\\beta$")


def plot_inversion_results(x_cells, G, model, mrecs, dobs, dpred, beta_index=0, ax=None):
    if ax is None: 
        fig, ax = plt.subplots(1, 3, figsize=(12, 3)) 
    
    ax[0].plot(x_cells, G.T)
    
    # plot our observed data 
    ax[1].plot(dobs, "o", label="observed data")
    ax[1].plot(dpreds[:, beta_index], "-s", label="predicted data")
    
    ax[2].plot(x_cells, model, label="true model")
    ax[2].plot(x_cells, mrecs[:, beta_index], label="recovered model")
    
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("kernel value")
    
    ax[1].set_xlabel("data index")
    ax[1].set_ylabel("data value")
    
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("model value")
    
    ax[1].legend()
    ax[2].legend()
    
    plt.tight_layout()