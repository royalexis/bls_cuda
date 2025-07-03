import numpy as np
import utils_python.transitmodel as transitm

def mcmcTransitModel(sol, params_to_fit):
    """
    Returns the new log functions and the sol array to use for mcmc

    sol: Transit Model object containing the initial parameters
    params_to_fit: List containing strings of the names of the parameters to fit according to the tm class

    return: New log likelihood function, New log Prior function, Array of initial parameters to pass to mcmc
    """
    id_to_fit = [transitm.var_to_ind[param] for param in params_to_fit]
    sol_a = sol.to_array()
    
    def newLoglikehood(modelFunc, fit_sol, time, flux, ferror, itime):
        for i, ind in enumerate(id_to_fit):
            sol_a[ind] = fit_sol[i]
        return loglikehood(modelFunc, sol_a, time, flux, ferror, itime)
    
    def newLogPrior(fit_sol, time):
        for i, ind in enumerate(id_to_fit):
            sol_a[ind] = fit_sol[i]
        return logprior(sol_a, time)
    
    return newLoglikehood, newLogPrior, sol_a[id_to_fit]


def loglikehood(modelFunc, sol, time, flux, ferror, itime):

    model = modelFunc(sol, time, itime)

    n = len(flux)

    if n < 1:
        ll = -1e30
    else:
        ll = -0.5*(n*np.log(2*np.pi) + np.sum(np.log(ferror*ferror) + ((flux - model)/ferror)**2))

    return ll

def logprior(sol, time):
    badprior = -1e30
    lprior = 0

    min_t = min(time)
    max_t = max(time)

    rho = sol[0]
    nl1, nl2, nl3, nl4 = sol[1], sol[2], sol[3], sol[4]
    dil, vof, zpt = sol[5], sol[6], sol[7]

    t0, per, bb, rdr = sol[8], sol[9], sol[10], sol[11]
    ecw, esw, krv, ted, ell, alb = sol[12], sol[13], sol[14], sol[15], sol[16], sol[17]

    cond1 = 1e-4 < rho < 1e3 and 0 <= nl1 <= 2 and -1 <= nl2 <= 1 and 0 <= nl3 <= 1 and 0 <= nl4 <= 1 and 0 <= dil <= 1 and 0 <= vof < 1 and -5 < zpt < 5
    cond2 = min_t < t0 < max_t and min_t < per < max_t and 0 <= bb < 2 and 0 <= rdr <= 1 and -1 <= ecw <= 1 and -1 <= esw <= 1
    # Unsure about what priors I should put for the last 4 params
    cond3 = -5 < krv < 5 and 0 <= ted < 1e3 and -1e3 < ell < 1e3 and 0 <= alb < 1e4

    if not (cond1 and cond2 and cond3):
        lprior = badprior
    
    return lprior
