import numpy as np
import utils_python.transitmodel as transitm

def genmcmcInput(sol, params_to_fit):
    """
    Returns the new log function, the sol array and the beta array to use for mcmc

    sol: Transit Model object containing the initial parameters
    params_to_fit: List containing strings of the names of the parameters to fit according to the tm class

    return: New log prob function, Array of initial parameters to pass to mcmc, Initial guess for beta using errors
    """
    id_to_fit = [transitm.var_to_ind[param] for param in params_to_fit]
    sol_a = sol.to_array()

    def newLogprob(fit_sol, time, flux, ferror, itime):
        for i, ind in enumerate(id_to_fit):
            sol_a[ind] = fit_sol[i]
        return logprob(sol_a, time, flux, ferror, itime)
    
    beta = sol.err_to_array()[id_to_fit]
    
    return newLogprob, sol_a[id_to_fit], beta

def getParams(chain, burnin, sol, params_to_fit):
    """
    Generates a transit model object will all the parameters
    """
    # Get params from mcmc
    npars = len(chain[1,:])
    mm = np.zeros(npars)
    std = np.zeros(npars)
    for i in range(npars):
        mm[i] = np.mean(chain[burnin:,i])
        std[i] = np.std(chain[burnin:,i])

    # Return to the full array
    id_to_fit = [transitm.var_to_ind[param] for param in params_to_fit]
    sol_full = sol.to_array()
    err_full = np.zeros(len(sol_full))

    for i, ind in enumerate(id_to_fit):
        sol_full[ind] = mm[i]
        err_full[ind] = std[i]

    # Generate the object
    sol_output = transitm.transit_model_class()
    sol_output.from_array(sol_full)
    sol_output.load_errors(err_full)

    return sol_output

def logprob(sol, time, flux, ferror, itime):
    return loglikehood(transitm.transitModel, sol, time, flux, ferror, itime) + logprior(sol, time)

def loglikehood(modelFunc, sol, time, flux, ferror, itime):

    model = modelFunc(sol, time, itime)

    n = len(flux)

    if n < 1:
        ll = -1e30
    else:
        ll = -0.5*(n*np.log(2*np.pi) + np.sum(np.log(ferror*ferror) + ((flux - model)/ferror)**2))

    return ll

def logprior(sol, time):
    badprior = -np.inf
    lprior = 0

    min_t = min(time)
    max_t = max(time)

    # Todo: Change to handle multiple planets
    ubounds = [1e3, 2, 1, 1, 1, 1, 1, 5, max_t, max_t, 2, 1, 1, 1, 5, 1e3, 1e3, 1e4]
    lbounds = [1e-4, 0, -1, 0, 0, 0, 0, -5, min_t, min_t, 0, 0, -1, -1, -5, 0, -1e3, 0]

    for i in range(len(sol)):
        if lbounds[i] <= sol[i] <= ubounds[i]:
            continue
        else:
            return badprior
    
    return lprior
