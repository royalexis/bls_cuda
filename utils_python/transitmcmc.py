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
    log_space_params = np.array([transitm.var_to_ind["rho"], transitm.var_to_ind["rdr"]]) # Rho and Rp/Rs are in log space

    # Expand indices arrays to fit multiple planets
    for i in range(sol.npl - 1):
        id_to_fit = np.append(id_to_fit, id_to_fit[id_to_fit >= transitm.nb_st_param] + transitm.nb_pl_param)
        log_space_params = np.append(log_space_params, log_space_params[log_space_params >= transitm.nb_st_param] + transitm.nb_pl_param)

    sol_a = sol.to_array()
    
    def newLogprob(fit_sol, time, flux, ferror, itime):
        for i, ind in enumerate(id_to_fit):
            if ind in log_space_params:
                sol_a[ind] = np.exp(fit_sol[i])
            else:
                sol_a[ind] = fit_sol[i]
        return logprob(sol_a, time, flux, ferror, itime)
    
    err_a = sol.err_to_array()

    # Change parameters in log space
    for i in log_space_params:
        if i in id_to_fit:
            err_a[i] = err_a[i]/sol_a[i] # Error on f=ln(x) is error(x)/x
            sol_a[i] = np.log(sol_a[i])
    
    return newLogprob, sol_a[id_to_fit], err_a[id_to_fit]

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
    log_space_params = np.array([transitm.var_to_ind["rho"], transitm.var_to_ind["rdr"]])

    # Expand indices arrays to fit multiple planets
    for i in range(sol.npl - 1):
        id_to_fit = np.append(id_to_fit, id_to_fit[id_to_fit >= transitm.nb_st_param] + transitm.nb_pl_param)
        log_space_params = np.append(log_space_params, log_space_params[log_space_params >= transitm.nb_st_param] + transitm.nb_pl_param)

    sol_full = sol.to_array()
    err_full = np.zeros(len(sol_full))
    for i, ind in enumerate(id_to_fit):
        if ind in log_space_params:
            sol_full[ind] = np.exp(mm[i])
            err_full[ind] = np.exp(mm[i]) * std[i] # Error on f=exp(x) is exp(x)*error(x)
        else:
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

    ubounds = np.array([1e3, 2, 1, 1, 1, 1, 1, 5, max_t, max_t, 2, 1, 1, 1, 5, 1e3, 1e3, 1e4])
    lbounds = np.array([1e-4, 0, -1, 0, 0, 0, 0, -5, min_t, min_t, 0, 0, -1, -1, -5, 0, -1e3, 0])

    npl = (len(sol) - transitm.nb_st_param) // transitm.nb_pl_param

    # Expand bounds for multiple planets
    for i in range(npl - 1):
        lower_bound = np.append(lower_bound, lower_bound[transitm.nb_st_param:])
        upper_bound = np.append(upper_bound, upper_bound[transitm.nb_st_param:])

    for i in range(len(sol)):
        if lbounds[i] <= sol[i] <= ubounds[i]:
            continue
        else:
            return badprior
    
    return lprior
