import numpy as np
from scipy.optimize import least_squares
import utils_python.transitmodel as transitm

def createBounds(time, id_to_fit, sol_obj):
    """
    Creates the bounds for the parameters

    time: time array
    id_to_fit: Array containing the indices of the parameters to fit
    sol_obj: Transit model object with initial parameters
    """
    min_t = min(time)
    max_t = max(time)

    # Rho and Rp/Rs are in log space
    lower_bound = np.array([np.log(1e-4), 0, -1, 0, 0, 0, -np.inf, -np.inf, min_t, 0, 0, -np.inf, -1, -1, -np.inf, -np.inf, -np.inf, -np.inf])
    upper_bound = np.array([np.log(1e3), 2, 1, 1, 1, 1, np.inf, np.inf, max_t, max_t, 2, 0, 1, 1, np.inf, np.inf, np.inf, np.inf])

    # Expand bounds for multiple planets
    for i in range(sol_obj.npl - 1):
        lower_bound = np.append(lower_bound, lower_bound[transitm.nb_st_param : (transitm.nb_pl_param + transitm.nb_st_param)])
        upper_bound = np.append(upper_bound, upper_bound[transitm.nb_st_param : (transitm.nb_pl_param + transitm.nb_st_param)])

    return (lower_bound[id_to_fit], upper_bound[id_to_fit])

def fitTransitModel(sol_obj, params_to_fit, phot, nintg=41, zerotime=0, use_flux_f=False, use_icut=False, ntt=-1, tobs=-1, omc=-1):
    """
    Function to call for fitting

    sol_obj: Transit model object with initial parameters
    params_to_fit: List containing strings of the names of the parameters to fit according to the tm class
    phot: Phot object from reading data file
    zerotime: Offset to shift time to 0
    use_flux_f: Boolean. Use preconditionned data or not
    use_icut: Boolean. Use phot.icut or not
    ntt: 1D array containing nb of ttv. shape=(nb_planet,)
    tobs: 2D array containing times of ttv. shape=(nb_planet, nb_ttv)
    omc: 2D array of o-c. shape=(nb_planet, nb_ttv)

    return: New transit model object containing the parameters and errors after fitting
    """

    n_planet = sol_obj.npl
    nb_pts = len(phot.time)
    # Handle TTV inputs
    if type(ntt) is int:
        ntt = np.zeros(n_planet, dtype="int32") # Number of TTVs measured 
        tobs = np.zeros((n_planet, nb_pts)) # Time stamps of TTV measurements (days)
        omc = np.zeros((n_planet, nb_pts)) # TTV measurements (O-C) (days)

    # Handle bad data cut
    if use_icut:
        icut = phot.icut
    else:
        icut = np.zeros(nb_pts)
    
    # Read phot class
    time = (phot.time - zerotime)[icut == 0]
    if use_flux_f:
        flux = (phot.flux_f - np.median(phot.flux_f) + 1)[icut == 0]
    else:
        flux = (phot.flux - np.median(phot.flux) + 1)[icut == 0]
    ferror = phot.ferr[icut == 0]
    itime = phot.itime[icut == 0]
    
    # Transform solution object to array
    sol = sol_obj.to_array()

    log_space_params = np.array([transitm.var_to_ind["rho"], transitm.var_to_ind["rdr"]]) # Rho and Rp/Rs are in log space
    id_to_fit = np.array([transitm.var_to_ind[param] for param in params_to_fit])

    # Expand indices arrays to fit multiple planets
    for i in range(sol_obj.npl - 1):
        mask = (id_to_fit >= transitm.nb_st_param) & (id_to_fit < (transitm.nb_pl_param + transitm.nb_st_param))
        id_to_fit = np.append(id_to_fit, id_to_fit[mask] + (i+1)*transitm.nb_pl_param)

        mask_log = (log_space_params >= transitm.nb_st_param) & (log_space_params < (transitm.nb_pl_param + transitm.nb_st_param))
        log_space_params = np.append(log_space_params, log_space_params[mask_log] + (i+1)*transitm.nb_pl_param)

    # Fit only the parameters in id_to_fit
    sol_full = np.copy(sol) # Copy the initial guess
    def wrapperTransit(sol_free, time, flux, ferror, itime, npl, nintg, ntt, tobs, omc):
        """ Wrapper function that takes only the free parameters as arguments """
        for i, ind in enumerate(id_to_fit):
            if ind in log_space_params:
                sol_full[ind] = np.exp(sol_free[i])
            else:
                sol_full[ind] = sol_free[i]
        
        return transitToOptimize(sol_full, time, flux, ferror, itime, npl, nintg, ntt, tobs, omc)
    
    bounds = createBounds(time, id_to_fit, sol_obj)

    # Take the log of log space parameters
    for i in log_space_params:
        # Only take the log if we fit the parameter
        if i in id_to_fit:
            sol[i] = np.log(sol[i])

    res = least_squares(wrapperTransit, sol[id_to_fit], bounds=bounds, \
                    args=(time, flux, ferror, itime, sol_obj.npl, nintg, ntt, tobs, omc))

    # Calculate error
    is_weighted = True
    fit_params, fit_error, fit_covar = calculate_parameter_errors(res, is_weighted)

    # Recreate the full solution with the result
    err_full = np.zeros(len(sol_full))
    for i, ind in enumerate(id_to_fit):
        if ind in log_space_params:
            sol_full[ind] = np.exp(fit_params[i])
            err_full[ind] = np.exp(fit_params[i]) * fit_error[i] # Error on f=exp(x) is exp(x)*error(x)
        else:
            sol_full[ind] = fit_params[i]
            err_full[ind] = fit_error[i]

    # Return to the transitmodel object
    fit_params = transitm.transit_model_class()
    fit_params.from_array(sol_full)
    fit_params.load_errors(err_full)

    return fit_params

def transitToOptimize(sol, time, flux, ferror, itime, npl, nintg, ntt, tobs, omc):
    """
    Handles constraints and returns the vector of differences. You shouldn't have to call this function

    sol: Array of transit model parameters for fitting
    time, flux, ferror: Data arrays
    itime: Integration time array
    npl: Number of planets
    """
    n = len(time)

    # Parameter constraints (Return a big number if constraint is not respected)
    for i in range(npl):
        b = sol[10 + i*transitm.nb_pl_param]
        Rp_Rs = sol[11 + i*transitm.nb_pl_param]
        if b > 1 + Rp_Rs:
            return np.full(n, 1e20)
    
    c1, c2, c3, c4 = sol[1], sol[2], sol[3], sol[4]
    # Quadratic coefficients
    if (c3 == 0 and c4 == 0):
        if not (0 < c1 < 2) or not (-1 < c2 < 1):
            return np.full(n, 1e20)

    # Kipping coefficients
    elif (c1 == 0 and c2 == 0):
        if not (0 < c3 < 1) or not (0 < c4 < 1):
            return np.full(n, 1e20)
    
    # Non linear law
    else:
        if not (0 < c1 < 1) or not (0 < c2 < 1) or not (0 < c3 < 1) or not (0 < c4 < 1):
            return np.full(n, 1e20)

    y_model = transitm._transitModel(sol, time, itime, nintg, ntt, tobs, omc)

    return (y_model - flux)/ferror

def calculate_parameter_errors(opt_result, residual_func_returns_weighted=True):
    """
    Calculates parameter errors (standard deviations) from scipy.optimize.least_squares results.

    Args:
        opt_result: The OptimizeResult object returned by least_squares.
        residual_func_returns_weighted (bool): Set to True if your residual
            function returns (data - model) / error. Set to False if it
            returns (data - model). Defaults to True.

    Returns:
        tuple: (best_fit_params, param_errors, covariance_matrix)
               param_errors and covariance_matrix might be None or contain NaNs
               if calculation fails.
    """
    best_fit_params = opt_result.x
    param_errors = None
    covariance_matrix = None

    if not opt_result.success:
        print(f"Warning: Optimization failed or did not converge: {opt_result.message}")
        return best_fit_params, np.full_like(best_fit_params, np.nan), None

    jacobian = opt_result.jac
    residuals = opt_result.fun # Residuals at the solution

    if jacobian is None or not np.all(np.isfinite(jacobian)):
         print("Warning: Jacobian is None or contains non-finite values. Cannot compute errors.")
         return best_fit_params, np.full_like(best_fit_params, np.nan), None

    N = len(residuals)
    M = len(best_fit_params)

    if N <= M:
        print(f"Warning: Number of data points ({N}) <= number of parameters ({M}).")
        print("Cannot reliably estimate errors or covariance.")
        return best_fit_params, np.full_like(best_fit_params, np.nan), None

    try:
        jtj = jacobian.T @ jacobian

        try:
            covariance_matrix = np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            print("Warning: Jacobian transpose * Jacobian is singular or near-singular.")
            print("Using pseudo-inverse. Parameter errors might be unreliable, check correlations.")
            covariance_matrix = np.linalg.pinv(jtj)

        if not residual_func_returns_weighted:
            dof = N - M
            mse = np.sum(residuals**2) / dof
            covariance_matrix = covariance_matrix * mse

        # --- Extract Errors ---
        # Variances are the diagonal elements
        # Ensure variances is a writeable copy using .copy()!
        variances = np.diag(covariance_matrix).copy() # <--- FIX IS HERE

        # Handle potential small negative values (now safe to modify)
        variances[variances < 0] = 0

        param_errors = np.sqrt(variances)

    except (np.linalg.LinAlgError, ValueError, TypeError, ZeroDivisionError) as e: # Added ZeroDivisionError
        print(f"Error calculating covariance/errors: {e}")
        param_errors = np.full_like(best_fit_params, np.nan)
        # Ensure covariance_matrix is None if errors couldn't be calculated,
        # unless it was successfully calculated before the error occurred during variance extraction
        if 'variances' not in locals(): # Check if error happened before variance calculation
             covariance_matrix = None

    return best_fit_params, param_errors, covariance_matrix