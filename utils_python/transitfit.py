import numpy as np
from scipy.optimize import least_squares
import utils_python.transitmodel as transitm
import transitPy5 as tpy5
import bls_cpu as gbls

def analyseLightCurve(gbls_inputs):
    """
    Function to call to analyse a light curve and get the best-fit parameters

    fileLoc: Location of string
    gbls_inputs: Inputs of the bls

    Returns: phot object, best-fit parameters returned by fit, error on params, answers from BLS
    """

    # Read data
    phot = tpy5.readphot(gbls_inputs.lcdir + gbls_inputs.filename)
    
    # Apply BLS
    gbls_inputs.zerotime = min(phot.time)
    gbls_ans = gbls.bls(gbls_inputs, phot.time[phot.icut == 0], phot.flux[phot.icut == 0])
    
    # Fit using BLS answers
    sol_fit, err_fit = fitFromBLS(gbls_ans, phot.time - gbls_inputs.zerotime, phot.flux + 1, phot.ferr, phot.itime)

    return phot, sol_fit, err_fit, gbls_ans

def fitFromBLS(gbls_ans, time, flux, ferror, itime):
    """
    Fits a transit model using the answers from the bls.

    gbls_ans: Answers from the bls
    time, flux, ferror: Data arrays
    itime: Integration time array

    return: Array containing the best-fit parameters for the transit model, Error on parameters
    """
    params_to_fit = ["rho", "zpt", "t0", "per", "bb", "rdr"] # We fit only: rho, zpt, t0, Per, b, Rp/Rs

    sol = transitm.transit_model_class() # Single planet model has up-to 18-model parameters

    # Set the initial guess using the bls answers.
    # ld coeff and rho are temporary values

    sol.rho = 0.6  
    sol.nl1 = 0.0  
    sol.nl2 = 0.0  
    sol.nl3 = 0.6  
    sol.nl4 = 0.4
    sol.dil = 0.0
    sol.vof = 0.0
    sol.zpt = np.mean(flux)
    sol.t0  = [gbls_ans.epo]
    sol.per = [gbls_ans.bper]
    sol.bb  = [0.5]
    if gbls_ans.depth < 0:
        sol.rdr = [1e-5]  
    else:
        sol.rdr = [np.sqrt(gbls_ans.depth)]
    sol.ecw = [0.0]
    sol.esw = [0.0]
    sol.krv = [0.0]
    sol.ted = [0.0]
    sol.ell = [0.0]
    sol.alb = [0.0]

    # Sometimes t0 is negative and crashes the fit
    if gbls_ans.epo < 0:
        sol.t0[0] += gbls_ans.bper

    return fitTransitModel(sol, params_to_fit, time, flux, ferror, itime)

def createBounds(time, id_to_fit):
    """
    Creates the bounds for the parameters

    time: time array
    id_to_fit: Array containing the indices of the parameters to fit
    """
    min_t = min(time)
    max_t = max(time)

    # Rho and Rp/Rs are in log space
    lower_bound = np.array([np.log(1e-4), 0, -1, 0, 0, 0, -np.inf, -np.inf, min_t, min_t, 0, -np.inf, -1, -1, -np.inf, -np.inf, -np.inf, -np.inf])
    upper_bound = np.array([np.log(1e3), 2, 1, 1, 1, 1, np.inf, np.inf, max_t, max_t, 2, 0, 1, 1, np.inf, np.inf, np.inf, np.inf])

    return (lower_bound[id_to_fit], upper_bound[id_to_fit])

def fitTransitModel(sol_obj, params_to_fit, time, flux, ferror, itime):
    """
    Function to call for fitting

    sol: Transit model object with initial parameters
    params_to_fit: Array containing the names of the parameters to fit according to the tm class
    time, flux, ferror: Data arrays
    itime: Integration time array

    return: Array containing the best-fit parameters for the transit model, Error on parameters
            These arrays are the same size as sol, with the fixed parameters untouched.
    """

    # Transform solution object to array
    sol = sol_obj.to_array()

    log_space_params = [transitm.var_to_ind["rho"], transitm.var_to_ind["rdr"]] # Rho and Rp/Rs are in log space
    id_to_fit = [transitm.var_to_ind[param] for param in params_to_fit]

    # Fit only the parameters in id_to_fit
    sol_full = np.copy(sol) # Copy the initial guess
    def wrapperTransit(sol_free, time, flux, ferror, itime):
        """ Wrapper function that takes only the free parameters as arguments """
        for i, ind in enumerate(id_to_fit):
            if ind in log_space_params:
                sol_full[ind] = np.exp(sol_free[i])
            else:
                sol_full[ind] = sol_free[i]
        return transitToOptimize(sol_full, time, flux, ferror, itime)
    
    bounds = createBounds(time, id_to_fit)

    # Take the log of log space parameters
    for i in log_space_params:
        sol[i] = np.log(sol[i])

    res = least_squares(wrapperTransit, sol[id_to_fit], bounds=bounds, args=(time, flux, ferror, itime))

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

def transitToOptimize(sol, time, flux, ferror, itime):
    """
    Handles constraints and returns the vector of differences. You shouldn't have to call this function

    sol: Array of transit model parameters for fitting
    time, flux, ferror: Data arrays
    itime: Integration time array
    """
    n = len(time)

    # Parameter constraints (Return a big number if constraint is not respected)
    b = sol[10]
    Rp_Rs = sol[11]
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

    y_model = transitm.transitModel(sol, time, itime=itime)

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