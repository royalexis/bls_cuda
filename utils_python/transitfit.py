import numpy as np
from scipy.optimize import least_squares
from utils_python.transitmodel import transitModel

def createBounds(time, id_to_fit):
    """
    Creates the bounds for the parameters
    """
    min_t = min(time)
    max_t = max(time)

    lower_bound = np.array([1e-4, 0, -1, 0, 0, 0, -np.inf, -np.inf, min_t, min_t, 0, 0, -1, -1, -np.inf, -np.inf, -np.inf, -np.inf])
    upper_bound = np.array([1e3, 2, 1, 1, 1, 1, np.inf, np.inf, max_t, max_t, 2, 1, 1, 1, np.inf, np.inf, np.inf, np.inf])

    return (lower_bound[id_to_fit], upper_bound[id_to_fit])

def fitTransitModel(sol, time, flux, ferror, itime):
    """
    Function to call for fitting
    """

    # Fit only the parameters in id_to_fit
    id_to_fit = np.array([0, 8, 10, 11])
    sol_full = np.copy(sol) # Copy the initial guess
    def wrapperTransit(sol_free, time, flux, ferror, itime):
        for i, id in enumerate(id_to_fit):
            sol_full[id] = sol_free[i]
        return transitToOptimize(sol_full, time, flux, ferror, itime)
    
    bounds = createBounds(time, id_to_fit)

    res = least_squares(wrapperTransit, sol[id_to_fit], bounds=bounds, args=(time, flux, ferror, itime))

    # Recreate the solution with the fixed params
    sol_full = np.copy(sol)
    for i, id in enumerate(id_to_fit):
        sol_full[id] = res.x[i]

    return sol_full

def transitToOptimize(sol, time, flux, ferror, itime):
    """
    Handles constraints and returns the vector of differences
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

    y_model = transitModel(sol, time, itime=itime)

    return (y_model - flux)/ferror