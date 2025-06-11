import numpy as np
from scipy.optimize import least_squares
from utils_python.transitmodel import transitModel

def fitFromBLS(gbls_ans, time, flux, ferror, itime):
    """
    Fits a transit model using the answers from the bls algorithm
    """
    id_to_fit = np.array([0, 7, 8, 9, 10, 11]) # We fit only: rho, zpt, t0, Per, b, Rp/Rs

    sol = np.zeros(18) # Single planet model has up-to 18-model parameters

    ### Handle special cases
    if gbls_ans.depth < 0:
        gbls_ans.depth = 0

    # Set the initial guess using the bls answers.
    # ld coeff and rho are temporary values

    sol[0]  = 0.6  # Mean stellar density (g/cm^3)
    sol[1]  = 0.0  # Only used for non-linear limb-darkening
    sol[2]  = 0.0  # Only used for non-linear limb-darkening
    sol[3]  = 0.6  # q1 (limb-darkening)
    sol[4]  = 0.4  # q2 (limb-darkening)
    sol[5]  = 0.0  # dilution
    sol[6]  = 0.0  # Velocity offset
    sol[7]  = 0.0  # Photometric zero point
    sol[8]  = gbls_ans.epo             # Center of transit time (days)
    sol[9]  = gbls_ans.bper            # Orbital Period (days)
    sol[10] = 0.5                      # Impact parameter
    sol[11] = np.sqrt(gbls_ans.depth)  # Rp/R*
    sol[12] = 0.0  # sqrt(e)cos(w)
    sol[13] = 0.0  # sqrt(e)sin(w)
    sol[14] = 0.0  # RV amplitude (m/s)
    sol[15] = 0.0  # thermal eclipse depth (ppm)
    sol[16] = 0.0  # Ellipsodial variations (ppm)
    sol[17] = 0.0  # Albedo amplitude (ppm)

    # Sometimes t0 is negative and crashes the fit
    if gbls_ans.epo < 0:
        sol[8] += gbls_ans.bper

    return fitTransitModel(sol, id_to_fit, time, flux, ferror, itime)

def createBounds(time, id_to_fit):
    """
    Creates the bounds for the parameters
    """
    min_t = min(time)
    max_t = max(time)

    lower_bound = np.array([1e-4, 0, -1, 0, 0, 0, -np.inf, -np.inf, min_t, min_t, 0, 0, -1, -1, -np.inf, -np.inf, -np.inf, -np.inf])
    upper_bound = np.array([1e3, 2, 1, 1, 1, 1, np.inf, np.inf, max_t, max_t, 2, 1, 1, 1, np.inf, np.inf, np.inf, np.inf])

    return (lower_bound[id_to_fit], upper_bound[id_to_fit])

def fitTransitModel(sol, id_to_fit, time, flux, ferror, itime):
    """
    Function to call for fitting
    """

    # Fit only the parameters in id_to_fit
    sol_full = np.copy(sol) # Copy the initial guess
    def wrapperTransit(sol_free, time, flux, ferror, itime):
        for i, ind in enumerate(id_to_fit):
            sol_full[ind] = sol_free[i]
        return transitToOptimize(sol_full, time, flux, ferror, itime)
    
    bounds = createBounds(time, id_to_fit)

    res = least_squares(wrapperTransit, sol[id_to_fit], bounds=bounds, args=(time, flux, ferror, itime))

    # Recreate the full solution with the result
    for i, ind in enumerate(id_to_fit):
        sol_full[ind] = res.x[i]

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