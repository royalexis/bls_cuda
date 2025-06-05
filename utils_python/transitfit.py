import numpy as np
from scipy.optimize import least_squares
from utils_python.transitmodel import transitModel

def fitTransitModel(sol, time, flux, ferror, itime):
    min_t = min(time)
    max_t = max(time)

    bounds = ([1e-4, 0, -1, 0, 0, 0, -np.inf, -np.inf, min_t, min_t, 0, 0, -1, -1, -np.inf, -np.inf, -np.inf, -np.inf],
              [1e3, 2, 1, 1, 1, 1, np.inf, np.inf, max_t, max_t, 2, 1, 1, 1, np.inf, np.inf, np.inf, np.inf])

    res = least_squares(transitToOptimize, sol, bounds=bounds, args=(time, flux, ferror, itime))

    return res.x

def transitToOptimize(sol, time, flux, ferror, itime):

    # For now, we manually set the parameters we don't want to fit
    # sol[0]  = 0.8  # Mean stellar density (g/cm^3)
    sol[1]  = 0.0  # Only used for non-linear limb-darkening
    sol[2]  = 0.0  # Only used for non-linear limb-darkening
    sol[3]  = 0.6  # q1 (limb-darkening)
    sol[4]  = 0.4  # q2 (limb-darkening)
    sol[5]  = 0.0  # dilution
    sol[6]  = 0.0  # Velocity offset
    sol[7]  = 0.0  # photometric zero point
    # sol[8]  = 0.449820           # Center of transit time (days)
    sol[9]  = 2.483415           # Orbital Period (days)
    # sol[10] = 0.5                     # Impact parameter
    # sol[11] = 0.0263 # Rp/R*
    sol[12] = 0.0  # sqrt(e)cos(w)
    sol[13] = 0.0  # sqrt(e)sin(w)
    sol[14] = 0.0  # RV amplitude (m/s)
    sol[15] = 0.0  # thermal eclipse depth (ppm)
    sol[16] = 0.0  # Ellipsodial variations (ppm)
    sol[17] = 0.0  # Albedo amplitude (ppm)

    # Parameter constraints (Return a big number if constraint is not respected)
    b = sol[10]
    Rp_Rs = sol[11]
    if b > 1 + Rp_Rs:
        return 1e20
    
    c1, c2, c3, c4 = sol[1], sol[2], sol[3], sol[4]
    # Quadratic coefficients
    if (c3 == 0 and c4 == 0):
        if not (0 < c1 < 2) or not (-1 < c2 < 1):
            return 1e20

    # Kipping coefficients
    elif (c1 == 0 and c2 == 0):
        if not (0 < c3 < 1) or not (0 < c4 < 1):
            return 1e20
    
    # Non linear law
    else:
        if not (0 < c1 < 1) or not (0 < c2 < 1) or not (0 < c3 < 1) or not (0 < c4 < 1):
            return 1e20

    y_model = transitModel(sol, time, itime=itime)

    return (y_model - flux)/ferror