import numpy as np
from scipy.optimize import least_squares
from utils_python.transitmodel import transitModel

def fitTransitModel(sol, time, flux, ferror, itime):
    bounds = ([0, 0, 0, 0, 0, 0, -np.inf, -np.inf, 0, 0, 0, 0, -1, -1, -np.inf, -np.inf, -np.inf, -np.inf],
              [np.inf, 1, 1, 1, 1, 1, np.inf, np.inf, np.inf, np.inf, 2, 1, 1, 1, np.inf, np.inf, np.inf, np.inf])

    res = least_squares(transitToOptimize, sol, bounds=bounds, args=(time, flux, ferror, itime))

    return res.x

def transitToOptimize(sol, time, flux, ferror, itime):

    # For now, we manually set the parameters we don't want to fit
    sol[0]  = 0.8  # Mean stellar density (g/cm^3)
    sol[1]  = 0.0  # Only used for non-linear limb-darkening
    sol[2]  = 0.0  # Only used for non-linear limb-darkening
    # sol[3]  = 0.6  # q1 (limb-darkening)
    # sol[4]  = 0.4  # q2 (limb-darkening)
    sol[5]  = 0.0  # dilution
    sol[6]  = 0.0  # Velocity offset
    sol[7]  = 0.0  # photometric zero point
    sol[8]  = 0.449820           # Center of transit time (days)
    sol[9]  = 2.483415           # Orbital Period (days)
    sol[10] = 0.5                     # Impact parameter
    sol[11] = 0.0263 # Rp/R*
    sol[12] = 0.0  # sqrt(e)cos(w)
    sol[13] = 0.0  # sqrt(e)sin(w)
    sol[14] = 0.0  # RV amplitude (m/s)
    sol[15] = 0.0  # thermal eclipse depth (ppm)
    sol[16] = 0.0  # Ellipsodial variations (ppm)
    sol[17] = 0.0  # Albedo amplitude (ppm)

    y_model = transitModel(sol, time, itime=itime)

    return (y_model - flux)/ferror