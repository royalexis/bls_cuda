import numpy as np
from numba import njit

@njit
def trueAnomaly(eccn, Eanom):
    """
    Calculates the true anomaly
    """

    ratio = (1 + eccn) / (1 - eccn)
    return 2 * np.arctan(np.sqrt(ratio) * np.tan(Eanom/2))

@njit
def distance(a, eccn, Tanom):
    """
    Calculates the distance between the star and the planet
    """

    return a * (1 - eccn*eccn) / (1 + eccn * np.cos(Tanom))

@njit
def solve_kepler_eq(eccn, Manom, Eanom, thres=1e-6, itmax=100):
    """
    Solves the Kepler equation using the Newton-Raphson method
    """

    diff = 1
    i = 0

    while (diff >= thres and i < itmax):
        diff = (Eanom - eccn*np.sin(Eanom) - Manom) / (1 - eccn*np.cos(Eanom))
        Eanom -= diff
        
        diff = abs(diff)
        i += 1

    return Eanom
