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
def solve_kepler_eq(eccn, Manom, Eanom, thres=1e-8, itmax=100):

    # First calculation to find diff
    Eold = max(Eanom, 0.0001)
    Eanom = Manom + eccn*np.sin(Eanom)
    diff = abs(1 - Eanom/Eold)
    Eold = Eanom

    i = 0
    while (diff >= thres and i < itmax):
        Eanom = Manom + eccn * np.sin(Eanom)
        diff = abs(1 - Eanom/Eold)
        Eold = Eanom
        i += 1

    return Eanom
