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

def transitDuration(sol):
    """
    Calculates the transit duration
    """
    G = 6.674e-11

    density = sol.rho
    P = sol.per[0]
    b = sol.bb[0]
    Rp_Rs = sol.rdr[0]

    a_Rs = 10 * np.cbrt(density * G * (P*86400)**2 / (3*np.pi))

    temp1 = (1 + Rp_Rs)**2 - b*b
    temp2 = 1 - (b/a_Rs)**2

    return P/np.pi * np.arcsin(min(1/a_Rs * np.sqrt(temp1/temp2), 1))

def rhostar(P, tdur):
    """
    Approximates the star density using the period and transit duration.
    Uses a simplified formula to relate tdur to a/Rs.
    P: period in days
    tdur: transit duration in days
    """
    G = 6.674e-11

    # Change to secs
    P = P*86400
    tdur = tdur*86400

    rho = 3*P/(np.pi**2 * tdur**3 * G)

    # Change to g/cm^3
    return rho / 1000
