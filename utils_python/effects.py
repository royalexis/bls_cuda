import numpy as np
from numba import njit

@njit
def albedoMod(phi, ag):
    phi = phi + np.pi
    if phi > 2*np.pi:
        phi = phi - 2*np.pi

    alpha = abs(phi)      
    alpha = alpha - 2*np.pi * int(alpha/(2.0*np.pi))
    if alpha > np.pi:
        alpha = abs(alpha - 2*np.pi)

    phase = (np.sin(alpha) + (np.pi - alpha)*np.cos(alpha)) / np.pi
      
    return ag * phase