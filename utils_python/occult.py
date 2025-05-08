import numpy as np
from elliptic_int import ellE, ellK
import mpmath as mpm

def occultUniform(z0, p):
    n = len(z0)
    lambdae = np.zeros(n)

    kap1 = np.arccos(min((1 - p*p + z0*z0) / (2*z0), 1))
    kap0 = np.arccos(min((p*p + z0*z0 - 1) / (2*p*z0), 1))

    for i in range(n):
        z = z0[i]

        # Unobscured
        if z > 1 + p:
            lambdae[i] = 0

        # Completely obscured
        if z <= p - 1:
            lambdae[i] = 1

        # Partially obscured and crossing star
        if (abs(1 - p) < z <= (1 + p)):
            lambdae[i] = (p*p*kap0[i] + kap1[i] - 0.5 * np.sqrt(max(4*z*z - (1 + z*z - p*p)**2), 0)) / np.pi

        # Partially obscured
        if (z <= 1 - p):
            lambdae[i] = p*p

    return 1 - lambdae

def occultQuad(z0, u1, u2, p):
    n = len(z0)

    # Calculate lambdae first (Two loops are ran instead of one, so it might be a bit slower)
    lambdae = 1 - occultUniform(z0, p)

    # Omega is actually 4*Omega
    Omega = 1 - u1/3 - u2/6

    lambdad = np.zeros(n)
    etad = np.zeros(n)

    for i in range(n):
        z = z0[i]
        a = (z - p) ** 2
        b = (z + p) ** 2
        q = p*p - z*z

        # Unocculted
        if z >= 1 + p:
            lambdad[i] = 0
            etad[i] = 0

        # Completely occulted
        elif (p <= 1) and (z <= p-1):
            lambdad[i] = 1
            etad[i] = 1

        # Edge of the occulting star lies at the origin
        elif (abs(z-p) < 1e-4 * (z+p)):
            pass
        

def lambda1(p, z, k, a, b, q):
    Kk = ellK(k)
    Ek = ellE(k)
    Pk = mpm.ellippi((a-1)/a, k)

    temp1 = ((1-b) * (2*b + a - 3) - 3*q*(b - 2)) * Kk
    temp2 = 4*p*z * (z*z + 7*p*p - 4)*Ek

    return 1/(9*np.pi*np.sqrt(p*z)) * (temp1 + temp2 - 3*q/a * Pk)

def lambda2(p, z, k, a, b, q):
    Kk = ellK(1/k)
    Ek = ellE(1/k)
    Pk = mpm.ellippi((a-b)/a, 1/k)

    temp1 = (1 - 5*z*z + p*p + q*q) * Kk
    temp2 = (1 - a) * (z*z + 7*p*p - 4) * Ek

    return 2/(9*np.pi*np.sqrt(1 - a)) * (temp1 + temp2 - 3*q/a * Pk)