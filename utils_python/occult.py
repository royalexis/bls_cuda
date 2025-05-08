import numpy as np
from utils_python.elliptic_int import ellE, ellK
import mpmath as mpm

def occultUniform(z0, p):
    n = len(z0)
    lambdae = np.zeros(n)

    for i in range(n):
        z = z0[i]
        kap1 = np.arccos(max(-1, min((1 - p*p + z*z) / (2*z), 1)))
        kap0 = np.arccos(max(-1, min((p*p + z*z - 1) / (2*p*z), 1)))

        # Unobscured
        if z > 1 + p:
            lambdae[i] = 0

        # Completely obscured
        if z <= p - 1:
            lambdae[i] = 1

        # Partially obscured and crossing star
        if (abs(1 - p) < z <= (1 + p)):
            lambdae[i] = (p*p*kap0 + kap1 - 0.5 * np.sqrt(max(4*z*z - (1 + z*z - p*p)**2, 0))) / np.pi

        # Partially obscured
        if (z <= 1 - p):
            lambdae[i] = p*p

    return 1 - lambdae

def occultQuad(z0, u1, u2, p):
    n = len(z0)

    # Calculate lambdae first (Two loops are run instead of one, so it might be a bit slower)
    lambdae = 1 - occultUniform(z0, p)

    # Omega is actually 4*Omega in the paper
    Omega = 1 - u1/3 - u2/6

    lambdad = np.zeros(n)
    etad = np.zeros(n)

    for i in range(n):
        z = z0[i]
        a = (z - p) ** 2
        b = (z + p) ** 2
        q = p*p - z*z
        kap1 = np.arccos(max(-1, min((1 - p*p + z*z) / (2*z), 1)))
        kap0 = np.arccos(max(-1, min((p*p + z*z - 1) / (2*p*z), 1)))

        # Unocculted
        if z >= 1 + p:
            lambdad[i] = 0
            etad[i] = 0

        # Completely occulted
        elif (p >= 1) and (z <= p-1):
            lambdad[i] = 1
            etad[i] = 1

        # Edge of the occulting star lies at the origin (z = p)
        elif (abs(z-p) < 1e-4 * (z+p)):
            if z >= 0.5:
                lambdad[i] = lambda3(p, p) # Why do we use p as the second argument here? If p=z, then k=1/(2p), not p
                etad[i] = eta1(kap0, kap1, p, z, a, b)

                if p == 0.5:
                    lambdad[i] = 1/3 - 4/(9*np.pi)
                    etad[i] = 3/32

            else:
                lambdad[i] = lambda4(p, p) # Same question for p here
                etad[i] = eta2(p, z)

        # Partly occults the source
        elif (0.5 + abs(p - 0.5) < z < 1 + p) or (p > 0.5 and abs(1 - p)*1.0001 < z < p):
            k = np.sqrt((1 - a)/(4*z*p))
            if k > 1:
                k = 0.99999

            lambdad[i] = lambda1(p, z, k, a, b, q)
            etad[i] = eta1(kap0, kap1, p, z, a, b)
            if z < p:
                lambdad[i] += 2/3

        # Transits the source
        elif p <= 1 and z <= (1 - p) * 1.0001:
            k = np.sqrt((b-a)/(1-a)) # k^(-1)
            if k > 1:
                k = 0.99999
            
            lambdad[i] = lambda2(p, z, 1/k, a, b, q)
            if z < p:
                lambdad[i] += 2/3
            if abs(p + z - 1) <= 1e-4:
                lambdad[i] = lambda5(p)

            etad[i] = eta2(p, z)
    
    c2 = u1 + 2*u2
    return 1 - ((1 - c2) * lambdae + c2*lambdad + u2*etad)/Omega
        

def lambda1(p, z, k, a, b, q):
    Kk = ellK(k)
    Ek = ellE(k)
    Pk = mpm.ellippi((a-1)/a, k*k) # ellippi takes m=k^2 as the second argument

    temp1 = ((1-b) * (2*b + a - 3) - 3*q*(b - 2)) * Kk
    temp2 = 4*p*z * (z*z + 7*p*p - 4)*Ek

    return 1/(9*np.pi*np.sqrt(p*z)) * (temp1 + temp2 - 3*q/a * Pk)

def lambda2(p, z, k, a, b, q):
    Kk = ellK(1/k)
    Ek = ellE(1/k)
    Pk = mpm.ellippi((a-b)/a, 1/(k*k)) # ellippi takes m=k^2 as the second argument

    temp1 = (1 - 5*z*z + p*p + q*q) * Kk
    temp2 = (1 - a) * (z*z + 7*p*p - 4) * Ek

    return 2/(9*np.pi*np.sqrt(1 - a)) * (temp1 + temp2 - 3*q/a * Pk)

def lambda3(p, k):
    Kk = ellK(1/(2*k))
    Ek = ellE(1/(2*k))

    temp1 = 16*p/(9*np.pi) * (2*p*p - 1) * Ek
    temp2 = Kk / (9*np.pi*p) * (1 - 4*p*p) * (3 - 8*p*p)

    return 1/3 + temp1 - temp2

def lambda4(p, k):
    Kk = ellK(2*k)
    Ek = ellE(2*k)

    temp1 = 4 * (2*p*p - 1) * Ek
    temp2 = (1 - 4*p*p) * Kk

    return 1/3 + 2/(9*np.pi) * (temp1 + temp2)

def lambda5(p):
    return 2/(3*np.pi) * np.arccos(1 - 2*p) - 4/(9*np.pi) * (3 + 2*p - 8*p*p) * np.sqrt(p*(1 - p)) # Why multiply by np.sqrt(p*(1 - p)) ?

def eta1(k0, k1, p, z, a ,b):
    return 1/(2*np.pi) * (k1 + 2*eta2(p, z)*k0 - 0.25*(1 + 5*p*p + z*z) * np.sqrt((1-a) * (b-1)))

def eta2(p, z):
    return p*p/2 * (p*p + 2*z*z)