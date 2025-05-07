import numpy as np

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
        