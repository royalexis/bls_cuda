import numpy as np
import utils_python.keplerian as kep
import utils_python.occult as occ
from utils_python.effects import albedoMod
from numba import njit

# Constants
G = 6.674e-11
Cs = 2.99792458e8

@njit
def transitModel(sol, time, itime, nintg=41):
    """
    Transit Model
    """

    # Reading parameters
    density = sol[0]
    c1 = sol[1]
    c2 = sol[2]
    c3 = sol[3]
    c4 = sol[4]
    dil = sol[5]
    voff = sol[6]
    zpt = sol[7]

    # Kipping Coefficients
    a1 = 2 * np.sqrt(c3) * c4
    a2 = np.sqrt(c3) * (1 - 2*c4)

    nb_pts = len(time)
    tmodel = np.zeros(nb_pts)
    dtype = np.zeros(nb_pts) # Photometry only

    tflux = np.zeros(nintg)
    vt = np.zeros(nintg)
    tide = np.zeros(nintg)
    alb = np.zeros(nintg)
    bt = np.zeros(nintg)

    lambdad = np.zeros(nintg)
    etad = np.zeros(nintg)
    lambdae = np.zeros(nintg)

    # Temporary
    n_planet = 1

    # Loop over every planet
    for ii in range(n_planet):

        # Read parameters for the planet
        epoch = sol[10*ii + 8 + 0]
        Per = sol[10*ii + 8 + 1]
        b = sol[10*ii + 8 + 2]
        Rp_Rs = sol[10*ii + 8 + 3]

        ecw = sol[10*ii + 8 + 4]
        esw = sol[10*ii + 8 + 5]
        eccn = ecw*ecw + esw*esw

        # Calculation for omega (w) here
        if eccn >= 1:
            eccn = 0.99
        elif eccn == 0:
            w = 0
        else:
            # arctan2 gives a result in [-pi, pi], so we add 2pi to the negative values
            w = np.arctan2(esw, ecw)
            if w < 0:
                w += 2*np.pi

        # Calculate a/R*
        #a_Rs = (4*np.pi/3 * density * Per * Per) ** (1/3) # There is something wrong with this formula (probably a unit problem)
        a_Rs = 10 * (density * G * (Per*86400)**2 / (3*np.pi)) ** (1/3)

        K = sol[10*ii + 8 + 6] # RV amplitude
        ted = sol[10*ii + 8 + 7]/1e6 # Occultation Depth
        ell = sol[10*ii + 8 + 8]/1e6 # Ellipsoidal variations
        ag = sol[10*ii + 8 + 9]/1e6 # Albedo amplitude

        # Calculate phi0
        Eanom = 2 * np.arctan(np.tan(w/2) * np.sqrt((1 - eccn)/(1 + eccn)))
        phi0 = Eanom - eccn*np.sin(Eanom)

        # Calculate inclinaison
        Tanom = kep.trueAnomaly(eccn, Eanom)
        d_Rs = kep.distance(a_Rs, eccn, Tanom) # Distance over R*
        incl = np.arccos(b/d_Rs)

        # Loop over all of the points
        for i in range(nb_pts):
            ttcor = 0 # For now

            for j in range(nintg):
                
                # Time-Convolution
                t = time[i] - itime[i] * (0.5 - 1/(2*nintg) - j/nintg) - epoch - ttcor

                phi = t/Per - np.floor(t/Per)
                Manom = phi * 2*np.pi + phi0

                # Make sure Manom is in [0, 2pi]
                if (Manom > 2*np.pi):
                    Manom -= 2*np.pi
                if (Manom < 0):
                    Manom += 2*np.pi
                
                Eanom = kep.solve_kepler_eq(eccn, Manom, Eanom)
                Tanom = kep.trueAnomaly(eccn, Eanom)
                d_Rs = kep.distance(a_Rs, eccn, Tanom)

                x2 = d_Rs * np.sin(Tanom-w)
                y2 = d_Rs * np.cos(Tanom-w)*np.cos(incl)

                bt[j] = np.sqrt(x2*x2 + y2*y2)

                # Calculation of RV, ellip and albedo here

                vt[j] = K * (np.cos(Tanom - w + np.pi/2) + eccn*np.cos(-w + np.pi/2))

                tide[j] = ell * (d_Rs/a_Rs)**(1/3) * np.cos(2*(Tanom-w + np.pi/2))

                alb[j] = albedoMod(Tanom - w, ag) * a_Rs/d_Rs
            
            if dtype[i] == 0:
                if y2 >= 0:
                    # Check for transit
                    is_transit = 0
                    for b in bt:
                        if b <= 1 + Rp_Rs:
                            is_transit = 1
                            break
                    
                    if is_transit:
                        # Quadratic coefficients
                        if (c3 == 0 and c4 == 0):
                            tflux = occ.occultQuad(bt, c1, c2, Rp_Rs, lambdad, etad, lambdae)
                        
                        # Kipping coefficients
                        elif (c1 == 0 and c2 == 0):
                            tflux = occ.occultQuad(bt, a1, a2, Rp_Rs, lambdad, etad, lambdae)
                        
                        # Non linear
                        else:
                            tflux = occ.occultSmall(bt, c1, c2, c3, c4, Rp_Rs)

                    # If no transit, tflux = 1
                    else:
                        tflux = np.ones(nintg)

                    if Rp_Rs <= 0:
                        tflux = np.ones(nintg)

                    # Add all the contributions
                    tm = tflux.sum() - vt.sum()/Cs + tide.sum() + alb.sum()

                    tm = tm/nintg
                
                # Eclipse
                else:
                    bp = bt/Rp_Rs
                    # Treat the star as the object blocking the light
                    occult = occ.occultUniform(bp, 1/Rp_Rs, lambdae)
                    
                    if Rp_Rs < 0:
                        ratio = np.zeros(nintg)
                    else:
                        ratio = 1 - occult

                    tm = (nintg - ted*ratio.sum()) - vt.sum()/Cs + tide.sum() + alb.sum()

                    tm = tm/nintg

                tm += (1 - tm)*dil # Add dilution
            
            # Radial velocity
            else:
                tm = 1
                pass # To do

            tmodel[i] += tm # /n_planet ? To check
    
    # Add zero point
    for i in range(nb_pts):
        if dtype[i] == 0:
            tmodel[i] += zpt
        else:
            tmodel[i] += voff - 1

    return tmodel

