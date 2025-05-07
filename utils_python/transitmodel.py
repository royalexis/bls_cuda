import numpy as np
import utils_python.keplerian as kep
import utils_python.occult as occ

def transitModel(sol, time, itime, nintg=41):
    """
    Transit Model
    """

    # Constants
    G = 6.674e-11

    # Reading parameters
    density = sol[0]
    c1 = sol[1]
    c2 = sol[2]
    c3 = sol[3]
    c4 = sol[4]
    dil = sol[5]
    voff = sol[6]
    zpt = sol[7]

    nb_pts = len(time)
    tmodel = np.zeros(nb_pts)
    dtype = np.zeros(nb_pts) # Photometry only

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
            if ecw == 0:
                w = np.pi/2
            else:
                w = np.arctan2(esw/ecw)
            
            if ecw > 0 and esw < 0:
                w += 2*np.pi
            elif (ecw < 0 and esw >= 0) or (ecw <= 0 and esw < 0):
                w += np.pi

        # Calculate a/R*
        #a_Rs = (4*np.pi/3 * density * Per * Per) ** (1/3) # There is something wrong with this formula (probably a unit problem)
        a_Rs = 10 * (density * G * (Per*86400)**2/(3*np.pi)) ** (1/3)

        K = sol[10*ii + 8 + 6] # RV amplitude
        ted = sol[10*ii + 8 + 7]/1e6 # Occultation Depth
        ell = sol[10*ii + 8 + 8]/1e6 # Ellipsoidal variations
        ag = sol[10*ii + 8 + 9]/1e6 # Albedo amplitude

        # Calculate phi0
        Eanom = 2 * np.arctan(np.tan(w/2) / np.sqrt((1 + eccn)/(1 - eccn)))
        phi0 = Eanom - eccn*np.sin(Eanom)

        # Calculate inclinaison
        Tanom = kep.trueAnomaly(eccn, Eanom)
        d_Rs = kep.distance(a_Rs, eccn, Tanom) # Distance over R*
        incl = np.arccos(b/d_Rs)

        # Loop over all of the points
        for i in range(nb_pts):
            ttcor = 0 # For now

            tflux = np.zeros(nintg)
            bt = np.zeros(nintg)

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
                if (phi > np.pi):
                    phi -= 2*np.pi  # Phi is never used anywhere after this
                d_Rs = kep.distance(a_Rs, eccn, Tanom)

                x2 = d_Rs * np.sin(Tanom-w)
                y2 = d_Rs * np.cos(Tanom-w)*np.cos(incl)

                bt[j] = np.sqrt(x2*x2 + y2*y2)

                ### Calculation of RV, ellip and albedo here (to do)
            
            if dtype[i] == 0:
                if y2 >= 0:
                    # Check for transit
                    is_transit = 0
                    for j in range(nintg):
                        if bt[j] <= 1 + Rp_Rs:
                            is_transit = 1
                            # Could probably optimize this

                    if is_transit:
                        # Quadratic coefficients
                        if (c3 == 0 and c4 == 0):
                            tflux = occ.occultUniform(bt, Rp_Rs)
                        
                        # Kipping coefficients
                        elif (c1 == 0 and c2 == 0):
                            pass # To do
                        
                        # Non linear
                        else:
                            pass # To do

                    # If no transit, tflux = 1
                    else:
                        tflux = np.ones(nintg)

                    # Temp value
                    tm = 0
                    for j in range(nintg):
                        # Could also probably optimize this since Rp/Rs is constant
                        if Rp_Rs <= 0:
                            tflux[j] = 1
                        # Add all the contributions (for now there is only one)
                        tm += tflux[j]
                    tm = tm/nintg
                
                # Eclipse
                else:
                    tm = 1
                    pass # To do
            else:
                tm = 1
                pass # To do

            tmodel[i] += tm

    return tmodel

