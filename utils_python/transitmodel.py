import os
import concurrent.futures as cf
import numpy as np
import utils_python.keplerian as kep
import utils_python.occult as occ
from utils_python.effects import albedoMod
from numba import njit

# Constants
G = 6.674e-11
Cs = 2.99792458e8

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
    dtype = np.zeros(nb_pts) # Photometry only

    tflux = np.zeros(nintg)
    vt = np.zeros(nintg)
    tide = np.zeros(nintg)
    alb = np.zeros(nintg)
    bt = np.zeros(nintg)

    lambdad = np.zeros(nintg)
    etad = np.zeros(nintg)
    lambdae = np.zeros(nintg)

    # Calculation for multithreading
    max_processes = os.cpu_count()
    ndiv = nb_pts // (max_processes - 1) + 1
    tmodel = np.zeros((max_processes, ndiv))

    iarg = np.zeros((max_processes, ndiv), dtype=np.int32)
    for i in range(0, max_processes):
        for k, j in enumerate(range(i, nb_pts, max_processes)):
            iarg[i, k] = j

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


        """
        To Implement multiprocessing

        - Copy the inside of the nb_pts loop to a seperate njit function (equivalent to bls_kernel())
            -> The function returns the tmodel for one point
        - Make a wrapper to that function that calculates all the points for a certain process (equivalent to compute_bls_kernel())
        - Replace the loop over the points to a process pool executor, making sure to give the right arguments to the function.
            -> There will be a lot of args: (time, itime, epoch, bt, vt, ...)
            -> This might require creating a 2D array containing indices to get the time_i and itime_i (equivalent to the iarg array)
        - I will have to remove the njit decorator of this main function.
            -> This means probably putting the code above to a seperate njit function

        To do eventually: Use most of the resources where there's lot of computation (where there is a transit)
        """

        # Computes the transit using multiprocessing
        with cf.ProcessPoolExecutor(max_workers=max_processes) as executor:
            futures = {executor.submit(compute_transit, iarg[i], tflux, time, itime, dtype, bt, vt, tide, alb, lambdae, lambdad, etad,
                                        c1, c2, c3, c4, a1, a2, nintg, epoch, Per, phi0, eccn, a_Rs, incl, Eanom,
                                        w, K, ell, ag, Rp_Rs, ted, dil, ndiv) : i for i in range(max_processes)}
            
            for future in cf.as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    tmodel[i,:] = result
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        tmodel = tmodel.T.ravel()[:nb_pts]
    
    # Add zero point
    for i in range(nb_pts):
        if dtype[i] == 0:
            tmodel[i] += zpt
        else:
            tmodel[i] += voff - 1

    return tmodel

def compute_transit(iarg, tflux, time, itime, dtype, bt, vt, tide, alb, lambdae, lambdad, etad,
                    c1, c2, c3, c4, a1, a2, nintg, epoch, Per, phi0, eccn, a_Rs, incl, Eanom,
                    w, K, ell, ag, Rp_Rs, ted, dil, ndiv):
    """This function computes all the the transit points that a certain process calculates"""
    
    tm = np.zeros(ndiv)

    for j, i in enumerate(iarg):
        tm[j] = transitOnePoint(tflux, time[i], itime[i], dtype[i], bt, vt, tide, alb, lambdae, lambdad, etad,
                    c1, c2, c3, c4, a1, a2, nintg, epoch, Per, phi0, eccn, a_Rs, incl, Eanom,
                    w, K, ell, ag, Rp_Rs, ted, dil)
        
    return tm


@njit
def transitOnePoint(tflux, time_i, itime_i, dtype_i, bt, vt, tide, alb, lambdae, lambdad, etad,
                    c1, c2, c3, c4, a1, a2, nintg, epoch, Per, phi0, eccn, a_Rs, incl, Eanom,
                    w, K, ell, ag, Rp_Rs, ted, dil):
    """This function computes the transit model for a single point"""

    ttcor = 0 # For now

    for j in range(nintg):
                
        # Time-Convolution
        t = time_i - itime_i * (0.5 - 1/(2*nintg) - j/nintg) - epoch - ttcor

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
            
    if dtype_i == 0:
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
                tflux[:] = 1

            if Rp_Rs <= 0:
                tflux[:] = 1

            # Add all the contributions
            tm = 0
            for j in range(nintg):
                tm += tflux[j] - vt[j]/Cs + tide[j] + alb[j]

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

            tm = 0
            for j in range(nintg):
                tm += 1 - ted*ratio[j] - vt[j]/Cs + tide[j] + alb[j]

            tm = tm/nintg

        tm += (1 - tm)*dil # Add dilution
            
    # Radial velocity
    else:
        tm = 1
        pass # To do

    return tm # /n_planet ? To check
