import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils_python.transitmodel import transitModel
from utils_python.keplerian import transitDuration

def plotTransit(phot, sol, nintg=41, pl_to_plot=0):
    """
    Plots a transit model. Assuming time is in days. Set flux=0 for no scatterplot.

    phot: Phot object from reading data file
    sol: Transit model object with parameters
    nintg: Number of points inside the integration time
    pl_to_plot: Index of planet to plot. 0 being the first planet
    """

    # Read phot class
    time = phot.time - min(phot.time)
    flux = phot.flux + 1
    itime = phot.itime

    t0 = sol.t0[pl_to_plot]
    per = sol.per[pl_to_plot]
    zpt = sol.zpt

    # Copy the original Rp/R* before modifying it
    rdr = sol.rdr.copy()

    # Remove the other planets from the model
    for i in range(sol.npl):
        if i != pl_to_plot:
            sol.rdr[i] = 0

    tmodel = transitModel(sol.to_array(), time, itime, nintg) - zpt
    flux = flux - zpt # Remove the zero point to always plot around 1

    tdur = transitDuration(sol, pl_to_plot)*24
    if tdur < 0.01 or np.isnan(tdur):
        tdur = 2

    # Restore the original Rp/R*
    sol.rdr = rdr

    # Fold the time array and sort it
    phase = (time - per*np.floor(time/per) - t0 + per*np.floor(t0/per))*24
    i_sort = np.argsort(phase)
    phase_sorted = phase[i_sort]
    model_sorted = tmodel[i_sort]

    stdev = np.std(flux - tmodel)

    # Find bounds of plot
    i1, i2 = np.searchsorted(phase_sorted, (-tdur, tdur))
    if i1 == i2:
        i1 = 0
        i2 = len(model_sorted)
    ymin = min(model_sorted[i1:i2])
    ymax = max(model_sorted[i1:i2])
    y1 = ymin - 0.1*(ymax-ymin) - 2.0*stdev
    y2 = ymax + 0.1*(ymax-ymin) + 2.0*stdev
    if np.abs(y2 - y1) < 1.0e-10:
        y1 = min(flux)
        y2 = max(flux)

    mpl.rcParams.update({'font.size': 22}) # Adjust font
    plt.figure(figsize=(12,6)) # Adjust size of figure
    plt.scatter(phase, flux, c="blue", s=100.0, alpha=0.35, edgecolors="none") #scatter plot
    plt.plot(phase_sorted, model_sorted, c="red", lw=3.0)
    plt.xlabel('Phase (hours)') #x-label
    plt.ylabel('Relative Flux') #y-label
    plt.axis((-1.5*tdur, 1.5*tdur, y1, y2))
    plt.tick_params(direction="in")
    plt.show()

def printParams(sol):
    """
    Prints the parameters in a nice way.

    sol: Transit model object containing the parameters to print
    """

    stellarDict = {
        "ρ* (g/cm³)": "rho", "c1": "nl1", "c2": "nl2", "q1": "nl3", "q2": "nl4",
        "Dilution": "dil", "Velocity Offset": "vof", "Photometric zero point": "zpt"
    }

    planetDict = {
        "t0 (days)": "t0", "Period (days)": "per", "Impact parameter": "bb", "Rp/R*": "rdr",
        "sqrt(e)cos(w)": "ecw", "sqrt(e)sin(w)": "esw", "RV Amplitude (m/s)": "krv",
        "Thermal eclipse depth (ppm)": "ted", "Ellipsoidal variations (ppm)": "ell", "Albedo amplitude (ppm)": "alb"
    }

    # Stellar params
    for key in stellarDict:
        var_name = stellarDict[key]
        val = getattr(sol, var_name)
        err = getattr(sol, "d" + var_name)

        if val != 0:
            exponent = np.floor(np.log10(abs(val)))
        else:
            exponent = 1

        if abs(exponent) > 2:
            print(f"{key + ':':<30} {val:>10.3e} ± {err:.3e}")
        elif len(str(val)) > 7:
            print(f"{key + ':':<30} {val:>10.7f} ± {err:.7f}")
        else:
            print(f"{key + ':':<30} {val:>10} ± {err}")

    # Planet params
    for j in range(sol.npl):
        if sol.npl > 1:
            print(f"\nPlanet #{j + 1}:")
        for key in planetDict:
            var_name = planetDict[key]
            val = getattr(sol, var_name)
            err = getattr(sol, "d" + var_name)

            p_val = val[j]
            p_err = err[j]
            if p_val != 0:
                exponent = np.floor(np.log10(abs(p_val)))
            else:
                exponent = 1

            if abs(exponent) > 2:
                print(f"{key + ':':<30} {p_val:>10.3e} ± {p_err:.3e}")
            elif len(str(p_val)) > 7:
                print(f"{key + ':':<30} {p_val:>10.7f} ± {p_err:.7f}")
            else:
                print(f"{key + ':':<30} {p_val:>10} ± {p_err}")
