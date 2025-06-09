import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils_python.transitmodel import transitModel
from utils_python.keplerian import transitDuration

def plotTransit(time, flux, sol, itime, nintg=41):
    """
    Plot a transit model. Assuming time is in days. Set flux=0 for no scatterplot
    """
    t0 = sol[8]
    per = sol[9]

    y_model = transitModel(sol, time, itime, nintg)

    tdur = transitDuration(sol)*24
    if tdur < 0.01:
        tdur = 2

    # Fold the time array and sort it
    phase = (time/per - np.floor(time/per) - t0/per)*per*24
    i_sort = np.argsort(phase)
    phase_sorted = phase[i_sort]
    model_sorted = y_model[i_sort]

    stdev = np.std(flux - y_model)

    # Find bounds of plot
    i1, i2 = np.searchsorted(phase_sorted, (-tdur, tdur))
    ymin = min(model_sorted[i1:i2])
    ymax = max(model_sorted[i1:i2])
    y1 = ymin - 0.1*(ymax-ymin) - 2.0*stdev
    y2 = ymax + 0.1*(ymax-ymin) + 2.0*stdev
    if np.abs(y2 - y1) < 1.0e-10:
        y1 = min(flux)
        y2 = max(flux)

    mpl.rcParams.update({'font.size': 22}) # Adjust font
    plt.figure(figsize=(12,6)) # Adjust size of figure
    if type(flux) is not int:
        plt.scatter(phase, flux, c="blue", s=100.0, alpha=0.35, edgecolors="none") #scatter plot
    plt.plot(phase_sorted, model_sorted, c="red", lw=3.0)
    plt.xlabel('Phase (hours)') #x-label
    plt.ylabel('Relative Flux') #y-label
    plt.axis((-1.5*tdur, 1.5*tdur, y1, y2))
    plt.show()

def printParams(sol, ind_to_print=[]):
    """
    Prints the parameters in a nice way.
    You can select the indices to print using a list or array.
    """

    paramsDict = {
        "ρ (g/cm³)": sol[0], "c1": sol[1], "c2": sol[2], "q1": sol[3], "q2": sol[4],
        "Dilution": sol[5], "Velocity Offset": sol[6], "Photometric zero point": sol[7],
        "t0 (days)": sol[8], "Period (days)": sol[9], "Impact parameter": sol[10], "Rp/R*": sol[11],
        "sqrt(e)cos(w)": sol[12], "sqrt(e)sin(w)": sol[13], "RV Amplitude (m/s)": sol[14],
        "Thermal eclipse depth (ppm)": sol[15], "Ellipsoidal variations (ppm)": sol[16], "Albedo amplitude (ppm)": sol[17]
    }

    # Select only certain keys if specified
    if len(ind_to_print) != 0:
        dictToPrint = {}
        for i, key in enumerate(paramsDict):
            if i in ind_to_print:
                dictToPrint[key] = paramsDict[key]
    else:
        dictToPrint = paramsDict

    # Print every value in the dictionary
    for key in dictToPrint:
        val = paramsDict[key]
        if val != 0:
            exponent = np.floor(np.log10(abs(val)))
        else:
            exponent = 1

        if abs(exponent) > 2:
            print(f"{key + ':':<30} {val:>10.3e}")
        elif len(str(val)) > 7:
            print(f"{key + ':':<30} {val:>10.7f}")
        else:
            print(f"{key + ':':<30} {val:>10}")
