import numpy as np
import utils_python.transitmodel as transitm
import utils_python.keplerian as kep
import utils_python.mcmcroutines as mcmc
import transitfit5 as tf
import time
import matplotlib.pyplot as plt

def genmcmcInput(sol, params_to_fit):
    """
    Returns the new log function, the sol array and the beta array to use for mcmc

    sol: Transit Model object containing the initial parameters
    params_to_fit: List containing strings of the names of the parameters to fit according to the tm class

    return: New log prob function, Array of initial parameters to pass to mcmc, Initial guess for beta using errors
    """
    id_to_fit = np.array([transitm.var_to_ind[param] for param in params_to_fit])
    log_space_params = np.array([transitm.var_to_ind["rho"], transitm.var_to_ind["rdr"]]) # Rho and Rp/Rs are in log space

    # Expand indices arrays to fit multiple planets
    for i in range(sol.npl - 1):
        mask = (id_to_fit >= transitm.nb_st_param) & (id_to_fit < (transitm.nb_pl_param + transitm.nb_st_param))
        id_to_fit = np.append(id_to_fit, id_to_fit[mask] + (i+1)*transitm.nb_pl_param)

        mask_log = (log_space_params >= transitm.nb_st_param) & (log_space_params < (transitm.nb_pl_param + transitm.nb_st_param))
        log_space_params = np.append(log_space_params, log_space_params[mask_log] + (i+1)*transitm.nb_pl_param)

    sol_a = sol.to_array()

    def newLogprob(fit_sol, time, flux, ferror, itime, nintg, ntt, tobs, omc):
        for i, ind in enumerate(id_to_fit):
            if ind in log_space_params:
                sol_a[ind] = np.exp(fit_sol[i])
            else:
                sol_a[ind] = fit_sol[i]
        return logprob(sol_a, time, flux, ferror, itime, nintg, ntt, tobs, omc)
    
    err_a = sol.err_to_array()

    # Change parameters in log space
    for i in log_space_params:
        if i in id_to_fit:
            err_a[i] = err_a[i]/sol_a[i] # Error on f=ln(x) is error(x)/x
            sol_a[i] = np.log(sol_a[i])
    
    return newLogprob, sol_a[id_to_fit], err_a[id_to_fit]

def getParams(chain, burnin, sol, params_to_fit):
    """
    Generates a transit model object from the markov chain.

    chain: 2D Markov chain
    burnin: Markov chain burnin
    sol: Transit model object containing the initial guess. Used to recreate the full solution
    params_to_fit: List of parameters used for the mcmc fitting

    return: Transit model object containing the parameters from the mcmc
    """
    # Get params from mcmc
    cut_chain = chain[burnin:,:]
    mm = np.median(cut_chain, axis=0)
    std = np.std(cut_chain, axis=0)

    # Return to the full array
    id_to_fit = np.array([transitm.var_to_ind[param] for param in params_to_fit])
    log_space_params = np.array([transitm.var_to_ind["rho"], transitm.var_to_ind["rdr"]])

    # Expand indices arrays for multiple planets
    for i in range(sol.npl - 1):
        mask = (id_to_fit >= transitm.nb_st_param) & (id_to_fit < (transitm.nb_pl_param + transitm.nb_st_param))
        id_to_fit = np.append(id_to_fit, id_to_fit[mask] + (i+1)*transitm.nb_pl_param)

        mask_log = (log_space_params >= transitm.nb_st_param) & (log_space_params < (transitm.nb_pl_param + transitm.nb_st_param))
        log_space_params = np.append(log_space_params, log_space_params[mask_log] + (i+1)*transitm.nb_pl_param)

    sol_full = sol.to_array()
    err_full = np.zeros(len(sol_full))
    for i, ind in enumerate(id_to_fit):
        if ind in log_space_params:
            sol_full[ind] = np.exp(mm[i])
            err_full[ind] = np.exp(mm[i]) * std[i] # Error on f=exp(x) is exp(x)*error(x)
        else:
            sol_full[ind] = mm[i]
            err_full[ind] = std[i]

    # Generate the object
    sol_output = transitm.transit_model_class()
    sol_output.from_array(sol_full)
    sol_output.load_errors(err_full)

    return sol_output

def cutOutOfTransit(sol, phot, tdurcut=2, zerotime=0, cut_flux_f=False, cut_icut=False):
    """
    Cuts data that is out of transit. Considers all the planets in sol.

    sol: Transit model object containing parameters
    phot: Phot object containing data
    tdurcut: Fraction of the transit duration to keep on each side of the transit
    zerotime: Offset to start time at 0
    
    Returns a new phot object.
    """

    condition = False

    for i in range(sol.npl):
        phase = (phot.time - sol.t0[i] - zerotime)/sol.per[i] - np.floor((phot.time - sol.t0[i] - zerotime)/sol.per[i])
        phase[phase<-0.5] += 1.0
        phase[phase>0.5] -= 1.0

        tdur = kep.transitDuration(sol, i)/sol.per[i]

        condition |= (phase > -tdurcut*tdur) & (phase < tdurcut*tdur)

    phot_out = tf.phot_class()
    phot_out.time = phot.time[condition]
    phot_out.flux = phot.flux[condition]
    phot_out.ferr = phot.ferr[condition]
    phot_out.itime = phot.itime[condition]
    if cut_flux_f:
        phot_out.flux_f = phot.flux_f[condition]
    if cut_icut:
        phot_out.icut = phot.icut[condition]

    return phot_out

def logprob(sol, time, flux, ferror, itime, nintg, ntt, tobs, omc):
    return loglikehood(transitm._transitModel, sol, time, flux, ferror, itime, nintg, ntt, tobs, omc) + logprior(sol, time)

def loglikehood(modelFunc, sol, time, flux, ferror, itime, nintg, ntt, tobs, omc):

    model = modelFunc(sol, time, itime, nintg, ntt, tobs, omc)

    n = len(flux)

    if n < 1:
        ll = -1e30
    else:
        ll = -0.5*(n*np.log(2*np.pi) + np.sum(np.log(ferror*ferror) + ((flux - model)/ferror)**2))

    return ll

def logprior(sol, time):
    badprior = -np.inf
    lprior = 0

    min_t = min(time)
    max_t = max(time)

    ubounds = np.array([1e3, 2, 1, 1, 1, 1, 1, 5, max_t, max_t, 2, 1, 1, 1, 5, 1e3, 1e3, 1e4])
    lbounds = np.array([1e-4, 0, -1, 0, 0, 0, 0, -5, 0, 0, 0, 0, -1, -1, -5, 0, -1e3, 0])

    npl = (len(sol) - transitm.nb_st_param) // transitm.nb_pl_param

    # Expand bounds for multiple planets
    for i in range(npl - 1):
        lbounds = np.append(lbounds, lbounds[transitm.nb_st_param : (transitm.nb_pl_param + transitm.nb_st_param)])
        ubounds = np.append(ubounds, ubounds[transitm.nb_st_param : (transitm.nb_pl_param + transitm.nb_st_param)])

    for i in range(len(sol)):
        if lbounds[i] <= sol[i] <= ubounds[i]:
            continue
        else:
            return badprior
    
    return lprior

def plotChainsTransit(phot, chain, burnin, sol, params_to_fit, zerotime=0, nplot=100, use_flux_f=False, use_icut=False):
    """
    Plots a random selection of models from the chain.

    phot: phot object containing data
    chain: 2D Markov chain
    burnin: Markov chain burnin
    sol: Transit model object containing the initial guess. Used to recreate the full solution
    params_to_fit: List of parameters used for the mcmc fitting
    zerotime: Offset to start time at 0
    nplot: Number of models to plot
    """

    # Handle bad data cut
    if use_icut:
        icut = phot.icut
    else:
        icut = np.zeros(len(phot.time))
    
    # Read phot class
    time = (phot.time - zerotime)[icut == 0]
    if use_flux_f:
        flux = phot.flux_f[icut == 0]
    else:
        flux = phot.flux[icut == 0]
    itime = phot.itime[icut == 0]

    nmcmc = chain.shape[0]

    best_sol = getParams(chain, burnin, sol, params_to_fit)
    t0 = best_sol.t0[0]
    per = best_sol.per[0]

    # Fold the time array and sort it. Handle TTVs
    ph1 = t0/per - np.floor(t0/per)
    b_phase = np.empty(len(time))
    for i, x in enumerate(time):
        ttcor = 0
        t = x - ttcor
        b_phase[i] = (t/per - np.floor(t/per) - ph1) * per*24

    plt.figure(figsize=(12,6))
    plt.scatter(b_phase, flux, c="blue", s=100, alpha=0.35, edgecolors="none")

    for i in range(nplot):

        nchain = np.random.randint(burnin, nmcmc)

        # Get parameters
        sol_chain = getParams(chain[nchain,:].reshape((1,chain.shape[1])), 0, sol, params_to_fit)

        t0 = sol_chain.t0[0]
        per = sol_chain.per[0]

        # Fold the time array and sort it. Handle TTVs
        ph1 = t0/per - np.floor(t0/per)
        phase = np.empty(len(time))
        for i, x in enumerate(time):
            ttcor = 0
            t = x - ttcor
            phase[i] = (t/per - np.floor(t/per) - ph1) * per*24

        tmodel = transitm.transitModel(sol_chain, time, itime=itime)

        i_sort = np.argsort(phase)
        phase_sorted = phase[i_sort]
        model_sorted = tmodel[i_sort]
        plt.plot(phase_sorted, model_sorted, c="red", alpha=0.1)

    b_tmodel = transitm.transitModel(best_sol, time, itime=itime)
    i_sort = np.argsort(b_phase)
    phase_sorted = b_phase[i_sort]
    model_sorted = b_tmodel[i_sort]

    stdev = np.std(flux - b_tmodel)
    tdur = kep.transitDuration(sol, i_planet=0)*24
    if tdur < 0.01 or np.isnan(tdur):
        tdur = 2

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

    plt.xlabel('Phase (hours)')
    plt.ylabel('Relative Flux')
    plt.axis((-1.5*tdur, 1.5*tdur, y1, y2))
    plt.tick_params(direction="in")
    plt.show()

def demcmcRoutine(x, beta, phot, sol_a, serr, params, lnprob, zerotime=0, nintg=41, ntt=-1, tobs=-1, omc=-1,
                  use_flux_f=False, use_icut=False, verbose=True, progress_bar=True):
    start_time = time.time()

    n_planet = (len(sol_a) - transitm.nb_st_param) // transitm.nb_pl_param
    nb_pts = len(phot.time)
    # Handle TTV inputs
    if type(ntt) is int:
        ntt = np.zeros(n_planet, dtype="int32") # Number of TTVs measured 
        tobs = np.zeros((n_planet, nb_pts)) # Time stamps of TTV measurements (days)
        omc = np.zeros((n_planet, nb_pts)) # TTV measurements (O-C) (days)

    # Handle bad data cut
    if use_icut:
        icut = phot.icut
    else:
        icut = np.zeros(nb_pts)
    
    # Read phot class
    time_a = (phot.time - zerotime)[icut == 0]
    if use_flux_f:
        flux = phot.flux_f[icut == 0]
    else:
        flux = phot.flux[icut == 0]
    ferror = phot.ferr[icut == 0]
    itime = phot.itime[icut == 0]

    # Read params
    nsteps1 = params[0]
    nsteps2 = params[1]
    nsteps_inc = params[2]
    burninf = params[3]
    niter_cor = params[4]
    burnin_cor = params[5]
    nthin = params[6]
    nloopmax = params[7]
    converge_crit = params[8] 
    buf_converge_crit = params[9]

    # Check prior
    prior = logprior(sol_a, time_a)
    print(f"Prior: {prior}")

    #Check for run-away model
    TPnsteps=5000
    TPnthin=1
    
    chain,accept=mcmc.genchain(x,beta,TPnsteps,lnprob,mcmc.mhgmcmc,time_a,flux,ferror,itime,nintg, ntt, tobs, omc, progress=progress_bar)
    
    runtest=np.array(tf.checkperT0(chain,burninf,TPnthin,sol_a,serr))
    if verbose:
        print('runtest:',runtest)
    runtest2 = runtest + 1.0e-10 #add small eps to avoid division by zero.
    if int(np.sum(runtest2[runtest2<1.0]/runtest2[runtest2<1.0]))==4.0:
        TPflag=1 #flag for looping until convergence is met.
    else:
        TPflag=0 #run-away
        
    #if models are so far so good.. continue
    if TPflag==1:

        #get better beta 
        corscale=mcmc.betarescale(x,beta,niter_cor,burnin_cor,lnprob,mcmc.mhgmcmc,
                                  time_a,flux,ferror,itime,nintg, ntt, tobs, omc, imax=10, verbose=verbose, progress=progress_bar)
        
        #first run with M-H to create buffer.

        nloop=0
        nsteps=np.copy(nsteps1)
        mcmcloop=True
        while mcmcloop==True:
        
            nloop+=1 #count number of loops
            
            hchain1,haccept1=mcmc.genchain(x,beta*corscale,TPnsteps,lnprob,mcmc.mhgmcmc,
                                           time_a,flux,ferror,itime,nintg, ntt, tobs, omc, progress=progress_bar)
            hchain2,haccept2=mcmc.genchain(x,beta*corscale,TPnsteps,lnprob,mcmc.mhgmcmc,
                                           time_a,flux,ferror,itime,nintg, ntt, tobs, omc, progress=progress_bar)
            hchain3,haccept3=mcmc.genchain(x,beta*corscale,TPnsteps,lnprob,mcmc.mhgmcmc,
                                           time_a,flux,ferror,itime,nintg, ntt, tobs, omc, progress=progress_bar)

            if nloop==1:
                chain1=np.copy(hchain1)
                chain2=np.copy(hchain2)
                chain3=np.copy(hchain3)
                accept1=np.copy(haccept1)
                accept2=np.copy(haccept2)
                accept3=np.copy(haccept3)
            else:
                chain1=np.concatenate((chain1,hchain1))
                chain2=np.concatenate((chain2,hchain2))
                chain3=np.concatenate((chain3,hchain3))
                accept1=np.concatenate((accept1,haccept1))
                accept2=np.concatenate((accept2,haccept2))
                accept3=np.concatenate((accept3,haccept3))
            
            burnin=int(chain1.shape[0]*burninf)
            if verbose:
                mcmc.calcacrate(accept1,burnin)

            grtest=mcmc.gelmanrubin(chain1,chain2,chain3,burnin=burnin,npt=len(time_a))
            if verbose:
                print('Gelman-Rubin Convergence:')
                print('parameter  Rc')
                for i in range(0,len(chain1[1,:])):
                    print('%8s  %.4f' %(str(i),grtest[i]))
            if int(np.sum(grtest[grtest<buf_converge_crit]/grtest[grtest<buf_converge_crit]))==len(grtest):
                mcmcloop=False
            else:
                mcmcloop=True
                nsteps+=nsteps1
                
            runtest=np.array(tf.checkperT0(chain1,burninf,TPnthin,sol_a,serr))
            if verbose:
                print('runtest:',runtest)
            if int(np.sum(runtest[runtest<1.0]/runtest[runtest<1.0]))!=4.0:
                mcmcloop=False #run-away
                
            if nloop>=nloopmax: #break if too many loops
                mcmcloop=False
            
            if verbose:
                print("---- %s seconds ----" % (time.time() - start_time))
                
        ## Re-use runtest for buffer creation loop.
        #runtest=np.array(tf.checkperT0(chain1,burninf,TPnthin,sol,serr))
        #print('runtest:',runtest)
        if int(np.sum(runtest[runtest<1.0]/runtest[runtest<1.0]))==4.0:
            mcmcloop=True #flag for looping until convergence is met.
        else:
            mcmcloop=False #run-away

        #mcmcloop=True
        nloop=0
        nsteps=np.copy(nsteps2)
        while mcmcloop==True:

            nloop+=1 #count number of loops

            burnin=int(chain1.shape[0]*burninf)
            buffer=np.concatenate((chain1[burnin:],chain2[burnin:],chain3[burnin:])) #create buffer for deMCMC
            x1=np.copy(chain1[chain1.shape[0]-1,:])
            x2=np.copy(chain1[chain1.shape[0]-1,:])
            x3=np.copy(chain1[chain1.shape[0]-1,:])
            corbeta=0.3
            burnin=int(chain1.shape[0]*burninf)
            chain1,accept1=mcmc.genchain(x1,beta*corscale,nsteps,lnprob,mcmc.demhmcmc,time_a,flux,ferror,itime,\
                                         nintg, ntt, tobs, omc,buffer=buffer,corbeta=corbeta, progress=progress_bar)
            chain2,accept2=mcmc.genchain(x2,beta*corscale,nsteps,lnprob,mcmc.demhmcmc,time_a,flux,ferror,itime,\
                                         nintg, ntt, tobs, omc,buffer=buffer,corbeta=corbeta, progress=progress_bar)
            chain3,accept3=mcmc.genchain(x3,beta*corscale,nsteps,lnprob,mcmc.demhmcmc,time_a,flux,ferror,itime,\
                                         nintg, ntt, tobs, omc,buffer=buffer,corbeta=corbeta, progress=progress_bar)

            burnin=int(chain1.shape[0]*burninf)

            grtest=mcmc.gelmanrubin(chain1,chain2,chain3,burnin=burnin,npt=len(time_a))
            if verbose:
                print('Gelman-Rubin Convergence:')
                print('parameter  Rc')
                for i in range(0,len(chain1[1,:])):
                    print('%8s  %.4f' %(str(i),grtest[i]))

            if int(np.sum(grtest[grtest<converge_crit]/grtest[grtest<converge_crit]))==len(grtest):
                mcmcloop=False
            else:
                mcmcloop=True

            burnin=int(chain1.shape[0]*burninf)
            chain=np.concatenate((chain1[burnin:,],chain2[burnin:,],chain3[burnin:,]))
            accept=np.concatenate((accept1[burnin:,],accept2[burnin:,],accept3[burnin:,]))
            burnin=int(chain.shape[0]*burninf)
            if verbose:
                mcmc.calcacrate(accept,burnin)

            nsteps+=nsteps_inc #make longer chain to help with convergence
            
            #check for run-away Chain.
            runtest=np.array(tf.checkperT0(chain1,burninf,nthin,sol_a,serr))
            if verbose:
                print('runtest:',runtest)
            if int(np.sum(runtest[runtest<1.0]/runtest[runtest<1.0]))!=4.0:
                mcmcloop=False #run-away

            if nloop>=nloopmax: #break if too many loops
                mcmcloop=False
            
            if verbose:   
                print("---- %s seconds ----" % (time.time() - start_time))

        if verbose:  
            print("done %s seconds ---" % (time.time() - start_time))

        return chain, accept, burnin
