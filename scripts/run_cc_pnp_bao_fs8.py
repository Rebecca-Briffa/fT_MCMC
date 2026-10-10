import sys
from pathlib import Path

# so this script can be run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import emcee
import sys
from multiprocessing import Pool, cpu_count
from models.ft_plm.likelihoods import *
ncpu = cpu_count()
# from schwimmbad import MPIPool


# setting the priori function
def log_prior(theta):
    H_0, Om, p, s80, M  = theta
    if 0.1 < Om < 0.9 and 50. < H_0 < 100. and  -1.0 < p < 1.0 and -80 < M < 60 and 0.6 < s80 < 1.0:
        return 0.0
    else:
        return -np.inf

# compute the posterior probability
def log_probability(theta):
    H_0, Om, p, s80, M  = theta
    lp = log_prior(theta)
    #print(lp)
    if not np.isfinite(lp):
        return -np.inf

    # define the number for interpolation
    n_interpolation = 100
    Hz = np.empty((n_interpolation))
    dHz = np.empty((n_interpolation))

    # choose 100 evenly spaced of z between z_min and z_max
    z_min = 0.0
    z_max = 2.4
    z_int = np.linspace(z_min, z_max, n_interpolation)

    # # choose 100 evenly spaced of z between z_min and z_max
    # z_min1 = 0.0
    # z_max1 = 2.4
    # z_int1 = np.linspace(z_min1, z_max1, 100)


    # compute the H(z) at the chosen redshifts using the above solver
    for index, z in enumerate(z_int):
        H_cosmo = H_z(z, H_0, Om,p)
        Hz[index] = H_cosmo
        dHz[index] = solver_dHz(z, H_0, Om, p, H_cosmo)


    # obtaining the integrand d_H = c/H(z)
    d_H = c / Hz

    # interpolating the integrand to obtain a function for the integration
    integrand = interpolate.interp1d(z_int, d_H)
    # d = interpolate.interp1d(z_int, dHz)
    #
    # for index, z in enumerate(z_int):
    #     dH_cosmo = solver_dHz(z, H_0, Om, p, integrand1)
    #     dHz[index] = dH_cosmo


    d_C = []
    # integrating for each z and then appending, to obtain d_C at each redshift
    for j in range(len(z_int)):
        d_C.append(integrate.romberg(integrand, 0, z_int[j], divmax=20))

    d = diff_d_d0(z_int, H_0, Om,p)

    #print(Likelihood_Pantheon().loglkl(theta,M))
    lp = lp  +  Likelihood_fs8().loglkl(theta, z_int, d, Hz) + Likelihood_clocks().loglkl(theta, z_int, Hz) + Likelihood_Pantheon_SH0ES().loglkl(theta, z_int, Hz, d_C) \
         + Likelihood_BAO_Alam().loglkl(theta, z_int, Hz, d_C)+ Likelihood_bao_boss().loglkl(theta, z_int, Hz, d_C) + Likelihood_BAO_DR14().loglkl(theta, z_int, Hz, d_C)
    if np.isnan(lp) == True:
        return -np.inf
    return lp

ndim, nwalkers = 5, 100#25
# set the length of each walker
niter = 100000#20
# initialise the varying parameters with H_0 in km/s/Mpc
p0 = [[74., 0.3, 0., 0.81, -19] + [5., 5.e-3, 1.e-2, 0.1, 20] * np.random.randn(ndim) for i in range(nwalkers)]
# storing the chain in memory
#backend = emcee.backends.HDFBackend("Chains_h5_files/CC_SN_n0H0_chain.h5")
# clearing the file, in case it already exists
#backend.reset(nwalkers, ndim)

if __name__ == '__main__':

    with Pool(6) as pool:
    # with MPIPool() as pool:
    #     if not pool.is_master():
    #         pool.wait()
    #         sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool) #, backend=backend)

        # burn-in progress
        print('\nRunning burn-in...')
        p0, _, _ = sampler.run_mcmc(p0, 100, progress=True)
        sampler.reset()
        # production process
        print('Running production...')
        #sampler.run_mcmc(p0, niter, progress=True)

        index = 0
        autocorr = np.empty(niter)

        old_tau = np.inf

        for sample in sampler.sample(p0, iterations = niter, progress=True):
            # print(sampler.iteration)
            if sampler.iteration % 100:
                continue

            # computing the autocorrelation time
            tau = sampler.get_autocorr_time(tol=0)
            # print(tau)
            autocorr[index]=np.mean(tau)
            index += 1

            converged = np.all(tau*100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau)/ tau < 0.0001)
            print('\n --> old_tau = {0}, new_tau = {1} not convergent!!'.format(old_tau, tau))
            if converged:
                print('\n --> old_tau = {0}, new_tau = {1} convergent!!'.format(old_tau, tau))
                break
            old_tau = tau

        # get the stored chain of MCMC samples
        flat_samples = sampler.get_chain(flat=True)
        np.savetxt('fs8_chain.txt', flat_samples)
        print(flat_samples)
        print(flat_samples.shape)

        # get the log probabilities of the MCMC chain
        log_prob = sampler.get_log_prob(flat=True)
        np.savetxt('fs8_log_prob.txt', log_prob)
