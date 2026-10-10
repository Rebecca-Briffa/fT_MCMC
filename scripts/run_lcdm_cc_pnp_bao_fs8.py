import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import emcee
from multiprocessing import Pool, cpu_count
from models.lcdm.likelihoods import *
ncpu = cpu_count()

# LCDM theta: H_0, Om, s80, M (no extra p)

def log_prior(theta):
    H_0, Om, s80, M  = theta
    if 0.1 < Om < 0.9 and 50. < H_0 < 100. and -80 < M < 60 and 0.6 < s80 < 1.0:
        return 0.0
    else:
        return -np.inf

def log_probability(theta):
    H_0, Om, s80, M  = theta
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    n_interpolation = 100
    Hz = np.empty((n_interpolation))
    dHz = np.empty((n_interpolation))
    z_min = 0.0
    z_max = 2.4
    z_int = np.linspace(z_min, z_max, n_interpolation)
    for index, z in enumerate(z_int):
        H_cosmo = H_z(z, H_0, Om)
        Hz[index] = H_cosmo
        dHz[index] = solver_dHz(z, H_0, Om, H_cosmo)
    d_H = c / Hz
    integrand = interpolate.interp1d(z_int, d_H)
    d_C = []
    for j in range(len(z_int)):
        d_C.append(integrate.simps(integrand(z_int[:j + 1]), x=z_int[:j + 1]))
    d = diff_d_d0(z_int, H_0, Om)
    lp = lp + Likelihood_clocks().loglkl(theta, z_int, Hz) + Likelihood_Pantheon_SH0ES().loglkl(theta, z_int, Hz, d_C) \
         + Likelihood_fs8_BAO().loglkl(theta, z_int, Hz, d, d_C)
    if np.isnan(lp) == True:
        return -np.inf
    return lp

ndim, nwalkers = 4, 100#25
niter = 100000#20
p0 = [[74., 0.3, 0.81, -19] + [5., 5.e-3, 0.1, 20] * np.random.randn(ndim) for i in range(nwalkers)]

if __name__ == '__main__':
    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        print('\nRunning burn-in...')
        p0, _, _ = sampler.run_mcmc(p0, 100, progress=True)
        sampler.reset()
        print('Running production...')
        index = 0
        old_tau = np.inf
        for sample in sampler.sample(p0, iterations = niter, progress=True):
            if sampler.iteration % 100:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            converged = np.all(tau*100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau)/ tau < 0.0001)
            if converged:
                break
            old_tau = tau
        np.savetxt('lcdm_chain.txt', sampler.get_chain(flat=True))
        np.savetxt('lcdm_log_prob.txt', sampler.get_log_prob(flat=True))
