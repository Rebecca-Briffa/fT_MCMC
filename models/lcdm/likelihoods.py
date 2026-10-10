from models.lcdm.pantheon import *
from numpy import loadtxt
from models.lcdm.util import *
from models.lcdm.paths import data_file
import scipy.linalg as la


# LCDM theta: H_0, Om, s80, M  (no extra f(T) parameter p)
# Growth+BAO uses RSD_BAO_data.txt in file order with fs8_RSD_cov_tot.txt.
# That vector is not sorted by redshift.

datafile, covfile, zCMB, zHEL, m_obs, cepheid_distance, is_calibrator, cov = Pantheon_data()


class Likelihood_clocks:
    '''
    Cosmic chronometers likelihood
    '''
    def __init__(self):
                H_file = data_file('H_' + 'CC' + 'noH0' + '.dat')
                (z_H_dat, H_z_mean, H_z_err) = loadtxt(H_file, unpack='True')
                self.z = z_H_dat
                self.Hz = H_z_mean
                self.err = H_z_err

    def loglkl(self, theta, z_int, Hz):
        H_0, Om, s80, M  = theta
        chi2 = 0
        if len(Hz) != 100:
            return -np.inf
        else:
            Hz_interpolate = interpolate.interp1d(z_int, Hz)
            H_z = Hz_interpolate(self.z)
            for index, z in enumerate(self.z):
                chi2 += (self.Hz[index] - H_z[index]) ** 2 / self.err[index] ** 2
            return -0.5 * chi2


class Likelihood_Pantheon_SH0ES:
    def loglkl(self, theta, z_int, Hz, d_C):
        H_0, Om, s80, M  = theta
        red_size = zCMB.size
        moduli = zCMB * np.nan
        cosmo_dist = distances(zCMB, Hz, z_int, d_C)
        if cosmo_dist == -np.inf:
            chi2 = -np.inf
        else:
            DA = cosmo_dist[0]
            f = interpolate.interp1d(zCMB, DA)
            moduli[np.array(is_calibrator, dtype='bool')] = cepheid_distance[np.array(is_calibrator, dtype='bool')]
            zcmb = zCMB[~np.array(is_calibrator, dtype='bool')]
            zhel = zHEL[~np.array(is_calibrator, dtype='bool')]
            moduli[~np.array(is_calibrator, dtype='bool')] = 5.0 * np.log10((1.0 + zcmb) * (1.0 + zhel) * np.atleast_1d(f(zcmb))) + 25.
            residuals = m_obs - M
            residuals -= moduli
            residuals = la.solve_triangular(cov, residuals, lower=True, check_finite=False)
            chi2 = (residuals ** 2).sum()
        return -0.5 * chi2


class Likelihood_fs8_BAO:
    def __init__(self):
        self.z_eboss = np.array([], 'float64')
        self.fs8_eboss = np.array([], 'float64')
        self.dM_eboss = np.array([], 'float64')
        self.dH_eboss = np.array([], 'float64')
        with open(data_file('RSD_BAO_data.txt'), 'r') as filein:
            for line in filein:
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    if this_line[2] == 'f_sigma8':
                        self.z_eboss=(np.append(self.z_eboss, float(this_line[0])))
                        self.fs8_eboss = np.append(self.fs8_eboss, float(this_line[1]))
                        self.dM_eboss = np.append(self.dM_eboss, 0)
                        self.dH_eboss = np.append(self.dH_eboss, 0)
                    elif this_line[2] == 'DM_over_rs':
                        self.z_eboss = (np.append(self.z_eboss, float(this_line[0])))
                        self.dM_eboss = np.append(self.dM_eboss, float(this_line[1]))
                        self.fs8_eboss = np.append(self.fs8_eboss, 0)
                        self.dH_eboss = np.append(self.dH_eboss, 0)
                    elif this_line[2] == 'DH_over_rs':
                        self.z_eboss = (np.append(self.z_eboss, float(this_line[0])))
                        self.dH_eboss = np.append(self.dH_eboss, float(this_line[1]))
                        self.fs8_eboss = np.append(self.fs8_eboss, 0)
                        self.dM_eboss = np.append(self.dM_eboss, 0)
        covdata = np.loadtxt(data_file('fs8_RSD_cov_tot.txt')) * 10 ** (-3)
        self.cov_inv = np.linalg.inv(covdata)

    def loglkl(self, theta, z_int, Hz, d, d_C):
        H_0, Om, s80, M  = theta
        od = interpolate.interp1d(z_int, d)
        dD = od(self.z_eboss)
        fsigma8_theo = -(1 + self.z_eboss) * s80 * dD
        data_array = np.array([], 'float64')
        r_fid = 147.8
        cosmo_dist = distances(self.z_eboss, Hz, z_int, d_C)
        D_M = cosmo_dist[1]
        Hz_interpolate = interpolate.interp1d(z_int, Hz)
        H_cosmo = Hz_interpolate(self.z_eboss)
        DH_theo = c/ H_cosmo / r_fid
        theo_DM = D_M / r_fid
        if any(i < 10 ** (-3) for i in fsigma8_theo):
            return -np.inf
        for i in range(len(self.z_eboss)):
            if self.fs8_eboss[i] != 0:
                data_array = np.append(data_array, self.fs8_eboss[i] - fsigma8_theo[i])
            elif self.dM_eboss[i] != 0:
                data_array = np.append(data_array, theo_DM[i] - self.dM_eboss[i])
            elif self.dH_eboss[i] != 0:
                data_array = np.append(data_array, DH_theo[i] - self.dH_eboss[i])
        chi2 = np.dot(np.dot(data_array, self.cov_inv), data_array)
        return -0.5 * chi2
