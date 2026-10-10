from models.logm.pantheon import *
from numpy import loadtxt
from models.logm.util import *
from models.logm.paths import data_file
import scipy.linalg as la
import time
import pandas as pd



class Likelihood_clocks:
    '''
    Cosmic chronometers likelihood
    '''

    def __init__(self):
                # Text file for Cosmic Chronometer (CC) data
                H_file = data_file('H_' + 'CC' + 'noH0' + '.dat')

                # load the CC H(z) data
                (z_H_dat, H_z_mean, H_z_err) = loadtxt(H_file, unpack='True')

                # Store the columns separately
                self.z = z_H_dat
                self.Hz = H_z_mean
                self.err = H_z_err


    def loglkl(self, theta, z_int, Hz):
        # define theta as the free paraemters
        H_0, Om, p, s80, M  = theta
        # computing chi2 by first starting with chi2 = 0 and then adding to this chi2
        chi2 = 0

        # catch error here
        if len(Hz) != 100:
            return -np.inf
        else:
            # compute the interpolation at the chosen redshifts
            Hz_interpolate = interpolate.interp1d(z_int, Hz)
            #obtain the 31 theoretical points
            H_z = Hz_interpolate(self.z)

            for index, z in enumerate(self.z):
                chi2 += (self.Hz[index] - H_z[index]) ** 2 / self.err[index] ** 2

            return -0.5 * chi2

datafile, covfile, zCMB, zHEL, m_obs, cepheid_distance, is_calibrator, cov = Pantheon_data()

class Likelihood_Pantheon_SH0ES:
    '''
    Pantheon Supernovae dataset
    '''

    def loglkl(self, theta, z_int, Hz, d_C):
        H_0, Om, p, s80, M  = theta


        zcmb = zCMB
        zhel = zHEL
        red_size = zcmb.size
        moduli = zCMB * np.nan

        # obtain the luminosity distance
        cosmo_dist = distances(zcmb, Hz, z_int, d_C)


        # catch error here
        if cosmo_dist == -np.inf:
            chi2 = -np.inf

        else:
            DA = cosmo_dist[0]
            # compute the moduli to be able to compare with data TO CHECK ABOUT SELF.KIND
            f = interpolate.interp1d(zcmb, DA)
            # Here we use the Cepheid host distances as the "theory".
            # This is the 2nd rung of the distance ladder and is what calibrates M
            moduli[np.array(is_calibrator, dtype='bool')] = cepheid_distance[np.array(is_calibrator, dtype='bool')]

            # Actually do the interpolation at the data redshifts
            zcmb = zCMB[~np.array(is_calibrator, dtype='bool')]
            zhel = zHEL[~np.array(is_calibrator, dtype='bool')]
            moduli[~np.array(is_calibrator, dtype='bool')] = 5.0 * np.log10((1.0 + zcmb) * (1.0 + zhel) * np.atleast_1d(f(zcmb))) + 25.

            # Compute the residuals (estimate of distance moduli - exact moduli)
            residuals = np.empty((red_size,))
            mu = m_obs
            # This operation loops over all supernovae!
            # Compute the approximate moduli
            residuals = mu - M

            # Remove from the approximate moduli the one computed from CLASS
            residuals -= moduli



            # Step 2) (Step 1 is done in the init) Solve the triangular system, also time expensive (0.02 seconds)
            residuals = la.solve_triangular(cov, residuals, lower=True, check_finite=False)

            # Finally, compute the chi2 as the sum of the squared residuals
            chi2 = (residuals ** 2).sum()
        return -0.5 * chi2

class Likelihood_BAO_Alam:

    '''
    BOSS data (DR12 galaxy sample) - from Alam et.al - https://arxiv.org/abs/1607.03155
    '''

    def __init__(self):

        # define arrays for values of z and data points
        self.z_Alam = np.array([], 'float64')
        self.dM_Alam = np.array([], 'float64')
        self.H_Alam = np.array([], 'float64')


        # read redshifts and data points for D_M and H
        with open(data_file('BAO_consensus_results_dM_Hz.txt'), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    # load redshifts and D_M * (r_s / r_s_fid)^-1 in Mpc
                    if this_line[1] == 'dM(rsfid/rs)':
                        self.z_Alam = np.append(self.z_Alam, float(this_line[0]))
                        self.dM_Alam = np.append(self.dM_Alam, float(this_line[2]))
                    # load H(z) * (r_s / r_s_fid) in km s^-1 Mpc^-1
                    elif this_line[1] == 'Hz(rs/rsfid)':
                        self.H_Alam = np.append(
                            self.H_Alam, float(this_line[2]))

        # read covariance matrix
        self.covdata = np.loadtxt(data_file('BAO_consensus_covtot_dM_Hz.txt'))

        # computing chi2

    def loglkl(self, theta, z_int, Hz, d_C):
        H_0, Om, p, s80, M  = theta
        # load the value for r_drag

        r_s = r_drag(H_0, Om,p)
        # define r_fiducial
        r_fid = 147.78

        # Obtain D_M in Mpc
        DM = self.dM_Alam
        # Obtain H(z) in in km s^-1 Mpc^-1
        H_z = self.H_Alam

        # obtain the comoving angular diameter
        cosmo_dist = distances(self.z_Alam, Hz, z_int, d_C)
        D_M = cosmo_dist[1]

        # compute the interpolation at the chosen redshifts
        Hz_interpolate = interpolate.interp1d(z_int, Hz)

        # obtain the theoretical points
        H_cosmo = Hz_interpolate(self.z_Alam)
        H_theo = H_cosmo * (r_s / r_fid)

        # catch error here
        if cosmo_dist == -np.inf:
            chi2 = -np.inf

        else:
            # calculating the theoretical comoving angular distance diameter
            theo_DM = D_M * (r_fid / r_s)
            # print(theo_DM)

            data_array = np.array([], 'float64')
            for index, z in enumerate(self.z_Alam):

                # calculate difference between the sampled point and observations for D_M and H (to be used in chi2)
                DM_diff = theo_DM[index] - DM[index]
                H_diff = H_theo[index] - H_z[index]
                # appending to the data array in order s.t (DM, H, DM, H,...)
                data_array = np.append(data_array, DM_diff)
                data_array = np.append(data_array, H_diff)

            cov_inv = np.linalg.inv(self.covdata)
            chi2 = np.dot(np.dot(data_array, cov_inv), data_array)

        return -0.5 * chi2


class Likelihood_BAO_DR14:
    '''
    eBOSS data (DR14 quasars sample) - from Zhao et. al - https://arxiv.org/abs/1801.03043
    '''

    def __init__(self):

       # define arrays for values of z and data points
        self.z_DR14 = np.array([], 'float64')
        self.dA_DR14 = np.array([], 'float64')
        self.H_DR14 = np.array([], 'float64')


        # read redshifts and data points
        with open(data_file('BAO_DR14Q_results_dA_Hz.txt'), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    # load redshifts and D_A * (r_s / r_s_fid)^-1 in Mpc
                    if this_line[1] == 'dA(rsfid/rs)':
                        self.z_DR14= np.append(self.z_DR14, float(this_line[0]))
                        self.dA_DR14 = np.append(
                            self.dA_DR14, float(this_line[2]))
                    # load H(z) * (r_s / r_s_fid) in km s^-1 Mpc^-1
                    elif this_line[1] == 'Hz(rs/rsfid)':
                        self.H_DR14 = np.append(
                            self.H_DR14, float(this_line[2]))

        # read correlation matrix
        rho_BAO = np.loadtxt(data_file('BAO_DR14Q_covtot_dA_Hz.txt'))

        # 1 sigma of DA and H in that order
        H_DA_sigma = [284.93, 14.63, 159.67, 12.42, 96.59, 12.75, 146.46, 14.79]

        # compute the covariance matrix
        self.BAO_cov = [[0 for x in range(len(H_DA_sigma))] for y in range(len(H_DA_sigma))]
        for ii in range(0, len(H_DA_sigma)):
            for jj in range(0, len(H_DA_sigma)):
                self.BAO_cov[ii][jj] = H_DA_sigma[ii] * rho_BAO[ii][jj] * H_DA_sigma[jj]


 # computing chi2
    def loglkl(self,theta, z_int, Hz, d_C):
        H_0, Om, p, s80, M = theta
        # load the value for r_drag
        r_s = r_drag(H_0, Om,p)

        # define r_fiducial
        r_fid = 147.78

        # Obtain D_A in in Mpc from data
        DA = self.dA_DR14
        # Obtain H(z) in in km s^-1 Mpc^-1 from data
        H_z = self.H_DR14

        data_array = np.array([], 'float64')

        # obtain the angular diameter distance d_A
        cosmo_dist = distances(self.z_DR14, Hz, z_int, d_C)
        D_A = cosmo_dist[0]
        theo_DA = D_A * (r_fid / r_s)

        # compute the interpolation at the chosen redshifts
        Hz_interpolate = interpolate.interp1d(z_int, Hz)
        # obtain the theoretical points
        H_cosmo = Hz_interpolate(self.z_DR14)
        H_theo = H_cosmo * (r_s / r_fid)

        # catch error here
        if cosmo_dist == -np.inf:
            chi2 = -np.inf
        else:

            for index, z in enumerate(self.z_DR14):

                # calculate difference between the sampled point and observations for D_M and H (to be used in chi2)
                DA_diff = theo_DA[index] - DA[index]
                H_diff = H_theo[index] - H_z[index]

                # appending to the data array in order s.t (DA, H, DA, H,...)
                data_array = np.append(data_array, DA_diff)
                data_array = np.append(data_array, H_diff)

            # compute the inverse of the covariance matrix
            cov_inv = np.linalg.inv(self.BAO_cov)
            # compute chi^2
            chi2 = np.dot(np.dot(data_array, cov_inv),data_array)

        return -0.5*chi2

class Likelihood_bao_boss:
    '''
    6DF, Beutler et. al 2011 - https://academic.oup.com/mnras/article/416/4/3017/976636
    SDSS DR7 MGS, Ross et al. - 1409.3242v1
    SDSS III Lymann alpha, - from Bourbous et. al - https://arxiv.org/abs/1708.02225
    '''

    def __init__(self):

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'int')

        # read redshifts and data points
        with open(data_file('bao_2014.txt'), 'r') as filein:
            for line in filein:
                if line.strip() and line.find('#') == -1:
                    # the first entry of the line is the identifier
                    this_line = line.split()

                    if this_line[0] == '6DF' or 'MGS' or 'Lya':
                        self.z = np.append(self.z, float(this_line[1]))
                        self.data = np.append(self.data, float(this_line[2]))
                        self.error = np.append(self.error, float(this_line[3]))
                        self.type = np.append(self.type, int(this_line[4]))

        # number of data points
        self.num_points = np.shape(self.z)[0]


    # compute likelihood

    def loglkl(self,theta, z_int, Hz, d_C):
        H_0, Om, p, s80, M  = theta
        chi2 = 0.

        if -1e-60 in Hz:
            print('WARNING')
            return -np.inf
        else:
            # compute the interpolation at the chosen redshifts
            Hz_interpolate = interpolate.interp1d(z_int, Hz)
            # obtain the theoretical points
            H_cosmo = Hz_interpolate(self.z)
            # for each point, compute angular distance da,
            # volume distance dv, sound horizon at baryon drag rs_d,
            # theoretical prediction and chi2 contribution
            for i in range(self.num_points):
                t5 = time.time()
                dv = v_weight_eff(self.z[i], Hz, z_int, d_C)
                t6 = time.time()
                # print('vol eff time', t6-t5)
                c_g = 299792458.  # speed of light in m/s
                c = c_g / 1000.  # speed of light in km/s
                dH = c / H_cosmo[i]
                rs = r_drag(H_0, Om,p)

                if self.type[i] == 1:
                    theo = dv / rs

                elif self.type[i] == 2:
                    theo = rs / dv

                elif self.type[i] == 3:
                    theo = dH / rs

                chi2 += ((theo - self.data[i]) / self.error[i]) ** 2

            return -0.5 * chi2

class Likelihood_BAO_eboss:

    '''
    BOSS data (DR12 galaxy sample) - from Alam et.al - https://arxiv.org/abs/1607.03155
    '''

    def __init__(self):

        # define arrays for values of z and data points
        self.z_eboss = np.array([], 'float64')
        self.dM_eboss = np.array([], 'float64')
        self.H_eboss = np.array([], 'float64')


        # read redshifts and data points for D_M and H
        with open(data_file('BAO-only_consensus_results_eboss_dM_dH.txt'), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    # load redshifts and D_M * (r_s / r_s_fid)^-1 in Mpc
                    if this_line[2] == 'DM_over_rs':
                        self.z_eboss = np.append(self.z_eboss, float(this_line[0]))
                        self.dM_eboss = np.append(self.dM_eboss, float(this_line[1]))
                    # load H(z) * (r_s / r_s_fid) in km s^-1 Mpc^-1
                    elif this_line[2] == 'DH_over_rs':
                        self.H_eboss = np.append(
                            self.H_eboss, float(this_line[1]))

        # read covariance matrix
        self.covdata = np.loadtxt(data_file('BAO-only_covtot_eboss_dM_dH.txt'))
        # print(self.H_eboss)
        # computing chi2

    def loglkl(self, theta, z_int, Hz, d_C):
        H_0, Om, p, s80, M  = theta
        # load the value for r_drag

        r_s = r_drag(H_0, Om,p)
        # define r_fiducial
        r_fid = 147.8

        # Obtain D_M in Mpc
        DM = self.dM_eboss
        # Obtain H(z) in in km s^-1 Mpc^-1
        DH = self.H_eboss

        # obtain the comoving angular diameter
        cosmo_dist = distances(self.z_eboss, Hz, z_int, d_C)
        D_M = cosmo_dist[1]
        # print(self.z_eboss, D_M)
        # compute the interpolation at the chosen redshifts
        Hz_interpolate = interpolate.interp1d(z_int, Hz)

        # obtain the theoretical points
        H_cosmo = Hz_interpolate(self.z_eboss)
        DH_theo = c/ H_cosmo
        DH_theo = DH_theo / r_fid


        # catch error here
        if cosmo_dist == -np.inf:
            chi2 = -np.inf

        else:
            # calculating the theoretical comoving angular distance diameter
            theo_DM = D_M / r_fid
            # print(theo_DM)

            data_array = np.array([], 'float64')
            for index, z in enumerate(self.z_eboss):

                # calculate difference between the sampled point and observations for D_M and H (to be used in chi2)
                DM_diff = theo_DM[index] - DM[index]

                DH_diff = DH_theo[index] - DH[index]
                # print(DH_theo[index], DH[index])
                # print('dH diff', DH_diff)
                # appending to the data array in order s.t (DM, H, DM, H,...)
                data_array = np.append(data_array, DM_diff)
                data_array = np.append(data_array, DH_diff)
            # print('dataarray', data_array)
            cov_inv = np.linalg.inv(self.covdata)
            chi2 = np.dot(np.dot(data_array, cov_inv), data_array)

        return -0.5 * chi2


# fs8_cov_tot.txt follows f_sigma8_updated.csv row by row.
# WiggleZ (z = 0.44, 0.60, 0.73) is one correlated block in that order.
# Do not sort by z: after sort_values, series[0] is a label lookup, not the first row.
fs8_file = data_file('f_sigma8_updated.csv')
_fs8_table = pd.read_csv(fs8_file)
z_fs8 = _fs8_table['z'].to_numpy()
fs8_data = _fs8_table['fs8'].to_numpy()
err_fs8 = _fs8_table['err_fs8'].to_numpy()

covdata = np.loadtxt(data_file('fs8_cov_tot.txt'))*10**(-3)
cov_inv = np.linalg.inv(covdata)



# Likelihood for emcee
class Likelihood_fs8:
    '''
    Cosmic chronometers likelihood
    '''

    def __init__(self):

                self.z = z_fs8
                self.fs8 = fs8_data
                self.err = err_fs8


    def loglkl(self, theta, z_int,d, Hz):
        # define theta as the free parameters
        H_0, Om, p, s80, M  = theta

        # computing chi2 by first starting with chi2 = 0 and then adding to this chi2
        chi2 = 0
        if -1e-60 in Hz:
            print('WARNING')
            return -np.inf
        else:

            od = interpolate.interp1d(z_int, d)
            dD = np.asarray(od(self.z), dtype=float)

            fsigma8_theo = -(1 + self.z) * s80 * dD
            # * AP_ratio

            data_array = np.array([], 'float64')

            if any(i < 10**(-3) for i in fsigma8_theo):
                chi2 =  -np.inf
                return chi2
            else:
                for index, z in enumerate(self.z):
                    # calculate difference between the sampled point and observations for D_M and H (to be used in chi2)
                    fs8_diff = self.fs8[index] - fsigma8_theo[index]
                    # appending to the data array in order s.t (DM, H, DM, H,...)
                    data_array = np.append(data_array, fs8_diff)

                chi2 = np.dot(np.dot(data_array, cov_inv), data_array)

            return -0.5 * chi2

class Likelihood_fs8_BAO:
    '''
    Cosmic chronometers likelihood
    '''

    def __init__(self):

# define arrays for values of z and data points
        self.z_eboss = np.array([], 'float64')
        self.z_fs8_eboss = np.array([], 'float64')
        self.z_DM_eboss = np.array([], 'float64')
        self.z_DH_eboss = np.array([], 'float64')
        self.fs8_eboss = np.array([], 'float64')
        self.dM_eboss = np.array([], 'float64')
        self.dH_eboss = np.array([], 'float64')


        # read redshifts and data points for D_M and H
        with open(data_file('RSD_BAO_data.txt'), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    if this_line[2] == 'f_sigma8':
                        self.z_eboss=(np.append(self.z_eboss, float(this_line[0])))
                        self.z_fs8_eboss = np.append(self.z_fs8_eboss, float(this_line[0]))
                        self.fs8_eboss = np.append(self.fs8_eboss, float(this_line[1]))
                        self.dM_eboss = np.append(self.dM_eboss, 0)
                        self.dH_eboss = np.append(self.dH_eboss, 0)
                    elif this_line[2] == 'DM_over_rs':
                        self.z_eboss = (np.append(self.z_eboss, float(this_line[0])))
                        self.z_DM_eboss = np.append(self.z_DM_eboss, float(this_line[0]))
                        self.dM_eboss = np.append(self.dM_eboss, float(this_line[1]))
                        self.fs8_eboss = np.append(self.fs8_eboss, 0)
                        self.dH_eboss = np.append(self.dH_eboss, 0)
                    # load H(z) * (r_s / r_s_fid) in km s^-1 Mpc^-1
                    elif this_line[2] == 'DH_over_rs':
                        self.z_eboss = (np.append(self.z_eboss, float(this_line[0])))
                        self.z_DH_eboss = np.append(self.z_DH_eboss, float(this_line[0]))
                        self.dH_eboss = np.append(self.dH_eboss, float(this_line[1]))
                        self.fs8_eboss = np.append(self.fs8_eboss, 0)
                        self.dM_eboss = np.append(self.dM_eboss, 0)

        covdata = np.loadtxt(data_file('fs8_RSD_cov_tot.txt')) * 10 ** (-3)
        self.cov_inv = np.linalg.inv(covdata)
                # print(self.fs8_eboss)


    def loglkl(self, theta, z_int,d, Hz, d_C):
        # define theta as the free parameters
        H_0, Om, p, s80, M = theta

        # computing chi2 by first starting with chi2 = 0 and then adding to this chi2
        chi2 = 0
        od = interpolate.interp1d(z_int, d)
        dD = od(self.z_eboss)
        # print(dD)

        fsigma8_theo = -(1 + self.z_eboss) * s80 * dD
        # print(fsigma8_theo)
        data_array = np.array([], 'float64')

        r_fid = 147.8
        # obtain the comoving angular diameter
        cosmo_dist = distances(self.z_eboss, Hz, z_int, d_C)
        D_M = cosmo_dist[1]
        # print(self.z_eboss, D_M)
        cosmo_dist1 = distances(0.38, Hz, z_int, d_C)
        # print('dm',cosmo_dist1[1])


        # compute the interpolation at the chosen redshifts
        Hz_interpolate = interpolate.interp1d(z_int, Hz)

        # obtain the theoretical points
        H_cosmo = Hz_interpolate(self.z_eboss)
        DH_theo = c/ H_cosmo
        DH_theo = DH_theo / r_fid
        # calculating the theoretical comoving angular distance diameter
        theo_DM = D_M / r_fid

        if any(i < 10 ** (-3) for i in fsigma8_theo):
            chi2 = -np.inf
            return chi2
        else:
            for i in range(len(self.z_eboss)):
                # print(i)
                if self.fs8_eboss[i] != 0:
                    # print(i)
                    fs8_diff = self.fs8_eboss[i] - fsigma8_theo[i]
                    # print(self.fs8_eboss[i], fsigma8_theo[i], fs8_diff)
                # appending to the data array in order s.t (fs8, fs8,..., DM,DH,fs8,.....)
                    data_array = np.append(data_array, fs8_diff)
                elif self.dM_eboss[i] != 0:
                    DM = self.dM_eboss
                    # print(DM)
                    DM_diff = theo_DM[i] - DM[i]
                    # print(theo_DM[i], DM[i])
                    # print(DM_diff)
                    data_array = np.append(data_array, DM_diff)
                elif self.dH_eboss[i] != 0:
                    DH = self.dH_eboss
                    # print(DH)
                    DH_diff = DH_theo[i] - DH[i]
                    # print(DH_theo[i], DH[i])
                    # print(DH_diff)
                    data_array = np.append(data_array, DH_diff)
            # print(len(data_array))


            chi2 = np.dot(np.dot(data_array, self.cov_inv), data_array)

            return -0.5 * chi2

# ''' s_{8,0} prior'''
#
# def likelihood_s80(theta):
#     H_0, Om, p  = theta
#
#     # computing chi2 by first starting with chi2 = 0 and then adding to this chi2
#     chi2 = 0
#     # Plancks sigma_8,0 value
#
#     s80_P = 0.8111
#     s80_P_err = 0.0060
#
#     chi2 += (s80 - s80_P) ** 2 / s80_P_err ** 2
#
#     return -0.5*chi2
