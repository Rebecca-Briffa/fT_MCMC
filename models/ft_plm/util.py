from scipy.integrate.odepack import ODEintWarning

from models.ft_plm.solver import *
import numpy as np
from scipy import integrate
from pandas import read_table
import time
from scipy import interpolate
from scipy.integrate import odeint
import warnings
# Obtain the  comoving sound horizon at the baryon drag epoch
def r_drag(H_0, Om,p):

    # defining the redshift at the drag epoch - z_drag- Planck18 Table 2
    z_drag = 1059.94
    # defining the upper limit of the integral - the scale factor at the drag epoch - a_drag
    a_drag = 1 / (1+z_drag)

   # the speed of light in km/s^2
    c = 2.99792458e8
    c_km = c / 1000

    # defining a function for the integrand
    def integrand(a):
        # define z with the integration argument a
        z = (1/a) - 1
        # call the Friedmann equation from the solver to obtain H(z)
        Hz = H_z(z,H_0, Om,p)

        # define h = H0/100
        h = H_0 /100

        # values obtained from Planck 18
        Ob_h2 = 0.02237
        Ob = Ob_h2 / h**2

        Oy_h2 = 2.4697*10**(-5)
        Oy = Oy_h2 / h**2

        R = (3/4) * (Ob/Oy)*a
        c_s = 1/(np.sqrt(3*(1 + R)))

        return (c_s / (a**2 * Hz))

    # solving the integration using integration.quad from 0 up to a_drag defined previously
    integration = integrate.quad(integrand, 0, a_drag)
    return integration[0]*c_km

# the speed of light in km/s^2
c_g = 299792458.  # speed of light in m/s
c = c_g / 1000.  # speed of light in km/s


# calculates the luminosity distance D_L = D_C * (1+z)
def distances(z, Hz, z_int, d_C):
    if -1e-60 in Hz:
        print('WARNING')
        return -np.inf
    else:

        dC_interpolation = interpolate.interp1d(z_int, d_C)
        # obtaining the function for d_C and then obtaining values at each z
        t12 = time.time()
        Dc = dC_interpolation(z)

        d_L = Dc * (1 + z)

        # calculates angular diameter distance d_A
        d_A = Dc / (1 + z)

        # calculates the comoving angular diameter D_M = D_A * (1+z)
        d_M = d_A * (1+z)
        t13 = time.time()
        # print('distances', t13 - t12, H_0, dH_0, d2H_0, Om, q)
        return  d_A, d_M,  d_L

#
def v_weight_eff(z, Hz, z_int, d_C):
    # compute the interpolation at the chosen redshifts
    Hz_interpolate = interpolate.interp1d(z_int, Hz)
    # obtain the 31 theoretical points
    H_z = Hz_interpolate(z)
    cosmo_dist = distances(z, Hz, z_int, d_C)
    if cosmo_dist == -np.inf:
        return -np.inf
    else:
        d_M = cosmo_dist[1]
        d_V = pow(d_M * d_M * c * z / H_z, 1. / 3.)
        return d_V

# read the matrix for the Supernoave data
def read_matrix(path):
    with open(path, 'r') as text:
        length = int(text.readline())
    matrix = read_table(path).to_numpy().reshape((length, length))
    return matrix


def read_light_curve_parameters(path):

    with open(path, 'r') as text:
        clean_first_line = text.readline()[1:].strip()
        names = [e.strip().replace('3rd', 'third')
                 for e in clean_first_line.split( )]

    lc_parameters = read_table(
        path, sep=' ', names=names, header=0, index_col=False)
    return lc_parameters




# First ode for D = d(z)/d_0 (d=delta)
def D_diff(y, z_int, H_0,Om,p):
    # compute the interpolation at the chosen redshifts
    # Hz_interpolate = interpolate.interp1d(z_int, Hz)
    # dHz_interpolate = interpolate.interp1d(z_int, dHz)

    D, g = y

    hz = H_z(z_int, H_0, Om,p)

    dhz = solver_dHz(z_int, H_0, Om,p, hz)


    part1 = -(dhz/hz - (1/(1+z_int)))
    h = H_0 / 100
    OR = 4.1534 * (10 ** (-5)) / h ** 2
    T = -6 * hz ** 2

    alpha = (6 * H_0 ** 2) ** (1 - p) * (1 - Om - OR) / (1- 2*p)

    Q = (1/ (1 + alpha * p * (-T)**(p-1)) )

    part2 =3/2*Q*(H_0/hz)**2*Om*(1+z_int)

    dydt = [g, part1*g + part2*D]
    return dydt

def diff_d_d0(z_int, H_0, Om,p):
    y0 = [1, -0.5]

    sol = odeint(D_diff, y0, z_int, args=(H_0, Om, p))
    dD = sol[:, 1]
    return dD
