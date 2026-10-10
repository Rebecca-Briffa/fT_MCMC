import numpy as np


def H_z(z,H_0, Om):
    h = H_0 / 100
    OR = 4.1534 * (10 ** (-5)) / h ** 2
    sqrt_H = H_0**2*(Om*(1. + z)**3 + OR*(1. + z)**4 + (1. - Om - OR))
    # print(sqrt_H)
    # catch error here
    # catch error here
    if sqrt_H<0:
        return -1.e-60
    else:
        H = np.sqrt(sqrt_H)
        return H


#derivative of the model
# Newton-Raphson method to find roots for E
def solver_dHz(z, H_0, Om, H):
    # define h =H_0 / 100
    h = H_0 / 100
    # define omega_radiation as from Planck 18
    OR = 4.1534 * (10 ** (-5)) / h ** 2

    dH = H/(1+z) + (H_0**2/(2*(1+z)*H) * (2*(OR*(1+z)**4) + Om*(1+z)**3 - 2*(1-Om-OR)))
    return dH
