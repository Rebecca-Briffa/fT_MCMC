import numpy as np
from scipy import optimize


# Newton-Raphson method to find roots for E
def solver_Ez(z, H_0, Om, p):
    def friedmann_E(E):
        h = H_0 / 100
        OR = 4.1534 * (10 ** (-5)) / h ** 2

        term = - (1 - Om - OR)/ (1 - ((1+(1/p)) * np.exp(-1/p)))
        LM = term * ((( 1+(1/p)*E)*np.exp(-(1/p)*E))-1)
        return E ** 2 - (Om * (1 + z) ** 3 + OR * (1 + z) ** 4 + LM)
    # root = optimize.newton(friedmann_E, 1)
    try:
        root = optimize.newton(friedmann_E, 1)
    except RuntimeError:
        try:
           root = optimize.newton(friedmann_E, 100)
        except RuntimeError:
            try:
                root = optimize.newton(friedmann_E, 1000)
            except RuntimeError:
                try:
                    root = optimize.newton(friedmann_E, 10000)
                except RuntimeError:
                    try:
                        root = optimize.newton(friedmann_E, 100000)
                    except RuntimeError:
                        try:
                            root = optimize.newton(friedmann_E, 100000000)
                        except RuntimeError:
                            root = -1.e-60
                            print('RuntimeError', z, H_0, Om, p)
                        except RuntimeWarning:
                            root = -1.e-60
                            print('RuntimeWarning',z, H_0, Om, p)
    return root

# finding H(z) using the equation E = H/H_0
def H_z(z, H_0, Om,p):
    Ez = solver_Ez(z, H_0, Om, p)
    if Ez == -1.e-60 :
        return -1.e-60
    else:
        sqrtHz = solver_Ez(z, H_0, Om, p) * H_0

    # catch error here
    if sqrtHz < 0:
        return -1.e-60
    else:
       # model = math.sqrt(sqrtHz)
        return sqrtHz


# Newton-Raphson method to find roots for E
def solver_dHz(z, H_0, Om, p, H):
    # define h =H_0 / 100
    h = H_0 / 100
    # define omega_radiation as from Planck 18
    OR = 4.1534 * (10 ** (-5)) / h ** 2
    T = -6*H**2
    T_0 = -6 * H_0 ** 2
    beta = -(1/p)*np.sqrt(T/T_0)

    alpha = - (1 - Om - OR)/ (1 - ((1+(1/p)) * np.exp(-1/p)))

    dH = 1/2 * (1/ (H*(1+z)*(1+ (0.5*alpha*(1/p)*np.exp(beta))*(-np.sqrt(T_0/T) + ((1/p)/T) + np.sqrt(T_0/T**3))))) *  (3*H_0**2*(Om * (1 + z) ** 3 + OR * (1 + z) ** 4) + H_0**2*OR * (1 + z) ** 4)
    return dH
