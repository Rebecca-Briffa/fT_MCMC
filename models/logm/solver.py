import numpy as np
from scipy import optimize

# Newton-Raphson method to find roots for E
def solver_Ez(z, H_0, Om):
    def friedmann_E(E):
        h = H_0 / 100
        OR = 4.1534 * (10 ** (-5)) / h ** 2
        # term = (1- Om- OR) / (np.log10(p) + 2)
        return E **2 - (Om * (1 + z) ** 3 + OR * (1 + z) ** 4 + ( (1- Om -OR)*E))
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

        # try:
    #    root = optimize.newton(friedmann_E, 1000)
    # except RuntimeError:
    #      pass
     #            try:
     #                root = optimize.newton(friedmann_E, 10000)
     #            except RuntimeError:
     #                root = optimize.newton(friedmann_E, 100000)
    # print(root)
    return root

# finding H(z) using the equation E = H/H_0
def H_z(z, H_0, Om):
    Ez = solver_Ez(z, H_0, Om)
    if Ez == -1.e-60 :
        return -1.e-60
    else:
        sqrtHz = solver_Ez(z, H_0, Om) * H_0

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
    alpha = -(1-Om-OR) * np.sqrt(p)/2
    FT = alpha/(2*(1/p)) * (T/((1/p)*T_0))**(-1/2) * np.log10(p*T_0/T) + alpha*T_0*np.sqrt(T/(p*T_0))*(-1/T)
    FTT = -alpha/(4*p*p*T_0)* (T/(p*T_0))**(-3/2) * np.log10(p*T_0/T) + (alpha/2*p)*(T/(p*T_0))**(-1/2) *(-1/T)- (alpha/2)*np.sqrt(1/p)*np.sqrt(T_0/T)*T**(-1)

    dH = 1/2 *(1/(H*(1+z)*(1-FT-2*T*FTT))) * (3*H_0**2*(Om * (1 + z) ** 3 + OR * (1 + z) ** 4) + H_0**2*OR * (1 + z) ** 4)
    return dH
