import numpy as np
from scipy import optimize


# Newton-Raphson method to find roots for E
def solver_Ez(z, H_0, Om, p):
    def friedmann_E(E):
        # define h = H_0 / 100
        h = H_0 / 100
        # define omega_radiation as from Planck 18
        OR = 4.1534 * (10 ** (-5)) / h ** 2
        return E ** 2 - (Om * (1 + z) ** 3 + OR * (1 + z) ** 4 + ((1 - Om - OR) * E ** (2 * p)))
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
                        # if there is no root return -1e-60
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
    # define h = H_0 / 100
    h = H_0 / 100
    # define omega_radiation as from Planck 18
    OR = 4.1534 * (10 ** (-5)) / h ** 2
    T = -6*H**2

    alpha = (6*H_0**2)**(1-p)*(1-Om-OR)/(1-2*p)

    dH = 1/2 *(1/(((H*(1+z)))*(1+p*alpha*(-T)**(p-1) + 2*alpha *p*(p-1)*(-T)**(p-1)))) * (3*H_0**2*(Om * (1 + z) ** 3 + OR * (1 + z) ** 4) + H_0**2*OR * (1 + z) ** 4)
    return dH
