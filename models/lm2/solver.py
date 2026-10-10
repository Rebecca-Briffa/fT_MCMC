import numpy as np
from scipy import optimize


# Newton-Raphson method to find roots for E
def solver_Ez(z, H_0, Om, p):
    def friedmann_E(E):
        h = H_0 / 100
        Or = 4.1534 * (10 ** (-5)) / h ** 2
        E_func = (E**2)-(Om*((1+z)**3))-(Or*((1+z)**4))-((1-Om-Or)*(1-((1+(2*E**2/p))*(np.exp(-E**2/p))))/(1-((1+(2/p))*(np.exp(-1/p)))))
        return E_func
    # root = optimize.newton(friedmann_E, 1)
    try:
        root = optimize.newton(friedmann_E, 100000, tol=10 ** (-5), maxiter=10000)
    except RuntimeError:
        try:
           root = optimize.newton(friedmann_E, 100000000, tol=10 ** (-5), maxiter=10000)
        except RuntimeError:
            try:
                root = optimize.newton(friedmann_E, 100000000000000, tol=10 ** (-5), maxiter=10000)
            except RuntimeError:
                root = -1.e-60

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
    alpha = (1-Om-OR)/(-1 + ((2/p + 1)*np.exp(-1/p)))

    dH = 1/2 *(1/(H*(1+z)*(1+(1/p*alpha*np.exp((-1/p)*T/T_0) * (-1 + (2*(1/p)*T/T_0)))))) * (3*H_0**2*(Om * (1 + z) ** 3 + OR * (1 + z) ** 4) + H_0**2*OR * (1 + z) ** 4)
    return dH
