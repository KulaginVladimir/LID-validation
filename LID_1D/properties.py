from festim import k_B
import numpy as np

rho_W = 19250.0 # Tungsten density

def thermal_cond_function_W(T):
    # Temperature dependence of the W thermal condutivity in W/m/K
    if not isinstance(T, float):
        if any(T.vector().get_local() == 0):
            return 149.441-45.466e-3*T+13.193e-6*T**2-1.484e-9*T**3
    else:
        return 149.441-45.466e-3*T+13.193e-6*T**2-1.484e-9*T**3+3.866e6/(T+1.0)**2

def heat_capacity_function_W(T):
    # Temperature dependence of the W volumetric heat capacity in J/kg
    if not isinstance(T, float):
        if any(T.vector().get_local() == 0):
            return (21.868372+8.068661e-3*T-3.756196e-6*T**2+1.075862e-9*T**3) / 183.84e-3  
    else:
        return (21.868372+8.068661e-3*T-3.756196e-6*T**2+1.075862e-9*T**3+1.406637e4/(T+1.0)**2) / 183.84e-3  

def heat_of_transport_function_W(T):
    # Temperature dependence of the Soret coefficient in eV
    return -0.0045 * k_B * T**2

def polynomial(coeffs, x, main):
    val = coeffs[0]
    for i in range(1, 4):
        if main:
            val += coeffs[i] * np.float_power(x, i)
        else:
            val += coeffs[i] * x**i
    return val

def rhoCp_Cu(T, main=False):
    coeffs = [3.45899e6, 4.67353e2, 6.14079e-2, 1.68402e-4]
    return polynomial(coeffs, T, main=main)


def thermal_cond_Cu(T, main=False):
    coeffs = [4.02301e02, -7.88669e-02, 3.76147e-05, -3.93153e-08]
    return polynomial(coeffs, T, main=main)