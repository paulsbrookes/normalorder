from normalorder.sense.model import Model
import sympy
import numpy as np
from scipy import constants
np.random.seed(seed=0)

Phi_0 = 2.07e-15

delta_sym, L_J_sym = sympy.symbols('delta L_J')
potential_param_symbols = {'L_J': L_J_sym}
potential_expr = - (delta_sym*Phi_0)**2 / (2*L_J_sym*constants.h)

order = 2

model = Model()
model.set_order(order)
model.set_potential(potential_expr, potential_param_symbols)

Z = 50  # ohms
v_p = 1.1817e8  # m/s
l = 0.007809801198766881  # m
L_0 = Z / v_p  # H/m
C_0 = 1 /(v_p*Z)  # F/m
x_J = 0.002626535356654626  # m
L_J = 1e-11

model.set_resonator_params(x_J=x_J, L_0=L_0, l=l, C_0=C_0)

delta_min_guess = None
potential_params = {'L_J': L_J}

model.set_potential_params(potential_params, delta_min_guess=delta_min_guess)

print(model.resonator_params['L_J'])