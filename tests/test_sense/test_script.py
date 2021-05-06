from normalorder.sense.model import Model
import sympy
import numpy as np
from scipy import constants
np.random.seed(seed=0)

Phi_0 = 2.07e-15

delta_sym, L_J_sym = sympy.symbols('delta L_J')
potential_param_symbols = {'L_J': L_J_sym}
potential_expr = (delta_sym*Phi_0)**2 / (2*L_J_sym*constants.h)
delta_min_guess = 0.0
order = 2

Z = 50  # ohms
v_p = 1.1817e8  # m/s
l = 0.007809801198766881  # m
L_0 = Z / v_p  # H/m
C_0 = 1 /(v_p*Z)  # F/m
x_J = 0.002626535356654626  # m
L_J_1 = 1e-11
Delta_L_J = 0.0005*L_J_1
L_J_2 = L_J_1 + Delta_L_J

model_1 = Model()
model_1.set_order(order)
model_1.set_potential(potential_expr, potential_param_symbols)
model_1.set_resonator_params(x_J=x_J, L_0=L_0, l=l, C_0=C_0)
potential_params = {'L_J': L_J_1}
model_1.set_potential_params(potential_params, delta_min_guess=delta_min_guess)
model_1.set_modes(['a'])

model_2 = Model()
model_2.set_order(order)
model_2.set_potential(potential_expr, potential_param_symbols)
model_2.set_resonator_params(x_J=x_J, L_0=L_0, l=l, C_0=C_0)
potential_params = {'L_J': L_J_2}
model_2.set_potential_params(potential_params, delta_min_guess=delta_min_guess)
model_2.set_modes(['a'])

print(model_2.modes['a'].frequency-model_1.modes['a'].frequency)

test1 = model_1.generate_potential_derivative_op('L_J')*Delta_L_J
test2 = model_1.dc_func(2,'L_J')*Delta_L_J*model_1.delta**2
print('done')