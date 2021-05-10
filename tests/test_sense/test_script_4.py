import sympy
from scipy import constants
from normalorder.sense.model import Model
import numpy as np

Phi_0 = constants.physical_constants['mag. flux quantum'][0]

order = 3

n = 2
delta_sym, f_J_sym, phi_ext_sym, nu_sym, L_sym = sympy.symbols('delta f_J phi_ext nu L')
potential_param_symbols = {'f_J': f_J_sym, 'phi_ext': phi_ext_sym, 'nu': nu_sym, 'L': L_sym}
potential_expr = ((delta_sym*Phi_0)**2 / (2*L_sym*constants.h)) - f_J_sym * nu_sym * sympy.cos(2*sympy.pi*delta_sym) - f_J_sym * n * sympy.cos(2*sympy.pi*(delta_sym-phi_ext_sym)/n)

potential_expr = -7.008178211387194e-08 * delta_sym + 4816115753545.432*delta_sym**2 + 3135236312.8449826*delta_sym**3
potential_expr += f_J_sym*(-1.9298686356167531*delta_sym + 29.127674045528007*delta_sym**2 + 3.1352363128449823*delta_sym**3)




L = 6.740357146050338e-10
C_0 = 2.6435816869821043e-12

f_J = 0.0 # Hz
phi_ext = 0.1
Delta_param = 1e8
nu = 1.0
delta_0 = 0.00020157441417927727

potential_params_1 = {'f_J': f_J, 'phi_ext': phi_ext, 'nu': nu, 'L': L}
model_1 = Model(lumped_element=True)
model_1.set_order(order)
model_1.set_potential(potential_expr, potential_param_symbols)
model_1.set_resonator_params(C_0=C_0)
model_1.set_potential_params(potential_params_1, delta_min_guess=delta_0)
model_1.set_modes(names=['a'])
model_1.generate_hamiltonian(drive=False)
print(model_1.c_func(1),model_1.c_func(2),model_1.dc_func(1,'f_J'),model_1.dc_func(2,'f_J'))

potential_1 = model_1.generate_potential_hamiltonian(inplace=False, orders=list(range(order+1)), rwa=False)
substitutions = {'f_J': f_J+Delta_param}
potential_1_prime = model_1.generate_potential_hamiltonian(inplace=False, orders=list(range(order+1)), substitutions=substitutions, rwa=False)
Delta_potential = potential_1_prime - potential_1
ham_prime = model_1.hamiltonian + Delta_potential

potential_params_2 = {'f_J': f_J+Delta_param, 'phi_ext': phi_ext, 'nu': nu, 'L': L}
model_2 = Model(lumped_element=True)
model_2.set_order(order)
model_2.set_potential(potential_expr, potential_param_symbols)
model_2.set_resonator_params(C_0=C_0)
model_2.set_potential_params(potential_params_2, delta_min_guess=delta_0)
model_2.set_modes(names=['a'])
model_2.generate_hamiltonian(drive=False)
ham_2 = model_2.hamiltonian

Delta = ham_prime[(2, 0)]
omega_a = ham_prime[(1, 1)]
mu = ham_prime[(1, 0)]
r = 0.5*np.arctanh(2*Delta/omega_a)
omega_b = omega_a / np.cosh(2*r)
s = mu / (omega_b*(np.cosh(r)+np.sinh(r)))

print((omega_b-ham_2[(1,1)]))
print((omega_b-ham_2[(1,1)])/omega_b)

