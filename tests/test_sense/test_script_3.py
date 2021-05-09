import sympy
from scipy import constants
from normalorder.sense.model import Model
import numpy as np

Phi_0 = constants.physical_constants['mag. flux quantum'][0]

order = 2

n = 2
delta_sym, delta_0_sym, L_sym = sympy.symbols('delta delta_0 L')
potential_param_symbols = {'delta_0': delta_0_sym, 'L': L_sym}
potential_expr = ((delta_sym*Phi_0)**2 / (2*L_sym*constants.h)) + 1.5*delta_0_sym*(((delta_sym-delta_0_sym)*Phi_0)**2 / (2*L_sym*constants.h))


L = 6.740357146050338e-10
C_0 = 2.6435816869821043e-12

Delta = 0.1
delta_0 = 0.1

potential_params_1 = {'delta_0': delta_0, 'L': L}
model_1 = Model(lumped_element=True)
model_1.set_order(order)
model_1.set_potential(potential_expr, potential_param_symbols)
model_1.set_resonator_params(C_0=C_0)
model_1.set_potential_params(potential_params_1, delta_min_guess=delta_0)
print(model_1.delta_0)
model_1.set_modes(names=['a'])
model_1.generate_hamiltonian(drive=False)

potential_1 = model_1.generate_potential_hamiltonian(inplace=False, orders=[0,1,2], rwa=False)
substitutions = {'delta_0': delta_0+Delta}
potential_1_prime = model_1.generate_potential_hamiltonian(inplace=False, orders=[0,1,2], substitutions=substitutions, rwa=False)
Delta_potential = potential_1_prime - potential_1
ham_prime = model_1.hamiltonian + Delta_potential

potential_params_2 = {'delta_0': delta_0+Delta, 'L': L}
model_2 = Model(lumped_element=True)
model_2.set_order(order)
model_2.set_potential(potential_expr, potential_param_symbols)
model_2.set_resonator_params(C_0=C_0)
model_2.set_potential_params(potential_params_2, delta_min_guess=delta_0)
print(model_2.delta_0)
model_2.set_modes(names=['a'])
model_2.generate_hamiltonian(drive=False)
ham_2 = model_2.hamiltonian

Delta = ham_prime[(2,0)]
omega_a = ham_prime[(1,1)]
mu = ham_prime[(1,0)]
r = 0.5*np.arctanh(2*Delta/omega_a)
omega_b = omega_a / np.cosh(2*r)
s = mu / (omega_b*(np.cosh(r)+np.sinh(r)))

print((omega_b - ham_2[(1,1)]))


