from sympy import symbols, diff, Order, collect
from scipy import special, optimize
import numpy as np
import functools
from normalorder.operator.boson import Operator

phi_sym = symbols('phi')

def apply_rwa(operator, mode_frequencies=None):
    if mode_frequencies is None:
        mode_frequencies = {}
        for idx, mode in enumerate(operator.modes):
            mode_frequencies[mode] = idx + 1
    net_frequencies = 0
    for mode, frequency in mode_frequencies.items():
        net_frequencies += frequency * (operator.data[mode] - operator.data[mode + '_dag'])
    mask = np.isclose(net_frequencies, 0.0)
    operator_rwa = Operator(operator.data.iloc[mask], modes=operator.modes)
    return operator_rwa

def solve_sum(coeffs, target):
    if len(coeffs) == 0:
        if target == 0:
            return np.zeros([1, 0], dtype=int)
        else:
            return np.zeros([0, 0], dtype=int)

    limit = target // coeffs[0]
    solutions_list = []
    for x in range(0, limit + 1):
        sub_target = target - coeffs[0] * x
        sub_solutions = solve_sum(coeffs[1:], sub_target)
        solutions = np.hstack([x * np.ones([sub_solutions.shape[0], 1], dtype=int), sub_solutions])
        solutions_list.append(solutions)
    combined_solutions = np.vstack(solutions_list)
    return combined_solutions


def inverse_coeff_expr_gen(n, coeffs):
    if len(coeffs) <= n:
        raise Exception('Too few coefficient symbols to define A_n.')
    constraint_coeffs = tuple(range(1, n))
    vectors = solve_sum(constraint_coeffs, n - 1)
    A_n = 0
    for v in vectors:
        term = 1
        for idx, power in enumerate(v):
            term *= coeffs[idx + 2] ** power / special.factorial(power)
        term *= (-1) ** (sum(v))
        term *= special.factorial(n + sum(v) - 1) / (n * special.factorial(n - 1) * coeffs[1] ** (sum(v) + 1 - n))
        A_n += term
    A_n /= (coeffs[1] ** (2 * n - 1))
    return A_n


@functools.lru_cache(maxsize=32)
def g_expr_gen(order=None, potential_coeffs=None, x_sym=None):

    delta_sym = symbols('delta')

    if potential_coeffs is None:
        c_sym = symbols(' '.join(['c_' + str(i) for i in range(order + 2)]))
    else:
        c_sym = potential_coeffs
        if len(c_sym) < order + 2:
            raise Exception('Insufficient coefficients to define all the requested g expressions.')

    if x_sym is None:
        x_sym = symbols('DeltaI_r')

    U_series_DeltaI_r_expr = 0
    for m in range(order + 1):
        U_series_DeltaI_r_expr += c_sym[m] * gen_delta_pow_series(m, order, c_sym, x_sym)
    U_series_DeltaI_r_expr = collect(U_series_DeltaI_r_expr.expand(), x_sym)

    g_expr = [U_series_DeltaI_r_expr.removeO().subs(x_sym, 0)]
    for m in range(1, order + 1):
        g_expr.append(U_series_DeltaI_r_expr.coeff(x_sym ** m))

    return c_sym, g_expr


@functools.lru_cache(maxsize=32)
def gen_delta_pow_series(exponent, order, potential_coeffs, x_sym):
    delta_sym = symbols('delta')

    if not isinstance(exponent, int) or exponent < 0:
        raise Exception('The exponent must be an integer greater than or equal to zero.')

    if exponent == 0:
        return 1
    elif exponent == 1:
        c_sym = potential_coeffs
        U_series_expr = sum([c_sym[m] * delta_sym ** m for m in range(order + 2)]) + Order(delta_sym ** (order + 2))
        DeltaI_r_series_expr = diff(U_series_expr, delta_sym)
        coeffs = [DeltaI_r_series_expr.removeO().coeff(delta_sym ** m) for m in range(order + 1)]

        delta_series_expr = 0
        for m in range(1, order + 1):
            delta_series_expr += inverse_coeff_expr_gen(m, coeffs) * (x_sym) ** m
        delta_series_expr += Order(x_sym) ** (order + 1)
        return delta_series_expr
    else:
        delta_pow_series = gen_delta_pow_series(1, order, potential_coeffs, x_sym) * gen_delta_pow_series(exponent - 1,
                                                                                                          order,
                                                                                                          potential_coeffs,
                                                                                                          x_sym)
        delta_pow_series = collect((delta_pow_series + Order(x_sym) ** (order + 1)).expand(), x_sym)
        return delta_pow_series


class Model:

    def __init__(self):
        self.c_sym = ()
        self.resonator_params = {}
        self.potential_expr = None
        self.potential_param_symbols = {}
        self.potential_params = {}
        self.c_expr = ()
        self.phi_min = None
        self.__g_expr = ()
        self.__max_order = 0
        self.order = 0
        self.set_order(self.order)

    def set_order(self, order: int):
        if order < 0:
            raise Exception('The order must be non-negative integer.')
        c_sym, g_expr = g_expr_gen(order=max(self.__max_order, order))
        self.__max_order = max(self.__max_order, order)
        self.order = order
        self.c_sym = c_sym[:order+2]
        self.__g_expr = g_expr[:order+1]

    def g_expr_gen(self, m):
        if m > self.order:
            raise Exception('Requested g expression is higher than the order of the model.')
        else:
            return self.__g_expr[m]

    def set_resonator_params(self, params):
        self.resonator_params = params

    def set_potential(self, potential_expr, potential_param_symbols):
        self.potential_param_symbols = potential_param_symbols
        self.potential_expr = potential_expr
        self.c_expr = tuple(self.c_expr_gen(m) for m in range(self.order + 2))

    def set_potential_params(self, params):
        self.potential_params = params

    def c_expr_gen(self, m):
        return diff(self.potential_expr, phi_sym, m) / (special.factorial(m))

    def c_func(self, m, phi):
        substitutions = [(self.potential_param_symbols[key], self.potential_params[key]) for key in self.potential_param_symbols.keys()]
        substitutions += [(phi_sym, phi)]
        c_value = np.float(self.c_expr[m].subs(substitutions).evalf())
        return c_value

    def U_func(self, phi, phi_ext, nu):
        if not isinstance(phi, float):
            phi = phi[0]
        U = np.float(self.potential.subs([(phi_ext_sym, phi_ext), (phi_sym, phi), (nu_sym, nu)]).evalf())
        return U

    def g_func(self, m, phi):
        substitutions = [(self.c_sym[m], self.c_func(m, phi)) for m in range(self.order + 2)]
        return np.float(self.g_expr_gen(m).subs(substitutions))

    def find_phi_min(self, x0=None):

        def wrapper(x):
            return self.c_func(1, x[0])

        if x0 is None:
            x0 = 2 * np.pi * 3 * np.random.rand(1)
        res = optimize.root(wrapper, x0, tol=0)
        self.phi_min = res.x[0]

    def generate_hamiltonian(self, modes=['a'], rotation_factors=None):
        if rotation_factors is None:
            rotation_factors = 1 + np.arange(len(modes))
        annihilation_ops = []
        for idx in range(len(modes)):
            specification = [[1] + 2 * len(modes) * [0]]
            specification[0][2 * (idx + 1)] = 1
            annihilation_ops.append(Operator(specification, modes=modes))
        self.annihilation_ops = annihilation_ops
        self.modes = modes
        self.rotation_factors = rotation_factors
        hamiltonian = [0]
        for factor, mode, op in zip(rotation_factors, modes, annihilation_ops):
            hamiltonian += [2 * np.pi * (self.resonator_params['f_' + mode] - factor * self.resonator_params['f_d']) * op.dag() * op]
        hamiltonian = sum(hamiltonian)
        hamiltonian += 2 * np.pi * self.resonator_params['epsilon'] * (annihilation_ops[0] + annihilation_ops[0].dag())

        x = 0
        for mode, op in zip(modes, annihilation_ops):
            x += params['I_ratio_' + mode] * (op + op.dag())
        self.x = x
        x_pow = 1
        U = 2 * np.pi * self.resonator_params['f_J'] * self.g_func(0, self.phi_min, self.resonator_params['phi_ext'], self.resonator_params['nu'])
        for m in range(1, self.order + 1):
            x_pow *= x
            U += 2 * np.pi * self.resonator_params['f_J'] * self.g_func(m, self.phi_min, self.resonator_params['phi_ext'],
                                                                        self.resonator_params['nu']) * x_pow
        hamiltonian += U
        hamiltonian = apply_rwa(hamiltonian)
        self.hamiltonian = hamiltonian

    def generate_eom(self):
        eom_operators = []
        for mode, op in zip(self.modes, self.annihilation_ops):
            eom_op = -1j * (op * self.hamiltonian - self.hamiltonian * op)
            eom_op -= 0.5 * self.resonator_params['kappa_' + mode] * op
            eom_operators.append(eom_op)
        eom_functions = [classical_function_factory(eom_op) for eom_op in eom_operators]
        self.eom_functions = eom_functions

        def eom(fields):
            Dfields = np.array([eom_func(fields) for eom_func in eom_functions])
            return Dfields

        self.eom = eom


