from sympy import symbols, diff, Order, collect
import sympy
from scipy import special, optimize
import numpy as np
import functools
from normalorder.operator.boson import Operator
import copy

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
        self.g_sym = ()
        self.resonator_params = {}
        self.resonator_syms = {}
        self.potential_expr = None
        self.potential_syms = {}
        self.potential_params = {}
        self.potential_param_substitutions = []
        self.annihilation_ops = ()
        self.c_expr = ()
        self.phi_min = None
        self.modes = None
        self.rotation_factors = None
        self.__g_expr = ()
        self.__max_order = 0
        self.order = 0
        self.hamiltonian = None
        self.potential_hamiltonian = None
        self.resonator_hamiltonian = None
        self.lindblad_ops = ()
        self.eom_expr = ()
        self.set_order(self.order)
        self.eom = None

    def set_order(self, order: int):
        if order < 0:
            raise Exception('The order must be non-negative integer.')
        c_sym, g_expr = g_expr_gen(order=max(self.__max_order, order))
        self.__max_order = max(self.__max_order, order)
        self.order = order
        self.c_sym = c_sym[:order+2]
        self.__g_expr = g_expr[:order+1]
        self.g_sym = symbols(' '.join(['g_' + str(i) for i in range(self.order + 1)]))

    def g_expr_gen(self, m):
        if m > self.order:
            raise Exception('Requested g expression is higher than the order of the model.')
        else:
            return self.__g_expr[m]

    def set_resonator_params(self, params):
        self.resonator_params = params
        self.resonator_syms = {}
        for key in params.keys():
            self.resonator_syms[key] = sympy.Symbol(key)

    def set_potential(self, potential_expr, potential_param_symbols):
        self.potential_syms = potential_param_symbols
        self.potential_expr = potential_expr
        self.c_expr = tuple(self.c_expr_gen(m) for m in range(self.order + 2))

    def set_potential_params(self, params):
        self.potential_params = params
        self.potential_param_substitutions = [(self.potential_syms[key], self.potential_params[key])
                                              for key in self.potential_syms.keys()]
        self.find_phi_min()

    def c_expr_gen(self, m):
        return diff(self.potential_expr, phi_sym, m) / (special.factorial(m))

    def c_func(self, m, phi):
        substitutions = self.potential_param_substitutions + [(phi_sym, phi)]
        c_value = np.float(self.c_expr[m].subs(substitutions).evalf())
        return c_value

    def potential_func(self, phi):
        substitutions = self.potential_param_substitutions + [(phi_sym, phi)]
        potential = np.float(self.potential_expr.subs(substitutions).evalf())
        return potential

    def g_func(self, m, phi):
        substitutions = [(self.c_sym[m], self.c_func(m, phi)) for m in range(self.order + 2)]
        return np.float(self.g_expr_gen(m).subs(substitutions))

    def find_phi_min(self, x0=None):

        def wrapper(x):
            return self.c_func(1, x[0])

        if x0 is None:
            x0 = 2 * np.pi * 3 * np.random.rand()
        res = optimize.root(wrapper, x0, tol=0)
        self.phi_min = res.x[0]

    def set_modes(self, modes=None, rotation_factors=None):
        if modes is None:
            self.modes = tuple('a')
        else:
            self.modes = modes
        if rotation_factors is None:
            self.rotation_factors = tuple(1 + np.arange(len(self.modes)))
        else:
            self.rotation_factors = rotation_factors
        annihilation_ops = []
        for idx in range(len(self.modes)):
            specification = [[1] + 2 * len(self.modes) * [0]]
            specification[0][2 * (idx + 1)] = 1
            annihilation_ops.append(Operator(specification, modes=self.modes))
        self.annihilation_ops = tuple(annihilation_ops)

    def generate_hamiltonian(self):
        hamiltonian = [0]
        for factor, mode, op in zip(self.rotation_factors, self.modes, self.annihilation_ops):
            hamiltonian += [2 * np.pi * (self.resonator_params['f_' + mode] - factor * self.resonator_params['f_d'])
                            * op.dag() * op]
        hamiltonian = sum(hamiltonian)
        hamiltonian += 2 * np.pi * self.resonator_params['epsilon'] * (self.annihilation_ops[0]
                                                                       + self.annihilation_ops[0].dag())

        U = 0.0
        mode_coeffs = tuple(self.resonator_params['I_ratio_'+mode] for mode in self.modes)
        for m in range(1, self.order + 1):
            U += 2 * np.pi * self.resonator_params['f_J'] * self.g_func(m, self.phi_min) \
                 * generate_x_pow(m, mode_coeffs=mode_coeffs)
        hamiltonian += U
        hamiltonian = apply_rwa(hamiltonian)
        self.hamiltonian = hamiltonian

    def generate_potential_hamiltonian(self):
        self.potential_hamiltonian = 0.0
        mode_coeffs = tuple(self.resonator_syms['I_ratio_'+mode] for mode in self.modes)

        for m in range(0, self.order + 1):
            x_pow = generate_x_pow(m, mode_coeffs=mode_coeffs)
            self.potential_hamiltonian += 2 * np.pi * self.resonator_syms['f_J'] * self.g_sym[m] * x_pow
        self.potential_hamiltonian = apply_rwa(self.potential_hamiltonian)

    def generate_resonator_hamiltonian(self):
        self.resonator_hamiltonian = 0.0

        for factor, mode, op in zip(self.rotation_factors, self.modes, self.annihilation_ops):
            self.resonator_hamiltonian += 2 * np.pi * (self.resonator_syms['f_' + mode]
                                                       - factor * self.resonator_syms['f_d']) * op.dag() * op
        self.resonator_hamiltonian += 2 * np.pi * self.resonator_syms['epsilon'] \
                                        * (self.annihilation_ops[0] + self.annihilation_ops[0].dag())

    def generate_hamiltonian(self):
        self.generate_potential_hamiltonian()
        self.generate_resonator_hamiltonian()
        self.hamiltonian = self.resonator_hamiltonian + self.potential_hamiltonian

    def generate_lindblad_ops(self):
        lindblad_ops = []
        for mode, op in zip(self.modes, self.annihilation_ops):
            decay_rate = sympy.sqrt(self.resonator_syms['kappa_'+mode])
            lindblad_ops.append(decay_rate*op)
        self.lindblad_ops = tuple(lindblad_ops)

    def generate_eom_expr(self):
        eom_expr = []
        for an_op in self.annihilation_ops:
            mode_eom_expr = 1j*(self.hamiltonian*an_op-an_op*self.hamiltonian)
            for l in self.lindblad_ops:
                mode_eom_expr += l.dag()*an_op*l - 0.5*an_op*l.dag()*l - 0.5*l.dag()*l*an_op
            eom_expr += [mode_eom_expr]
        self.eom_expr = tuple(eom_expr)

    def generate_eom_func(self):
        eom_functions = []
        g_substitutions = [(g_sym, g_expr) for g_sym, g_expr in zip(self.g_sym, self.__g_expr)]
        c_substitutions = [(c_sym, self.c_func(m, self.phi_min)) for m, c_sym in enumerate(self.c_sym)]
        resonator_substitutions = [(self.resonator_syms[key], self.resonator_params[key]) for key in self.resonator_syms.keys()]

        for eom_expr in self.eom_expr:
            eom_sub = copy.deepcopy(eom_expr)
            for idx in range(eom_sub.data.shape[0]):
                eom_sub.data['coeff'].iloc[idx] \
                    = eom_sub.data['coeff'].iloc[idx].subs(g_substitutions).subs(c_substitutions).subs(resonator_substitutions)
            eom_functions.append(classical_function_factory(eom_sub))

        def eom(fields):
            Dfields = np.array([eom_func(fields) for eom_func in eom_functions])
            return Dfields

        self.eom = eom


def convert_op_to_expr(op):
    field_syms = sympy.symbols(op.modes)
    field_array = []
    for sym in field_syms:
        field_array += [sympy.conjugate(sym), sym]
    field_array = np.array(field_array)
    field_names = []
    for mode in op.modes:
        field_names += [mode+'_dag', mode]

    expr = 0
    for idx in range(op.data.shape[0]):
        exponents = op.data[field_names].iloc[idx].values
        term = op.data['coeff'].iloc[idx]*np.product(field_array**exponents)
        expr += term

    return expr


@functools.lru_cache(maxsize=64)
def generate_x_pow(exponent: int, mode_coeffs=(1,)):
    n_modes = len(mode_coeffs)
    if exponent > 0:
        annihilation_ops = []
        for idx in range(n_modes):
            ledger = [[mode_coeffs[idx]] + [0, 0]*n_modes]
            ledger[0][2*(idx+1)] = 1
            annihilation_ops.append(Operator(ledger))
            x = sum(op+op.dag() for op in annihilation_ops)
            return x * generate_x_pow(exponent - 1, mode_coeffs=mode_coeffs)
    else:
        return 1

def classical_function_factory(operator):
    modes = operator.modes
    modes_dag = [mode+'_dag' for mode in modes]
    coeffs = np.copy(operator.data['coeff'].values)
    powers = np.copy(operator.data[modes].values)
    powers_star = np.copy(operator.data[modes_dag].values)
    def classical_approximation(fields):
        out = np.sum(coeffs * np.product(np.conjugate(fields)**powers_star * fields**powers, axis=1))
        return out
    return classical_approximation
