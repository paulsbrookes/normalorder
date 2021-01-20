from sympy import symbols, diff, Order, collect
import sympy
from scipy import special, optimize, constants
import numpy as np
import functools
from normalorder.operator.boson import Operator
from .mode import Mode
import copy

delta_sym = symbols('delta')

Phi_0 = 2.07e-15


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
        self.c_expr = ()
        self.delta_min = None
        self.__g_expr = ()
        self.__max_order = 0
        self.order = 0
        self.hamiltonian = None
        self.potential_hamiltonian = None
        self.resonator_hamiltonian = None
        self.lindblad_ops = {}
        self.eom_ops = {}
        self.eom_exprs = {}
        self.set_order(self.order)
        self.eom = None
        self.drive_params = {}
        self.drive_syms = {}
        self.param_substitutions = []
        self.g_substitutions = []
        self.mode_names = ()
        self.modes = {}
        self.mode_frequencies = {}
        self.mode_ops = {}
        self.mode_numbers = {}
        self.decay_rates = {}
        self.decay_rate_syms = {}
        self.delta = 0

    def set_order(self, order: int):
        """
        Set the order of the Taylor expansion which approximates the potential close to its minimum.

        Parameters
        ----------
        order : int
        The Taylor expansion will be carried out up to and including the supplied order.

        Returns
        -------
        None
        """
        if order < 0:
            raise Exception('The order must be non-negative integer.')
        c_sym, g_expr = g_expr_gen(order=max(self.__max_order, order))
        self.__max_order = max(self.__max_order, order)
        self.order = order
        self.c_sym = c_sym[:order+2]
        self.__g_expr = g_expr[:order+1]
        self.g_sym = symbols(' '.join(['g_' + str(i) for i in range(self.order + 1)]))

    def generate_g_expr(self, m: int):
        """
        Generate a coefficient, g_m, of the Taylor expansion of the potential as a function of the current.

        Parameters
        ----------
        m : int
        The function will generate the mth coefficient.

        Returns
        -------
        g_m : sympy.Add
        The mth g coefficient of the Taylor expansion.
        """
        if m > self.order:
            raise Exception('Requested g expression is higher than the order of the model.')
        else:
            return self.__g_expr[m]

    def set_resonator_params(self, l: float=0.5, L_0: float=1.0, C_0: float=1.0, x_J: float=0.0):
        """
        Set the parameters of the resonator.

        Parameters
        ----------
        l : float
        Half he length of the resonator.

        L_0 : float
        The inductance per unit length of the resonator.

        C_0 : float
        The capacitance per unit length of the resonator.

        x_J : The position of the inductive circuit in the resonator.

        Returns
        -------
        None
        """
        self.resonator_params['l'] = l
        self.resonator_params['L_0'] = L_0
        self.resonator_params['C_0'] = C_0
        self.resonator_params['x_J'] = x_J

    def set_modes(self, names: list=['a'], indices=None, initial_wavevectors=None):

        if indices is None:
            indices = np.arange(1, len(names)+1)
        if initial_wavevectors is None:
            initial_wavevectors = indices*np.pi/(2*self.resonator_params['l'])
        self.modes = dict()
        velocity = 1/np.sqrt(self.resonator_params['C_0']*self.resonator_params['L_0'])

        self.mode_numbers = {name: idx for name, idx in zip(names, indices)}
        self.mode_names = names
        self.mode_ops = {}
        self.delta = 0.0
        self.decay_rates = {}
        self.decay_rate_syms = {}
        C_total = 2 * self.resonator_params['l'] * self.resonator_params['C_0']

        for name, index, k_init in zip(names, indices, initial_wavevectors):
            mode = Mode()
            mode.set_params(self.resonator_params)
            mode.solve(k_init)
            self.modes[name] = mode
            mode_frequency = mode.frequency
            self.mode_frequencies[name] = mode_frequency

            specification = [[1] + 2 * len(names) * [0]]
            specification[0][2*self.mode_numbers[name]] = 1
            op = Operator(specification, modes=self.mode_names)
            self.mode_ops[name] = op
            #if not np.isclose(mode.Delta, 0.0, atol=1e-12):
                #C_prime = C_total/mode.Delta**2
            self.delta += mode.Delta * np.sqrt(constants.hbar/(4*np.pi*mode_frequency*C_total))*(op+op.dag())*(1/Phi_0)

            self.decay_rate_syms[name] = sympy.symbols('kappa_'+name)
            self.decay_rates[name] = None

        wavevectors = np.array([mode.k for mode in self.modes.values()])
        if len(set(np.round(wavevectors,10))) < len(names):
            raise Exception('Some of the calculated wavevectors are not unique. '
                            'Try different initial guesses to identify unique modes.')

    def set_decay_rates(self, decay_rates):
        self.decay_rates = decay_rates

    def set_drive_params(self, f_d: float, epsilon: complex):
        self.drive_params['f_d'] = f_d
        self.drive_params['epsilon'] = epsilon
        self.drive_syms = {'f_d': sympy.Symbol('f_d'),
                           'epsilon': sympy.Symbol('epsilon')}

    def set_potential(self, potential_expr: sympy.Add, potential_param_syms: dict):
        """
        Supply a symbolic expression constructed with sympy to specify the potential of the inductive element. Also
        supply a list of symbols used in this expression.

        Parameters
        ----------
        potential_expr : sympy.Add
        The potential expression constructed with sympy.

        potential_param_syms : dict
        An iterable of symbols used to construct the potential expression.

        Returns
        -------
        None
        """
        self.potential_syms = potential_param_syms
        self.potential_expr = potential_expr
        self.c_expr = tuple(self.generate_c_expr(m) for m in range(self.order + 2))

    def set_potential_params(self, params: dict, delta_min_guess: float=None):
        """
        Supply the values of the parameters which specify the form of the potential and find the minimum of that
        potential.

        Parameters
        ----------
        params : dict
        A dictionary of potential parameter values.

        Returns
        -------
        None
        """
        self.potential_params = params
        self.potential_param_substitutions = [(self.potential_syms[key], self.potential_params[key])
                                              for key in self.potential_syms.keys()]
        self.find_delta_min(delta_min_guess)

        substitutions = [(delta_sym, self.delta_min)]
        for name in self.potential_syms.keys():
            pair = (self.potential_syms[name], self.potential_params[name])
            substitutions.append(pair)

        c_2_at_min = self.c_expr[2].subs(substitutions)
        L_J = Phi_0**2/(4*np.pi*constants.hbar*c_2_at_min)
        L_J = 2*np.pi*np.float(L_J.evalf())
        self.resonator_params['L_J'] = L_J

    def generate_c_expr(self, m: int):
        """
        Generate a sympy expression describing a coefficient of the Taylor series of the potential in terms of the phase
        difference across the inductive element by calculating derivatives of the potential. This coefficient will be
        given as a function of the potential parameters and the phase difference about which Taylor expansion is carried
        out.

        Parameters
        ----------
        m : int
        The index of the coefficient to be generated.

        Returns
        -------
        c_m : sympy.Add
        A sympy expression describing the mth coefficient of the Taylor expansion.
        """
        return diff(self.potential_expr, delta_sym, m) / (special.factorial(m))

    def c_func(self, m: int, delta: float):
        """
        Calculate the numerical value of a coefficient of the Taylor expansion of the potential in terms of the phase
        difference across the inductive element.

        Parameters
        ----------
        m : int
        The index of the coefficient to be calculated.

        phi : float
        The value of the phase difference over the inductive element about which the Taylor expansion is calculated.

        Returns
        -------
        c_m : float
        The value of the specified coefficient at the given phase difference.
        """
        substitutions = self.potential_param_substitutions + [(delta_sym, delta)]
        c_value = np.float(self.c_expr[m].subs(substitutions).evalf())
        return c_value

    def potential_func(self, phi: float):
        """
        Calculate the value of the potential for a given phase difference over the inductive element.

        Parameters
        ----------
        phi : float
        The phase difference at which the value of the potential is to be calculated.

        Returns
        -------
        None
        """
        substitutions = self.potential_param_substitutions + [(delta_sym, phi)]
        potential = np.float(self.potential_expr.subs(substitutions).evalf())
        return potential

    def find_delta_min(self, delta_min_guess: float=None):
        """
        Find the value of the phase difference over the inductive element at which the potential is minimized.

        Parameters
        ----------
        delta_min_guess : float, optional
        Initial guess for the minimum of the potential.

        Returns
        -------
        None
        """
        def wrapper(x):
            return self.c_func(1, x[0])

        if delta_min_guess is None:
            delta_min_guess = 2 * np.pi * 3 * np.random.rand()
        elif isinstance(delta_min_guess, float):
            delta_min_guess = np.array([delta_min_guess])
        res = optimize.root(wrapper, delta_min_guess, tol=0)
        self.delta_min = res.x[0]

    def generate_potential_hamiltonian(self):
        """
        Generate the Hamiltonian describing the potential, i.e the inductive element, in terms of ladder operators of
        the resonator modes.

        Returns
        -------
        None
        """
        self.potential_hamiltonian = 0.0*self.mode_ops['a']

        for m in range(3, self.order + 1):
            self.potential_hamiltonian += self.c_func(m, self.delta_min) * generate_op_pow(m, self.delta)
        self.potential_hamiltonian *= (1/constants.hbar)
        self.potential_hamiltonian = apply_rwa(self.potential_hamiltonian, mode_frequencies=self.mode_numbers)

    def generate_resonator_hamiltonian(self):
        """
        Generate the Hamiltonian describing the resonator in terms of ladder operators of the resonator modes.

        Returns
        -------
        None
        """
        self.resonator_hamiltonian = 2j*sympy.pi*self.drive_syms['epsilon']*self.mode_ops[self.mode_names[0]].dag()
        self.resonator_hamiltonian += self.resonator_hamiltonian.dag()

        for name in self.mode_names:
            self.resonator_hamiltonian += 2*sympy.pi*(self.mode_frequencies[name] -
                                                      self.mode_numbers[name] * self.drive_syms['f_d']) \
                                          * self.mode_ops[name].dag() * self.mode_ops[name]

    def generate_hamiltonian(self):
        """
        Generate the Hamiltonian of the resonator and the inductive element in terms of ladder operators of the the
        resonator modes.

        Returns
        -------
        None
        """
        self.generate_potential_hamiltonian()
        self.generate_resonator_hamiltonian()
        self.hamiltonian = self.resonator_hamiltonian + self.potential_hamiltonian

    def generate_lindblad_ops(self):
        """
        Generate the Lindblad operators which describe dissipation of energy in the resonator modes.

        Returns
        -------
        None
        """
        self.lindblad_ops = {}
        for name, op in self.mode_ops.items():
            self.lindblad_ops[name] = sympy.sqrt(2*sympy.pi*self.decay_rate_syms[name])*op

    def generate_eom_ops(self):
        """
        Generate the operators which represent the Heisenberg equations of motion of the annihilation operators of the
        resonator modes.

        Returns
        -------
        None
        """
        self.eom_ops = {}
        for name, mode_op in self.mode_ops.items():
            eom_op = 1j * (self.hamiltonian * mode_op - mode_op * self.hamiltonian)
            for mode_name, l_op in self.lindblad_ops.items():
                eom_op += l_op.dag() * mode_op * l_op - 0.5 * mode_op * l_op.dag() * l_op \
                                 - 0.5 * l_op.dag() * l_op * mode_op
            self.eom_ops[name] = eom_op

    def generate_eom_exprs(self):
        """
        Convert the Heisenberg equations of motion to sympy expressions describing the equations of motion of the field
        amplitudes.

        Returns
        -------
        None
        """
        self.eom_exprs = {}
        for mode_name, eom_op in self.eom_ops.items():
            eom_expr = convert_op_to_expr(eom_op)
            self.eom_exprs[mode_name] = eom_expr

    def generate_param_substitutions(self):
        substitutions = []
        for name in self.mode_names:
            substitutions.append((self.decay_rate_syms[name], self.decay_rates[name]))
            #substitutions += package_substitutions(self.resonator_syms[mode_name], self.resonator_params[mode_name])
        substitutions.append((self.drive_syms['f_d'], self.drive_params['f_d']))
        substitutions.append((self.drive_syms['epsilon'], self.drive_params['epsilon']))
        #substitutions += package_substitutions(self.potential_syms, self.potential_params)
        self.param_substitutions = substitutions

    def generate_eom(self, potential_variables: list=[]):
        """
        Convert the sympy expressions describing the equations of motion of the fields into equations of motion which
        are suitable for numerical integration.

        Returns
        -------
        None
        """
        eom_funcs = {}
        self.generate_param_substitutions()
        combined_substitutions = self.param_substitutions + [(delta_sym, self.delta_min)]
        field_syms = [sympy.Symbol(mode_name) for mode_name in self.mode_names]
        arg_syms = [sympy.Symbol('delta_'+sym_name) for sym_name in potential_variables]
        combined_syms = field_syms + arg_syms

        for mode_name, eom_expr in self.eom_exprs.items():
            combined_eom_expr = eom_expr
            for arg_sym, sym_name in zip(arg_syms, potential_variables):
                combined_eom_expr += arg_sym * diff(eom_expr, self.potential_syms[sym_name]).subs(
                    self.potential_param_substitutions+[(delta_sym, self.delta_min)])
            for pair in combined_substitutions:
                combined_eom_expr = combined_eom_expr.replace(*pair)
            eom_func = sympy.lambdify(combined_syms, combined_eom_expr)
            eom_funcs[mode_name] = eom_func

        if len(potential_variables) == 0:
            def eom(fields):
                Dfields = np.array([eom_funcs[mode_name](*fields) for mode_name in self.mode_names])
                return Dfields
        else:
            def eom(fields, *args):
                Dfields = np.array([eom_funcs[mode_name](*fields, *args) for mode_name in self.mode_names])
                return Dfields
        self.eom = eom


def package_substitutions(syms, params):
    substitutions = []
    for sym_name, sym in syms.items():
        substitutions.append((sym, params[sym_name]))
    return substitutions


def convert_op_to_expr(op, return_syms=False):
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

    if return_syms:
        return expr, field_syms
    else:
        return expr


@functools.lru_cache(maxsize=64)
def generate_op_pow(exponent: int, operator):
    if exponent > 0:
        return copy.deepcopy(operator * generate_op_pow(exponent - 1, operator))
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
