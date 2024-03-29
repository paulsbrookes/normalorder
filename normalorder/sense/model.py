from sympy import symbols, diff, Order, collect
import sympy
from scipy import special, optimize, constants
import numpy as np
import functools
from normalorder.operator.boson import Operator
from normalorder.sense.mode import Mode
import matplotlib.pyplot as plt
import string
from sortedcontainers import SortedDict

delta_sym = symbols('delta')


def apply_rwa(operator, mode_frequencies=None):
    if mode_frequencies is None:
        mode_frequencies = list(range(1,operator.n_modes+1))
    mode_frequencies = np.array(mode_frequencies)
    rotation_frequencies = (mode_frequencies[:, np.newaxis] * np.array([-1, 1])[np.newaxis, :]).flatten()
    net_frequencies = list(map(lambda key: sum(np.array(key)*rotation_frequencies), operator.keys()))
    operator_rwa = Operator(n_modes=operator.n_modes)
    for key, frequency, coeff in zip(operator.keys(), net_frequencies, operator.values()):
        if np.isclose(frequency, 0.0):
            operator_rwa[key] = coeff
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

    def __init__(self, lumped_element=False):
        self.c_sym = ()
        self.resonator_params = {}
        self.resonator_syms = {}
        self.potential_expr = None
        self.potential_syms = {}
        self.potential_params = {}
        self.potential_param_substitutions = []
        self.c_expr = ()
        self.delta_0 = None
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
        self.mode_names = ()
        self.modes = {}
        self.mode_frequencies = {}
        self.mode_ops = {}
        self.mode_numbers = {}
        self.decay_rates = {}
        self.decay_rate_syms = {}
        self.delta = None
        self.drive_syms = {'f_d': sympy.Symbol('f_d'),
                           'epsilon': sympy.Symbol('epsilon')}
        self.Phi_0 = constants.physical_constants['mag. flux quantum'][0]
        self.lumped_element = lumped_element
        self.combined_eom_exprs = {}
        self.res = None
        self.mode_indices = {}
        self.steady_state_res = None
        self.mode_syms = None
        self.dalpha_func = None

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

    def set_resonator_params(self, l: float=None, L_0: float=None, C_0: float=None, x_J: float=None):
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

        if self.lumped_element:
            if l is not None or x_J is not None or L_0 is not None:
                raise Exception(
                    'Only C_0 should be specified for a lumped element resonator. '
                    'The inductance should be included in the potential')
            self.resonator_params['C_0'] = C_0
        else:
            self.resonator_params['l'] = l
            self.resonator_params['L_0'] = L_0
            self.resonator_params['C_0'] = C_0
            self.resonator_params['x_J'] = x_J

    def set_modes(self, names: list = ['a'], harmonic_numbers=None, initial_wavevectors=None):

        if not self.lumped_element:
            if harmonic_numbers is None:
                harmonic_numbers = np.arange(1, len(names) + 1)
            if initial_wavevectors is None:
                initial_wavevectors = harmonic_numbers * np.pi / (2 * self.resonator_params['l'])
            self.modes = dict()

            self.harmonic_numbers = {name: idx for name, idx in zip(names, harmonic_numbers)}
            self.mode_indices = {name: idx for idx, name in enumerate(names)}
            self.harmonic_numbers_array = harmonic_numbers
            self.mode_names = names
            self.mode_ops = {}
            self.delta = 0.0
            self.decay_rates = {}
            self.decay_rate_syms = {}
            C_total = 2 * self.resonator_params['l'] * self.resonator_params['C_0']

            for name, k_init in zip(names, initial_wavevectors):
                mode = Mode()
                mode.set_params(self.resonator_params)
                mode.solve(k_init)
                self.modes[name] = mode
                mode_frequency = mode.frequency
                self.mode_frequencies[name] = mode_frequency

                specification = [[1] + 2 * len(names) * [0]]
                specification[0][2 * (self.mode_indices[name]+1)] = 1
                op = Operator(spec=specification)
                self.mode_ops[name] = op
                self.delta += mode.Delta * np.sqrt(constants.hbar / (4 * np.pi * mode_frequency * C_total)) * (
                        op + op.dag()) * (1 / self.Phi_0)

                self.decay_rate_syms[name] = sympy.symbols('kappa_' + name)
                self.decay_rates[name] = None

            wavevectors = np.array([mode.k for mode in self.modes.values()])
            if len(set(np.round(wavevectors, 10))) < len(names):
                raise Exception('Some of the calculated wavevectors are not unique. '
                                'Try different initial guesses to identify unique modes.')

        else:
            self.modes = dict()
            self.harmonic_numbers = {names[0]: 1}
            self.harmonic_numbers_array = np.array([1])
            self.mode_names = names
            self.mode_ops = {}
            self.decay_rates = {}
            self.decay_rate_syms = {}

            self.L_prime = self.resonator_params['L_J']
            frequency = 1 / (np.sqrt(self.L_prime * self.resonator_params['C_0']) * 2 * np.pi)
            self.mode_frequencies[names[0]] = frequency

            specification = [[1] + 2 * len(names) * [0]]
            specification[0][2] = 1
            op = Operator(spec=specification)
            self.mode_ops[names[0]] = op

            self.delta = np.sqrt(constants.hbar / (4 * np.pi * frequency * self.resonator_params['C_0'])) * (
                    op + op.dag()) * (1 / self.Phi_0)

        self.mode_syms = {mode_name: sympy.Symbol(mode_name) for mode_name in self.mode_names}

    def set_decay_rates(self, decay_rates):
        self.decay_rates = decay_rates

    def set_drive_params(self, f_d: float, epsilon: complex):
        self.drive_params['f_d'] = f_d
        self.drive_params['epsilon'] = epsilon

    def set_potential(self, potential_expr: sympy.Add, potential_param_syms: dict):
        """
        Supply a symbolic expression constructed with sympy to specify the potential of the embedded inductive element.
        Also supply a list of symbols used in this expression. The potential should be given in natural frequency units
        i.e. occurrences per second. The potential must be given in terms of the symbol symbols('delta') which is in
        units the flux quantum model.Phi_0.

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

    def set_potential_params(self, params: dict):
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

        self.potential_params.update(params)
        self.potential_param_substitutions = [(self.potential_syms[key], self.potential_params[key])
                                              for key in self.potential_params.keys()]

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

    def c_func(self, m: int, delta=None, substitutions={}):
        """
        Calculate the numerical value of a coefficient of the Taylor expansion of the potential in terms of the phase
        difference across the inductive element.

        Parameters
        ----------
        m : int
        The index of the coefficient to be calculated.

        delta : float
        The value of the phase difference over the inductive element about which the Taylor expansion is calculated.

        Returns
        -------
        c_m : float
        The value of the specified coefficient at the given phase difference.
        """

        if delta is None:
            delta = self.delta_0

        potential_param_substitutions = []
        for param_name, param_value in substitutions.items():
            potential_param_substitutions.append((self.potential_syms[param_name], param_value))
        for param_name, param_sym in self.potential_syms.items():
            if param_name not in substitutions.keys():
                potential_param_substitutions.append((param_sym, self.potential_params[param_name]))

        potential_param_substitutions += [(delta_sym, delta)]
        c_value = np.float(self.c_expr[m].subs(potential_param_substitutions).evalf())
        return c_value

    def dc_func(self, m: int, potential_variable_name):

        dc_expr = diff(self.c_expr[m], self.potential_syms[potential_variable_name])
        dc = np.float(dc_expr.subs(self.potential_param_substitutions + [(delta_sym, self.delta_0)]))
        return dc

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

    def find_delta_min_with_c4_constraint(self, x_initial, potential_variable):
        def wrapper(x):
            c_1 = self.c_func(1, delta=x[0])
            c_4 = self.c_func(4, delta=x[0], substitutions={potential_variable: x[1]})
            return np.array([c_1, c_4])

        res = optimize.root(wrapper, x_initial)
        self.res = res
        self.delta_0 = res.x[0]
        self.potential_params[potential_variable] = self.res.x[1]

    def find_delta_min(self, delta_min_guess: float=None, kwargs={}, use_minimizer=True):
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

        if delta_min_guess is None:
            delta_min_guess = 2 * np.pi * 3 * np.random.rand()
        elif isinstance(delta_min_guess, float):
            delta_min_guess = np.array([delta_min_guess])

        if use_minimizer:
            def wrapper_minimize(x):
                return self.potential_func(x[0])
            self.res_minimize = optimize.minimize(wrapper_minimize, delta_min_guess)
            delta_min_guess = self.res_minimize.x[0]

        def wrapper(x):
            return self.c_func(1, delta=x[0])

        if not 'tol' in kwargs.keys():
            kwargs['tol'] = 0.0

        res = optimize.root(wrapper, delta_min_guess, **kwargs)
        self.res = res
        self.delta_0 = res.x[0]
        if self.c_func(2) < 0:
            raise Exception('The second derivative is less than zero. This indicates you may have found a maximum, '
                            'rather than a minimum. Please use an alternative value for delta_min_guess.')

    def set_delta_0(self, delta_0):
        self.delta_0 = delta_0

    def calculate_L_J(self):
        if self.delta_0 is None:
            self.find_delta_min()

        substitutions = [(delta_sym, self.delta_0)]
        for name in self.potential_syms.keys():
            pair = (self.potential_syms[name], self.potential_params[name])
            substitutions.append(pair)

        c_2_at_min = self.c_expr[2].subs(substitutions)
        L_J = self.Phi_0**2/(2*c_2_at_min*constants.h)
        L_J = np.float(L_J.evalf())
        self.resonator_params['L_J'] = L_J

    def generate_potential_hamiltonian(self, rwa=True, substitutions={}, orders=None, inplace=True):
        """
        Generate the Hamiltonian describing the potential, i.e the inductive element, in terms of ladder operators of
        the resonator modes.

        Returns
        -------
        None
        """
        potential_hamiltonian = 0.0*self.mode_ops['a']
        if orders is None:
            orders = [1] + list(range(3, self.order+1))
        for m in orders:
            potential_hamiltonian += self.c_func(m, substitutions=substitutions) * self.delta**m
        if rwa:
            potential_hamiltonian = apply_rwa(potential_hamiltonian, mode_frequencies=self.harmonic_numbers_array)
        if inplace:
            self.potential_hamiltonian = potential_hamiltonian
        else:
            return potential_hamiltonian

    def generate_potential_derivative_op(self, potential_variable_name, rwa=True):
        potential_derivative_op = 0
        for m in range(1, self.order+1):
            potential_derivative_op += self.dc_func(m, potential_variable_name)*self.delta**m
        if rwa:
            potential_derivative_op = apply_rwa(potential_derivative_op, mode_frequencies=self.harmonic_numbers_array)
        return potential_derivative_op

    def generate_potential_derivative_eom_expr(self, potential_variable_name, mode_name):
        potential_derivative_op = self.generate_potential_derivative_op(potential_variable_name)
        mode_op = self.mode_ops[mode_name]
        eom_op = 1j * (potential_derivative_op * mode_op - mode_op * potential_derivative_op)
        eom_expr = 2*sympy.pi*convert_op_to_expr(eom_op, mode_names=self.mode_names)
        return eom_expr

    def generate_resonator_hamiltonian(self, drive=True):
        """
        Generate the Hamiltonian describing the resonator in terms of ladder operators of the resonator modes.

        Returns
        -------
        None
        """
        self.resonator_hamiltonian = 0.0
        for name in self.mode_names:
            self.resonator_hamiltonian += self.mode_frequencies[name] * self.mode_ops[name].dag() * self.mode_ops[name]

        if drive:
            drive_hamiltonian = 1j*self.drive_syms['epsilon']*self.mode_ops[self.mode_names[0]].dag()
            drive_hamiltonian += drive_hamiltonian.dag()
            self.resonator_hamiltonian += drive_hamiltonian
            for name in self.mode_names:
                self.resonator_hamiltonian -= self.harmonic_numbers[name] * self.drive_syms['f_d'] \
                                              * self.mode_ops[name].dag() * self.mode_ops[name]

    def generate_hamiltonian(self, drive=True, potential_variables=[]):
        """
        Generate the Hamiltonian of the resonator and the inductive element in terms of ladder operators of the the
        resonator modes.

        Returns
        -------
        None
        """
        self.generate_potential_hamiltonian(rwa=drive)
        self.generate_resonator_hamiltonian(drive=drive)

        self.potential_variable_syms = SortedDict({sym_name: sympy.Symbol('delta_' + sym_name) for sym_name in potential_variables})
        self.potential_derivative = 0
        for name, sym in self.potential_variable_syms.items():
            self.potential_derivative += sym*self.generate_potential_derivative_op(name)

        self.hamiltonian = self.resonator_hamiltonian + self.potential_hamiltonian + self.potential_derivative

    def generate_lindblad_ops(self):
        """
        Generate the Lindblad operators which describe dissipation of energy in the resonator modes.

        Returns
        -------
        None
        """
        self.lindblad_ops = {}
        for name, op in self.mode_ops.items():
            self.lindblad_ops[name] = sympy.sqrt(self.decay_rate_syms[name])*op

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
            self.eom_ops[name] = 2*sympy.pi*eom_op

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
            eom_expr = convert_op_to_expr(eom_op, mode_names=self.mode_names)
            self.eom_exprs[mode_name] = eom_expr

    def generate_param_substitutions(self):
        substitutions = []
        for name in self.mode_names:
            substitutions.append((self.decay_rate_syms[name], self.decay_rates[name]))
        substitutions.append((self.drive_syms['f_d'], self.drive_params['f_d']))
        substitutions.append((self.drive_syms['epsilon'], self.drive_params['epsilon']))
        self.param_substitutions = substitutions

    def generate_eom(self, timescale=1e-9):
        """
        Convert the sympy expressions describing the equations of motion of the fields into equations of motion which
        are suitable for numerical integration.

        Returns
        -------
        None
        """
        eom_funcs = {}
        self.generate_param_substitutions()
        field_syms = [sympy.Symbol(mode_name) for mode_name in self.mode_names]
        #potential_variable_syms = [sympy.Symbol('delta_'+sym_name) for sym_name in potential_variables]
        combined_syms = field_syms + list(self.potential_variable_syms.values())

        for mode_name, eom_expr in self.eom_exprs.items():
            for pair in self.param_substitutions:
                eom_expr = eom_expr.replace(*pair)
            eom_func = sympy.lambdify(combined_syms, eom_expr*timescale)
            eom_funcs[mode_name] = eom_func

        if len(self.potential_variable_syms) == 0:
            def eom(fields):
                Dfields = np.array([eom_funcs[mode_name](*fields) for mode_name in self.mode_names])
                return Dfields
        else:
            def eom(fields, *args):
                Dfields = np.array([eom_funcs[mode_name](*fields, *args) for mode_name in self.mode_names])
                return Dfields
        self.eom = eom

    def find_steady_state(self, x0=None):
        n_modes = len(self.modes)
        n_potential_variables = len(self.potential_variable_syms)

        def root_wrapper(x):
            y = (x.reshape(-1, 2) * np.array([1, 1j])[np.newaxis,:]).sum(axis=1)
            args = tuple(0 for i in range(n_potential_variables))
            dy = self.eom(y, *args)
            return np.vstack([dy.real, dy.imag]).T.flatten()

        if x0 is None:
            x0 = np.array([0 for i in range(2*n_modes)])

        res = optimize.root(root_wrapper, x0)
        self.steady_state_res = res

    def plot_potential(self, delta_limits=[-0.5, 0.5], n_points=51, ax=None):
        delta_array = np.linspace(*delta_limits, n_points)
        potential_array = np.array([self.potential_func(delta) for delta in delta_array])
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(delta_array, potential_array)
        ax.set_xlabel(r'$\delta$')
        ax.set_ylabel(r'$U(\delta)$')
        if self.delta_0 is not None:
            ax.axvline(self.delta_0)
        return ax

    def generate_dfield_func(self):
        single_dfield_funcs = []
        field_syms = [sympy.Symbol(mode_name) for mode_name in self.mode_names]
        conjugate_field_syms = [sympy.Symbol(mode_name+'^*') for mode_name in self.mode_names]

        temporary_conjugate_substitutions = []
        reverse_conjugate_substitutions = []
        for jdx in range(len(self.mode_names)):
            temporary_conjugate_substitutions += [(sympy.conjugate(field_syms[jdx]), conjugate_field_syms[jdx])]
            reverse_conjugate_substitutions += [(conjugate_field_syms[jdx], sympy.conjugate(field_syms[jdx]))]

        for idx, mode_name in enumerate(self.mode_names):
            temporary_eom_expr = self.eom_exprs[mode_name]
            temporary_eom_expr = temporary_eom_expr.subs(temporary_conjugate_substitutions)
            A = sympy.diff(self.eom_exprs[mode_name], self.drive_syms['f_d'])
            B = sympy.diff(temporary_eom_expr, field_syms[idx])
            B = B.subs(reverse_conjugate_substitutions)
            C = sympy.diff(temporary_eom_expr, conjugate_field_syms[idx])
            C = C.subs(reverse_conjugate_substitutions)
            dfield_expr = (A.conjugate() * C - A * B.conjugate()) / (B * B.conjugate() - C * C.conjugate())
            single_dfield_func = sympy.lambdify(field_syms, dfield_expr.subs(self.param_substitutions))
            single_dfield_funcs.append(single_dfield_func)

        def dfield_func(*fields):
            dfields = np.array([f(*fields) for f in single_dfield_funcs])
            return dfields

        self.dfield_func = dfield_func

    def generate_dalpha_func(self):
        assert len(self.mode_names) == 1
        mode_name = self.mode_names[0]
        param_syms = [self.drive_syms['f_d'], self.drive_syms['epsilon'], self.decay_rate_syms[mode_name]]
        field_sym = sympy.Symbol(mode_name)
        conjugate_field_sym = sympy.Symbol(mode_name+'^*')

        combined_syms = [field_sym] + param_syms

        eom_expr = self.eom_exprs[mode_name]
        for potential_variable_sym in self.potential_variable_syms.values():
            eom_expr = eom_expr.subs(potential_variable_sym, 0.0)

        eom_func = sympy.lambdify(combined_syms, eom_expr, 'numpy')

        temporary_conjugate_substitution = (sympy.conjugate(field_sym), conjugate_field_sym)
        reverse_conjugate_substitution = (conjugate_field_sym, sympy.conjugate(field_sym))

        eom_expr = eom_expr.subs(*temporary_conjugate_substitution)
        A = sympy.diff(eom_expr, self.drive_syms['f_d'])
        B = sympy.diff(eom_expr, field_sym)
        C = sympy.diff(eom_expr, conjugate_field_sym)
        dalpha_expr = (A.conjugate() * C - A * B.conjugate()) / (B * B.conjugate() - C * C.conjugate())
        dalpha_expr = dalpha_expr.subs(*reverse_conjugate_substitution)
        dalpha_func_alpha_dependent = sympy.lambdify(combined_syms, dalpha_expr)

        def root_objective(x, f_d, epsilon, kappa):
            alpha = x[0] + 1j*x[1]
            dalpha = eom_func(alpha, f_d, epsilon, kappa)
            return np.array([dalpha.real, dalpha.imag])

        def dalpha_func(f_d, epsilon, kappa, alpha_ss_0=0.0, return_alpha_ss=True, method='lm'):
            args = (f_d, epsilon, kappa)
            x0 = [alpha_ss_0.real, alpha_ss_0.imag]
            res = optimize.root(root_objective, args=args, x0=x0, tol=0.0, method=method)
            alpha_ss = res.x[0] + 1j*res.x[1]
            dalpha_ss = dalpha_func_alpha_dependent(alpha_ss, f_d, epsilon, kappa)
            if return_alpha_ss:
                return dalpha_ss, alpha_ss
            else:
                return dalpha_ss

        self.dalpha_func = dalpha_func



def package_substitutions(syms, params):
    substitutions = []
    for sym_name, sym in syms.items():
        substitutions.append((sym, params[sym_name]))
    return substitutions


def convert_op_to_expr(op, return_syms=False, mode_names=None, mode_syms=None):
    if mode_names is not None and mode_syms is not None:
        raise Exception('mode_names and mode_syms should not both be specified.')

    if mode_syms is None:
        if mode_names is None:
            mode_names = [letter for letter in string.ascii_lowercase[:op.n_modes]]
        mode_syms = sympy.symbols(mode_names)

    field_array = []
    for sym in mode_syms:
        field_array += [sympy.conjugate(sym), sym]
    field_array = np.array(field_array)

    expr = 0
    for exponents, coeff in op.items():
        term = coeff * np.product(field_array ** exponents)
        expr += term

    if return_syms:
        return expr, mode_syms
    else:
        return expr


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
