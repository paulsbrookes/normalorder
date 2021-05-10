from normalorder.sense.model import Model, apply_rwa
from normalorder.operator.boson import Operator
import sympy
import numpy as np
from unittest import TestCase
from scipy import constants


def reform_ham(ham):
    Delta = ham.get((2, 0), 0)
    mu = ham.get((1, 0), 0)
    omega_a = ham.get((1, 1), 0)
    r = 0.5 * np.arctanh(2 * Delta / omega_a)
    omega_b = omega_a / np.cosh(2 * r)
    s = mu / (omega_b * (np.cosh(r) + np.sinh(r)))
    b = Operator([[1.0, 0, 1]])
    a = np.cosh(r) * b - np.sinh(r) * b.dag() - s * (np.cosh(r) - np.sinh(r))
    new_ham = Operator(n_modes=1)
    for key, coeff in ham.items():
        new_ham += a.dag() ** key[0] * a ** key[1] * coeff
    return new_ham


class TestModel(TestCase):

    def setUp(self):
        np.random.seed(seed=0)
        self.model = Model()
        self.Phi_0 = constants.physical_constants['mag. flux quantum'][0]

    def test_init(self):
        self.assertEqual(self.model.order, 0)
        self.assertIsInstance(self.model.c_sym, tuple)
        self.assertEqual(len(self.model.c_sym), 2)
        self.assertIsInstance(self.model.resonator_params, dict)

    def test_obtaining_L_J(self):
        L_J_list = [1e-11, 1e-10, 1e-9]
        for i, L_J in enumerate(L_J_list):
            with self.subTest(i=i):
                delta_sym, L_J_sym = sympy.symbols('delta L_J')
                potential_param_symbols = {'L_J': L_J_sym}
                potential_expr = (delta_sym * self.Phi_0) ** 2 / (2 * L_J_sym * constants.h)
                order = 2
                self.model.set_order(order)
                self.model.set_potential(potential_expr, potential_param_symbols)
                potential_params = {'L_J': L_J}
                self.model.set_potential_params(potential_params, delta_min_guess=0.0)
                delta = 1e-5
                self.assertAlmostEqual(1.0, self.model.resonator_params['L_J']/L_J, delta=delta)

    def test_c_2(self):
        L_J_list = [1e-11, 1e-10, 1e-9]
        for i, L_J in enumerate(L_J_list):
            with self.subTest(i=i):
                delta_sym, L_J_sym = sympy.symbols('delta L_J')
                potential_param_symbols = {'L_J': L_J_sym}
                potential_expr = (delta_sym * self.Phi_0) ** 2 / (2 * L_J_sym * constants.h)
                order = 2
                self.model.set_order(order)
                self.model.set_potential(potential_expr, potential_param_symbols)
                potential_params = {'L_J': L_J}
                self.model.set_potential_params(potential_params, delta_min_guess=0.0)
                c_2_target = self.model.Phi_0**2 / (2*L_J*constants.h)
                delta = 1e-5
                self.assertAlmostEqual(1.0, self.model.c_func(2)/c_2_target, delta=delta)

    def test_crosscheck_frequency_shift_from_dc_2_with_Mode(self):

        delta_sym, L_J_sym = sympy.symbols('delta L_J')
        potential_param_symbols = {'L_J': L_J_sym}
        potential_expr = (delta_sym * self.Phi_0) ** 2 / (2 * L_J_sym * constants.h)
        delta_min_guess = 0.0
        order = 2

        Z = 50  # ohms
        v_p = 1.1817e8  # m/s
        l = 0.007809801198766881  # m
        L_0 = Z / v_p  # H/m
        C_0 = 1 / (v_p * Z)  # F/m
        x_J = 0.002626535356654626  # m
        L_J_1 = 1e-11
        Delta_L_J = 0.0005 * L_J_1
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

        frequency_shift_from_Mode = model_2.modes['a'].frequency - model_1.modes['a'].frequency
        frequency_shift_from_dc_2 = (model_1.dc_func(2, 'L_J') * Delta_L_J * model_1.delta ** 2)[(1, 1)]
        self.assertAlmostEqual(1.0, frequency_shift_from_dc_2/frequency_shift_from_Mode, delta=1e-5)

    def test_generate_potential_derivative_op(self):

        Z = 50  # ohms
        v_p = 1.1817e8  # m/s
        l = 0.007809801198766881  # m
        L_0 = Z / v_p  # H/m
        C_0 = 1 / (v_p * Z)  # F/m
        x_J = 0.002626535356654626  # m

        beta_list = [1e11, 1e10, 1e9]
        for i, beta in enumerate(beta_list):
            with self.subTest(i=i):
                delta_sym, beta_sym = sympy.symbols('delta beta')
                potential_param_symbols = {'beta': beta_sym}
                potential_expr = beta_sym * (delta_sym * self.Phi_0) ** 2 / (2 * constants.h)
                order = 2
                self.model.set_order(order)
                self.model.set_resonator_params(x_J=x_J, L_0=L_0, l=l, C_0=C_0)
                self.model.set_potential(potential_expr, potential_param_symbols)
                potential_params = {'beta': beta}
                self.model.set_potential_params(potential_params, delta_min_guess=0.0)
                self.model.set_modes(['a'])
                Delta_potential = self.model.generate_potential_derivative_op('beta')
                target_op = self.Phi_0**2 * self.model.delta**2 * (1 / (2 * constants.h))
                target_op = apply_rwa(target_op)
                difference = target_op - Delta_potential
                distance = sum(map(abs, difference.values()))
                self.assertAlmostEqual(0.0, distance, delta=1e-5)

    def test_different_delta_0_to_produce_same_hamiltonian(self):
        order = 4
        delta_sym = sympy.symbols('delta')
        potential_expr = 5e12 * delta_sym ** 2 + 5e9 * delta_sym ** 3 + 5e6 * delta_sym ** 4
        C_0 = 3e-12
        delta_0 = 1e-1

        potential_params = {}
        potential_param_symbols = {}
        model = Model(lumped_element=True)
        model.set_order(order)
        model.set_potential(potential_expr, potential_param_symbols)
        model.set_resonator_params(C_0=C_0)

        model.set_potential_params(potential_params, delta_0=delta_0*np.random.randn())
        model.set_modes(names=['a'])
        model.generate_hamiltonian(drive=False)
        ham_1 = model.hamiltonian

        model.set_potential_params(potential_params, delta_0=delta_0*np.random.randn())
        model.set_modes(names=['a'])
        model.generate_hamiltonian(drive=False)
        ham_2 = model.hamiltonian

        ham_1_prime = ham_1
        ham_2_prime = ham_2
        for i in range(10):
            ham_1_prime = reform_ham(ham_1_prime)
            ham_2_prime = reform_ham(ham_2_prime)
        difference = ham_1_prime - ham_2_prime
        difference[(0, 0)] = 0.0
        for key, value in difference.items():
            self.assertAlmostEqual(value, 0.0, delta=1e-5)

    def test_changing_the_parameters_of_the_potential(self):
        order = 4
        delta_sym, beta_sym = sympy.symbols('delta beta')
        potential_param_symbols = {'beta': beta_sym}
        potential_expr = (5e12 * (delta_sym - beta_sym) ** 2 + 5e9 * (delta_sym - beta_sym) ** 3 + 5e6 * (
                    delta_sym - beta_sym) ** 4)

        L_0 = 6.740357146050338e-10
        C_0 = 2.6435816869821043e-12

        beta = 1e-2 * np.random.randn()
        Delta_beta = 1e-2 * np.random.randn()
        delta_min_guess = 0.0

        potential_params_1 = {'beta': beta}
        model_1 = Model(lumped_element=True)
        model_1.set_order(order)
        model_1.set_potential(potential_expr, potential_param_symbols)
        model_1.set_resonator_params(L_0=L_0, C_0=C_0)
        model_1.set_potential_params(potential_params_1, delta_min_guess=delta_min_guess)
        model_1.set_modes(names=['a'])
        model_1.generate_hamiltonian(drive=False)

        potential_params_2 = {'beta': beta + Delta_beta}
        model_2 = Model(lumped_element=True)
        model_2.set_order(order)
        model_2.set_potential(potential_expr, potential_param_symbols)
        model_2.set_resonator_params(L_0=L_0, C_0=C_0)
        model_2.set_potential_params(potential_params_2, delta_min_guess=delta_min_guess)
        model_2.set_modes(names=['a'])
        model_2.generate_hamiltonian(drive=False)

        potential_1 = model_1.generate_potential_hamiltonian(inplace=False, orders=range(order + 1), rwa=False)
        potential_1_prime = model_1.generate_potential_hamiltonian(inplace=False,
                                                                   substitutions={'beta': beta + Delta_beta},
                                                                   orders=range(order + 1), rwa=False)
        ham_1_prime = model_1.hamiltonian + potential_1_prime - potential_1
        ham_2 = model_2.hamiltonian

        ham_1_ref = ham_1_prime
        ham_2_ref = ham_2
        for i in range(10):
            ham_1_ref = reform_ham(ham_1_ref)
            ham_2_ref = reform_ham(ham_2_ref)
        difference = ham_1_ref - ham_2_ref
        difference[(0, 0)] = 0.0
        for key, value in difference.items():
            self.assertAlmostEqual(value, 0.0, delta=1e-7)



    def test_set_order(self):
        for order in [0, 1, 2, 1, 0]:
            self.model.set_order(order)
            self.assertEqual(self.model.order, order)
            self.assertIsInstance(self.model.c_sym, tuple)
            self.assertEqual(len(self.model.c_sym), order + 2)
            for m in range(order + 1):
                self.assertIsNotNone(self.model.generate_g_expr(m))

    def test_c_and_g_expr_gen(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym = sympy.symbols('phi phi_ext nu n')
        potential_param_syms = {'phi_ext': phi_ext_sym,
                                'nu': nu_sym,
                                'n': n_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        potential_params = {'phi_ext': phi_ext,
                            'n': n,
                            'nu': nu}
        order = 3

        self.model.set_order(order)
        self.model.set_potential(potential_expr, potential_param_syms)
        self.model.set_potential_params(potential_params)

        for idx in range(order + 2):
            phi = n * 2 * np.pi * np.random.rand()
            assert isinstance(self.model.c_func(idx, phi), float)

        for idx in range(order + 1):
            phi = n * 2 * np.pi * np.random.rand()
            assert isinstance(self.model.g_func(idx, phi), float)

    def test_set_resonator_params(self):
        params_a = {'f': 1.0,
                    'kappa': 'test'}
        params_b = {'f': 2.0,
                    'kappa': 'asdf'}
        params = {'a': params_a,
                  'b': params_b}
        self.model.set_resonator_params(params)
        self.assertEqual(self.model.resonator_params, params)

    def test_set_potential_and_params(self):
        phi_sym, x, y = sympy.symbols('phi x y')
        potential_param_symbols = {'x': x,
                                   'y': y}
        potential_expr = x * y * phi_sym
        self.model.set_potential(potential_expr, potential_param_symbols)
        self.assertIsInstance(self.model.c_sym, tuple)
        self.assertEqual(self.model.potential_expr, potential_expr)
        self.assertEqual(self.model.potential_syms, potential_param_symbols)

        potential_params = {'x': np.random.rand(),
                            'y': np.random.randn()}
        self.model.set_potential_params(potential_params)
        self.assertEqual(self.model.potential_params, potential_params)

    def test_c_func(self):
        phi_sym = sympy.symbols('phi')
        test_c = np.random.randn(self.model.order+5)
        offset = np.random.randn()
        potential_expr = sum(test_c[m]*(phi_sym-offset)**m for m in range(self.model.order+2))
        potential_param_symbols = {}
        potential_params = {}

        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        for m in range(self.model.order+2):
            self.assertAlmostEqual(self.model.c_func(m, offset), test_c[m], places=14)

    def test_potential_func(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym, f_J_sym = sympy.symbols('phi phi_ext nu n f_J')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym,
                                   'f_J': f_J_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        f_J = 7500.0
        potential_params = {'phi_ext': phi_ext,
                            'nu': nu,
                            'n': n,
                            'f_J': f_J}

        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        param_substitutions = [(potential_param_symbols[key], potential_params[key])
                               for key in potential_param_symbols.keys()]

        for trial_idx in range(10):
            with self.subTest(i=trial_idx):
                phi = 2 * np.pi * n * np.random.rand()
                substitutions = param_substitutions + [(phi_sym, phi)]
                self.assertEqual(self.model.potential_func(phi), potential_expr.subs(substitutions))

    def test_g_func(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym, f_J_sym = sympy.symbols('phi phi_ext nu n f_J')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym,
                                   'f_J': f_J_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        f_J = 7500.0
        potential_params = {'phi_ext': phi_ext,
                            'nu': nu,
                            'n': n,
                            'f_J': f_J}

        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        for idx in range(self.model.order):
            phi = 2*np.pi*n*np.random.rand()
            self.assertNotNone(self.mode.g_func(idx, phi))

    def test_find_phi_min(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym, f_J_sym = sympy.symbols('phi phi_ext nu n f_J')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym,
                                   'f_J': f_J_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        f_J = 7500.0
        potential_params = {'phi_ext': phi_ext,
                            'nu': nu,
                            'n': n,
                            'f_J': f_J}
        self.model.set_order(1)
        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        self.model.find_phi_min()
        self.assertAlmostEqual(self.model.c_func(1, self.model.phi_min), 0.0, places=14)
        self.assertAlmostEqual(self.model.g_func(1, self.model.phi_min), 0.0, places=14)

        x0 = n * 2 * np.pi * np.random.rand()
        self.model.find_phi_min(phi_min_guess=x0)
        self.assertAlmostEqual(self.model.c_func(1, self.model.phi_min), 0.0, places=14)
        self.assertAlmostEqual(self.model.g_func(1, self.model.phi_min), 0.0, places=14)

    def test_generate_hamiltonian(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym, f_J_sym = sympy.symbols('phi phi_ext nu n f_J')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym,
                                   'f_J': f_J_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        f_J = 7500.0
        potential_params = {'phi_ext': phi_ext,
                            'nu': nu,
                            'n': n,
                            'f_J': f_J}
        order = 3

        self.model.set_order(order)
        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        resonator_params_a = {'f': 1.0,
                              'I_ratio': 0.001,
                              'kappa': 0.0001}

        resonator_params_b = {'f': 2.0,
                              'I_ratio': 0.001,
                              'kappa': 0.0001}

        resonator_params = {'a': resonator_params_a,
                            'b': resonator_params_b}

        f_d = 1.1
        epsilon = 0.001

        self.model.set_resonator_params(resonator_params)
        self.model.set_drive_params(f_d, epsilon)
        self.model.generate_hamiltonian()

    def test_generate_eom_func(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym, f_J_sym = sympy.symbols('phi phi_ext nu n f_J')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym,
                                   'f_J': f_J_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        f_J = 7500.0
        potential_params = {'phi_ext': phi_ext,
                            'nu': nu,
                            'n': n,
                            'f_J': f_J}
        order = 3

        self.model.set_order(order)
        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        resonator_params_a = {'f': 1.0,
                              'I_ratio': 0.001,
                              'kappa': 0.0001}

        resonator_params_b = {'f': 2.0,
                              'I_ratio': 0.001,
                              'kappa': 0.0001}

        resonator_params = {'a': resonator_params_a,
                            'b': resonator_params_b}

        f_d = 1.1
        epsilon = 0.001

        self.model.set_resonator_params(resonator_params)
        self.model.set_drive_params(f_d, epsilon)
        self.model.generate_hamiltonian()
        self.model.generate_lindblad_ops()
        self.model.generate_eom_ops()
        self.model.generate_eom_exprs()
        self.model.generate_eom()
        x = np.array([1.0, 2.0])
        output = self.model.eom(x)

        self.model.generate_eom(potential_variables=['phi_ext'])
        x = np.array([1.0, 2.0])
        delta_phi_ext = 0.0
        output = self.model.eom(x,delta_phi_ext)

    def test_Dphimin_Dparam_func(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym, f_J_sym = sympy.symbols('phi phi_ext nu n f_J')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym,
                                   'f_J': f_J_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        f_J = 7500.0
        potential_params = {'phi_ext': phi_ext,
                            'nu': nu,
                            'n': n,
                            'f_J': f_J}
        order = 3

        self.model.set_order(order)
        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)
        self.model.Dphimin_Dparam_func('phi_ext')
        self.model.Dg_at_phimin_Dparam_func(3, 'phi_ext')