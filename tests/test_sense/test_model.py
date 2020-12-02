from normalorder.sense.model import Model
import sympy
import numpy as np
from unittest import TestCase


class TestModel(TestCase):

    def setUp(self):
        np.random.seed(seed=0)
        self.model = Model()

    def test_init(self):
        self.assertEqual(self.model.order, 0)
        self.assertIsInstance(self.model.c_sym, tuple)
        self.assertEqual(len(self.model.c_sym), 2)
        self.assertIsInstance(self.model.resonator_params, dict)

    def test_set_order(self):
        for order in [0, 1, 2, 1, 0]:
            self.model.set_order(order)
            self.assertEqual(self.model.order, order)
            self.assertIsInstance(self.model.c_sym, tuple)
            self.assertEqual(len(self.model.c_sym), order + 2)
            for m in range(order + 1):
                self.assertIsNotNone(self.model.g_expr_gen(m))

    def test_c_and_g_expr_gen(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym = sympy.symbols('phi phi_ext nu n')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
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
        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        for idx in range(order + 2):
            phi = n * 2 * np.pi * np.random.rand()
            assert isinstance(self.model.c_func(idx, phi), float)

        for idx in range(order + 1):
            phi = n * 2 * np.pi * np.random.rand()
            assert isinstance(self.model.g_func(idx, phi), float)

    def test_set_resonator_params(self):
        params = {'a': 1.0,
                  'b': 'test'}
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
        phi_sym, phi_ext_sym, nu_sym, n_sym = sympy.symbols('phi phi_ext nu n')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        potential_params = {'phi_ext': phi_ext,
                            'nu': nu,
                            'n': n}

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
        phi_sym, phi_ext_sym, nu_sym, n_sym = sympy.symbols('phi phi_ext nu n')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        potential_params = {'phi_ext': phi_ext,
                            'nu': nu,
                            'n': n}

        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        for idx in range(self.model.order):
            phi = 2*np.pi*n*np.random.rand()
            self.assertNotNone(self.mode.g_func(idx, phi))

    def test_find_phi_min(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym = sympy.symbols('phi phi_ext nu n')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        potential_params = {'phi_ext': phi_ext,
                            'nu': nu,
                            'n': n}
        self.model.set_order(1)
        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        self.model.find_phi_min()
        self.assertAlmostEqual(self.model.c_func(1, self.model.phi_min), 0.0, places=14)
        self.assertAlmostEqual(self.model.g_func(1, self.model.phi_min), 0.0, places=14)

        x0 = n * 2 * np.pi * np.random.rand()
        self.model.find_phi_min(x0=x0)
        self.assertAlmostEqual(self.model.c_func(1, self.model.phi_min), 0.0, places=14)
        self.assertAlmostEqual(self.model.g_func(1, self.model.phi_min), 0.0, places=14)

    def test_generate_hamiltonian(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym = sympy.symbols('phi phi_ext nu n')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
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
        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        resonator_params = {'f_a': 0.0,
                            'f_b': 0.0,
                            'f_d': 0.0,
                            'f_J': 7500.0,
                            'phi_ext': phi_ext,
                            'nu': nu,
                            'I_ratio_a': 0.001,
                            'I_ratio_b': 0.001,
                            'epsilon': 0.0001,
                            'kappa_a': 0.0001,
                            'kappa_b': 0.0002}
        self.model.set_resonator_params(resonator_params)
        self.model.set_modes()
        self.model.generate_hamiltonian()

    def test_generate_hamiltonian(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym = sympy.symbols('phi phi_ext nu n')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        potential_params = {'phi_ext': phi_ext,
                            'n': n,
                            'nu': nu}
        order = 2

        self.model.set_order(order)
        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        resonator_params = {'f_a': 0.0,
                            'f_b': 0.0,
                            'f_d': 0.0,
                            'f_J': 7500.0,
                            'phi_ext': phi_ext,
                            'nu': nu,
                            'I_ratio_a': 0.001,
                            'I_ratio_b': 0.001,
                            'epsilon': 0.0001,
                            'kappa_a': 0.0001,
                            'kappa_b': 0.0002}

        self.model.set_resonator_params(resonator_params)
        self.model.set_modes()
        self.model.generate_resonator_hamiltonian()
        self.model.generate_potential_hamiltonian()

    def test_generate_eom_func(self):
        phi_sym, phi_ext_sym, nu_sym, n_sym = sympy.symbols('phi phi_ext nu n')
        potential_param_symbols = {'phi_ext': phi_ext_sym,
                                   'nu': nu_sym,
                                   'n': n_sym}
        potential_expr = -nu_sym * sympy.cos(phi_sym) - n_sym * sympy.cos((phi_ext_sym - phi_sym) / n_sym)
        n = 3
        phi_ext = 2 * np.pi * np.random.rand()
        nu = np.random.rand() * 0.9 / n
        potential_params = {'phi_ext': phi_ext,
                            'n': n,
                            'nu': nu}
        order = 4

        self.model.set_order(order)
        self.model.set_potential(potential_expr, potential_param_symbols)
        self.model.set_potential_params(potential_params)

        resonator_params = {'f_a': 0.0,
                            'f_b': 0.0,
                            'f_d': 0.0,
                            'f_J': 7500.0,
                            'phi_ext': phi_ext,
                            'nu': nu,
                            'I_ratio_a': 0.001,
                            'I_ratio_b': 0.001,
                            'epsilon': 0.0001,
                            'kappa_a': 0.0001,
                            'kappa_b': 0.0002}

        self.model.set_resonator_params(resonator_params)
        self.model.set_modes()
        self.model.generate_hamiltonian()
        self.model.generate_lindblad_ops()
        self.model.generate_eom_expr()
        self.model.generate_eom_func()
        pass

