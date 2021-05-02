from unittest import TestCase
import numpy as np
from normalorder.sense.mode import Mode
from scipy.integrate import quad


def calc_relative_error(val1, val2):
    return abs(val1-val2)/min(abs(val1), abs(val2))


class Test(TestCase):

    def setUp(self):
        np.random.seed(seed=0)

    def test_crosscheck_potential_energy(self):
        n_subtests = 10
        places = 5
        eps = 10 ** (-places - 3)

        for i in range(n_subtests):
            with self.subTest(i=i):
                Z = 50 * np.random.uniform(0.1, 10.0)  # ohms
                v_p = 1.1817e8 * np.random.uniform(0.1, 10.0)  # m/s
                l = 0.007809801198766881  # m
                L_0 = Z / v_p  # H/m
                C_0 = 1 / (v_p * Z)  # F/m
                x_J = 0.12481751778187578 * l # m
                L_J = 7.696945473011622e-11 * np.random.uniform(0.0, 1.0)  # H
                C_i = 0.0
                C_o = 0.0
                params = {'C_i': C_i, 'C_o': C_o, 'L_0': L_0, 'C_0': C_0, 'x_J': x_J, 'l': l, 'L_J': L_J}
                k_init = np.pi / (2 * params['l'])

                mode = Mode(eps=eps)
                mode.set_params(params)
                mode.solve(k_init)

                transmission_line_potential_energy = quad(lambda x: mode.calc_u_grad(x) ** 2 / (2 * L_0), -l, l,
                                                          epsrel=eps, epsabs=eps)[0]
                element_potential_energy = mode.Delta ** 2 / (2 * L_J)
                total_potential_energy = transmission_line_potential_energy + element_potential_energy
                crosscheck_potential_energy = 1 / (2 * mode.L)

                relative_error = calc_relative_error(crosscheck_potential_energy, total_potential_energy)

                self.assertAlmostEqual(relative_error, 0.0, places=places)

    def test_current_continuity(self):
        n_subtests = 10
        places = 10
        eps = 10 ** (-places - 3)

        for i in range(n_subtests):
            with self.subTest(i=i):
                Z = 50 * np.random.uniform(0.1, 10.0)  # ohms
                v_p = 1.1817e8 * np.random.uniform(0.1, 10.0)  # m/s
                l = 0.007809801198766881  # m
                L_0 = Z / v_p  # H/m
                C_0 = 1 / (v_p * Z)  # F/m
                x_J = 0.12481751778187578 * l # m
                L_J = 7.696945473011622e-11 * np.random.uniform(0.0, 1.0)  # H
                C_i = 0.0
                C_o = 0.0
                params = {'C_i': C_i, 'C_o': C_o, 'L_0': L_0, 'C_0': C_0, 'x_J': x_J, 'l': l, 'L_J': L_J}
                k_init = np.pi / (2 * params['l'])

                mode = Mode(eps=eps)
                mode.set_params(params)
                mode.solve(k_init)

                current_l = mode.calc_u_l_grad(x_J) / L_0
                current_r = mode.calc_u_r_grad(x_J) / L_0
                current_element = mode.Delta / L_J

                self.assertAlmostEqual(calc_relative_error(current_l, current_r), 0.0, places=places)
                self.assertAlmostEqual(calc_relative_error(current_l, current_element), 0.0, places=places)

    def test_wavevector_solution(self):
        n_subtests = 10
        places = 10
        eps = 10 ** (-places - 3)

        for i in range(n_subtests):
            with self.subTest(i=i):
                Z = 50 * np.random.uniform(0.1, 10.0)  # ohms
                v_p = 1.1817e8 * np.random.uniform(0.1, 10.0)  # m/s
                l = 0.007809801198766881  # m
                L_0 = Z / v_p  # H/m
                C_0 = 1 / (v_p * Z)  # F/m
                x_J = 0.12481751778187578 * l  # m
                L_J = 7.696945473011622e-11 * np.random.uniform(0.0, 1.0)  # H
                C_i = 0.0
                C_o = 0.0
                params = {'C_i': C_i, 'C_o': C_o, 'L_0': L_0, 'C_0': C_0, 'x_J': x_J, 'l': l, 'L_J': L_J}
                k_init = np.pi / (2 * params['l'])

                mode = Mode(eps=eps)
                mode.set_params(params)
                mode.solve(k_init)

                lhs = L_0*l*(np.tan(mode.k*(x_J-l) + mode.phi_o)-np.tan(mode.k*(x_J+l)-mode.phi_i))/L_J
                rhs = mode.k*l
                self.assertAlmostEqual(calc_relative_error(lhs, rhs), 0.0, places=places)

    def test_derivative_orthogonality(self):
        n_subtests = 10
        places = 8
        eps = 10 ** (-places - 3)

        for i in range(n_subtests):
            with self.subTest(i=i):
                Z = 50 * np.random.uniform(0.1, 10.0)  # ohms
                v_p = 1.1817e8 * np.random.uniform(0.1, 10.0)  # m/s
                l = 0.007809801198766881  # m
                L_0 = Z / v_p  # H/m
                C_0 = 1 / (v_p * Z)  # F/m
                x_J = 0.12481751778187578 * l # m
                L_J = 7.696945473011622e-11 * np.random.uniform(0.0, 1.0)  # H
                C_i = 0.0
                C_o = 0.0
                params = {'C_i': C_i, 'C_o': C_o, 'L_0': L_0, 'C_0': C_0, 'x_J': x_J, 'l': l, 'L_J': L_J}

                m = 1
                k_init_m = m * np.pi / (2 * params['l'])
                mode_m = Mode(eps=eps)
                mode_m.set_params(params)
                mode_m.solve(k_init_m)

                n = 2
                k_init_n = n * np.pi / (2 * params['l'])
                mode_n = Mode(eps=eps)
                mode_n.set_params(params)
                mode_n.solve(k_init_n)

                inner_product = quad(lambda x: mode_m.calc_u_grad(x) * mode_n.calc_u_grad(x) / (2 * L_0), -l, l,
                                                          epsrel=eps, epsabs=eps)[0]
                inner_product += mode_m.Delta * mode_n.Delta / (2 * L_J)

                relative_error = inner_product * max(mode_m.L, mode_n.L)
                self.assertAlmostEqual(relative_error, 0.0, places=places)

    def test_derivative_inner_product_orthonormality(self):

        # To-do: include higher modes.

        n_subtests = 10
        places = 5
        eps = 10 ** (-places - 3)

        for i in range(n_subtests):
            with self.subTest(i=i):
                Z = 50 * np.random.uniform(0.1, 10.0)  # ohms
                v_p = 1.1817e8 * np.random.uniform(0.1, 10.0)  # m/s
                l = 0.007809801198766881  # m
                L_0 = Z / v_p  # H/m
                C_0 = 1 / (v_p * Z)  # F/m
                x_J = 0.12481751778187578 * l # m
                L_J = 7.696945473011622e-11 * np.random.uniform(0.0, 1.0)  # H
                C_i = 0.0
                C_o = 0.0
                params = {'C_i': C_i, 'C_o': C_o, 'L_0': L_0, 'C_0': C_0, 'x_J': x_J, 'l': l, 'L_J': L_J}

                m = 1
                k_init_m = m * np.pi / (2 * params['l'])
                mode_m = Mode(eps=eps)
                mode_m.set_params(params)
                mode_m.solve(k_init_m)

                n = 1
                k_init_n = n * np.pi / (2 * params['l'])
                mode_n = Mode(eps=eps)
                mode_n.set_params(params)
                mode_n.solve(k_init_n)

                inner_product = quad(lambda x: mode_m.calc_u_grad(x) * mode_n.calc_u_grad(x) / L_0,
                                     -l, l, epsrel=eps, epsabs=eps)[0]
                inner_product += mode_m.Delta * mode_n.Delta / L_J

                target = 1 / mode_m.L

                relative_error = calc_relative_error(inner_product, target)
                self.assertAlmostEqual(relative_error, 0.0, places=places)