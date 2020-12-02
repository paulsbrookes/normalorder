from unittest import TestCase
from normalorder.sense.model import generate_x_pow
import sympy
from copy import deepcopy


class TestGenerate_x_pow(TestCase):
    def test_generate_x_pow(self):
        exponent = 0
        for n_modes in [1, 2, 3]:
            with self.subTest(i=n_modes):
                mode_coeffs = tuple((n_modes)*[1])
                self.assertEqual(generate_x_pow(exponent, mode_coeffs=mode_coeffs), 1)

        n_modes = 1
        x = sympy.Symbol('x')
        mode_coeffs = (x,)
        for exponent in [1, 2, 3]:
            with self.subTest(i=exponent):
                out = generate_x_pow(exponent, mode_coeffs=mode_coeffs)
                self.assertIsNotNone(generate_x_pow(exponent, mode_coeffs=mode_coeffs))

        U = 0
        x_pow_list = []
        for exponent in [0, 1, 2]:
            x_pow = deepcopy(generate_x_pow(exponent, mode_coeffs=mode_coeffs))
            x_pow_list.append(x_pow)
            U += generate_x_pow(exponent, mode_coeffs=mode_coeffs)
        pass

