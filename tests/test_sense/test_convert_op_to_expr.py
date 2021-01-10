from unittest import TestCase
from normalorder.sense.model_current import convert_op_to_expr
from normalorder.operator.boson import Operator
import sympy
import numpy as np
import string


class TestConvert_op_to_expr(TestCase):
    def setUp(self):
        np.random.seed(seed=0)

    def test_convert_op_to_expr(self):
        op = Operator([[1, 0, 1]], modes=['a'])
        expr = convert_op_to_expr(op)
        test_expr = sympy.Symbol('a')
        self.assertEqual(expr, test_expr)

        for trial_idx in range(10):
            with self.subTest(i=trial_idx):
                coeff = np.random.randn()
                n_modes = 3
                modes = list(string.ascii_lowercase[:n_modes])
                mode_syms = sympy.symbols(modes)
                exponents = list(np.random.randint(0, 10, size=2*n_modes))
                ledger = [[coeff] + exponents]
                op = Operator(ledger, modes=modes)
                expr = convert_op_to_expr(op)
                test_expr = coeff
                for idx in range(n_modes):
                    test_expr *= sympy.conjugate(mode_syms[idx])**exponents[2*idx]
                    test_expr *= mode_syms[idx]**exponents[2*idx + 1]
                self.assertEqual(expr, test_expr)

