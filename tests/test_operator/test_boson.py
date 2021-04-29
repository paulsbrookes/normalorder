from unittest import TestCase
from normalorder.operator.boson import Operator, tensor
import numpy as np


class TestOperatorMethods(TestCase):

    def setUp(self):
        np.random.seed(seed=0)

    def test_add_single_term_operator_to_itself(self):
        n_subtests = 10
        for i in range(n_subtests):
            with self.subTest(i=i):
                n_modes = np.random.randint(1, 4)
                key = tuple(np.random.randint(1, 10, size=2*n_modes))
                coeff = np.random.randn()
                op = Operator({key: coeff})
                out = op + op
                target = Operator({key: 2*coeff})
                self.assertEqual(out, target)

    def test_add_different_single_term_operators(self):
        n_subtests = 10
        for i in range(n_subtests):
            with self.subTest(i=i):
                n_modes = np.random.randint(1, 4)
                key_1 = np.random.randint(0, 10, size=2*n_modes)
                key_2 = tuple(key_1 + np.random.randint(1, 10, size=2*n_modes))
                key_1 = tuple(key_1)
                coeff_1, coeff_2 = np.random.randn(2)
                op_1 = Operator({key_1: coeff_1})
                op_2 = Operator({key_2: coeff_2})
                out = op_1 + op_2
                target = Operator({key_1: coeff_1, key_2: coeff_2})
                self.assertEqual(out, target)

    def test_add_different_multi_term_operators(self):
        n_subtests = 10
        for i in range(n_subtests):
            self.subTest(i=i)
            n_modes = np.random.randint(1, 4)
            n_terms_1, n_terms_2 = np.random.randint(1, 10, size=2)
            dict_1 = {tuple(np.random.randint(0, 10, size=2*n_modes)): np.random.randn() for i in range(n_terms_1)}
            dict_2 = {tuple(np.random.randint(0, 10, size=2*n_modes)): np.random.randn() for i in range(n_terms_2)}
            dict_added = {x: dict_1.get(x, 0) + dict_2.get(x, 0) for x in set(dict_1.keys()).union(dict_2.keys())}
            op_1 = Operator(dict_1)
            op_2 = Operator(dict_2)
            op_added = op_1 + op_2
            op_target = Operator(dict_added)
            self.assertEqual(op_added, op_target)

    def test_add_zero_operator(self):
        n_subtests = 10
        for i in range(n_subtests):
            with self.subTest(i=i):
                n_modes = np.random.randint(1, 4)
                zero_op = Operator(n_modes=n_modes)
                key = tuple(np.random.randint(1, 10, size=2*n_modes))
                coeff = np.random.randn()
                op = Operator({key: coeff})
                out = op + zero_op
                target = Operator({key: coeff})
                self.assertEqual(out, target)

    def test_scalar_mul(self):
        n_subtests = 10
        for i in range(n_subtests):
            with self.subTest(i=i):
                n_modes = np.random.randint(1, 4)
                test_spec = {}
                target_spec = {}
                n_terms = np.random.randint(1, 5)
                scalar = np.random.randn()
                for term_idx in range(n_terms):
                    key = tuple(np.random.randint(0, 5, size=2*n_modes))
                    coeff = np.random.randn()
                    test_spec[key] = coeff
                    target_spec[key] = coeff*scalar
                test_op = scalar*Operator(test_spec)
                target_op = Operator(target_spec)
                self.assertEqual(test_op, target_op)

        for n_modes in range(1,4):
            with self.subTest(i=n_modes):
                zero_op = Operator(n_modes=n_modes)
                scalar = np.random.randn()
                self.assertEqual(zero_op, scalar*zero_op)

    def test_single_mode_mul(self):
        a = Operator([[1.0, 0, 1]])

        output = a*a
        target = Operator([[1.0, 0, 2]])
        self.assertEqual(output, target)

        output = a*a.dag()
        target = Operator([[1.0, 1, 1], [1.0, 0, 0]])
        self.assertEqual(output, target)

        output = a.dag()*a
        target = Operator([[1.0, 1, 1]])
        self.assertEqual(output, target)

        output = a.dag()*a.dag()
        target = Operator([[1.0, 2, 0]])
        self.assertEqual(output, target)

        output = a*a*a
        target = Operator([[1.0, 0, 3]])
        self.assertEqual(output, target)

        output = a*a*a.dag()
        target = Operator([[1.0, 1, 2], [2.0, 0, 1]])
        self.assertEqual(output, target)

        output = a*a.dag()*a
        target = Operator([[1.0, 1, 2], [1.0, 0, 1]])
        self.assertEqual(output, target)

        output = a*a.dag()*a.dag()
        target = Operator([[1.0, 2, 1], [2.0, 1, 0]])
        self.assertEqual(output, target)

        output = a.dag()*a*a
        target = Operator([[1.0, 1, 2]])
        self.assertEqual(output, target)

        output = a.dag()*a*a.dag()
        target = Operator([[1.0, 2, 1], [1.0, 1, 0]])
        self.assertEqual(output, target)

        output = a.dag()*a.dag()*a
        target = Operator([[1.0, 2, 1]])
        self.assertEqual(output, target)

        output = a.dag()*a.dag()*a.dag()
        target = Operator([[1.0, 3, 0]])
        self.assertEqual(output, target)

    def test_multi_mode_mul_and_tensor(self):
        np.random.seed(seed=0)

        n_subtests = 10
        for i in range(n_subtests):
            with self.subTest(i=i):
                exponents = np.random.randint(0, 5, size=4)
                a = Operator([[1.0, 0, 1, 0, 0]])
                b = Operator([[1.0, 0, 0, 0, 1]])
                output = a.dag()**exponents[0] * a**exponents[1] * b.dag()**exponents[2] * b**exponents[3]

                a_1 = Operator([[1.0, 0, 1]])
                b_1 = Operator([[1.0, 0, 1]])
                target_a = a_1.dag()**exponents[0] * a_1**exponents[1]
                target_b = b_1.dag()**exponents[2] * b_1**exponents[3]
                target = tensor(target_a, target_b)

                self.assertEqual(output, target)

        for i in range(10):
            with self.subTest(i=i):
                exponents = np.random.randint(0, 5, size=6)
                a = Operator([[1.0, 0, 1, 0, 0, 0, 0]])
                b = Operator([[1.0, 0, 0, 0, 1, 0, 0]])
                c = Operator([[1.0, 0, 0, 0, 0, 0, 1]])
                output = a.dag()**exponents[0] * a**exponents[1] * b.dag()**exponents[2] * b**exponents[3] * c.dag()**exponents[4] * c**exponents[5]

                a_1 = Operator([[1.0, 0, 1]])
                b_1 = Operator([[1.0, 0, 1]])
                c_1 = Operator([[1.0, 0, 1]])
                target_a = a_1.dag()**exponents[0] * a_1**exponents[1]
                target_b = b_1.dag()**exponents[2] * b_1**exponents[3]
                target_c = c_1.dag()**exponents[4] * c_1**exponents[5]
                target = tensor(tensor(target_a, target_b), target_c)

                self.assertEqual(output, target)

    def test_pow_against_mul(self):
        n_terms = 3
        n_modes = 3
        op = Operator({tuple(np.random.randint(0, 5, size=2*n_modes)): np.random.randn() for i in range(n_terms)})

        for power in range(0, 4):
            with self.subTest(i=power):
                target_op = Operator({tuple(np.zeros(2*n_modes, dtype=int)): 1.0})
                for i in range(power):
                    target_op *= op
                test_op = op**power
                self.assertEqual(test_op, target_op)