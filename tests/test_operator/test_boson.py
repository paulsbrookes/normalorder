from unittest import TestCase
from normalorder.operator.boson import Operator, tensor
import numpy as np


class TestOperatorMethods(TestCase):

    def test_add(self):


    def test_scalar_mul(self):
        np.random.seed(seed=0)

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

    def test_multi_mode_mul(self):
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
