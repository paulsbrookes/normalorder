from unittest import TestCase
from normalorder.operator.boson import Operator, multiply_operators
import sympy


class TestMultiply_operators(TestCase):
    def test_function_multiply_operators(self):
        coeff = sympy.Symbol('x')
        a = Operator([[coeff,0,1]])
        x = a + a.dag()
        output = multiply_operators(x, x)

    def test_class_operator_multiply(self):
        coeff = sympy.Symbol('x')
        a = Operator([[coeff,0,1]])
        x = a + a.dag()
        output = x*x




