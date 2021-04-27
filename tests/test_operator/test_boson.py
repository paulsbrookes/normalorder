from unittest import TestCase
from normalorder.operator.boson import Operator


class TestOperatorMethods(TestCase):

    def test_mul(self):
        a = Operator([[1.0, 0, 1]])

        output = a*a
        target = Operator([[1.0, 0, 2]])
        self.assertEqual(output.data, target.data)

        output = a*a.dag()
        target = Operator([[1.0, 1, 1], [1.0, 0, 0]])
        self.assertEqual(output.data, target.data)

        output = a.dag()*a
        target = Operator([[1.0, 1, 1]])
        self.assertEqual(output.data, target.data)

        output = a.dag()*a.dag()
        target = Operator([[1.0, 2, 0]])
        self.assertEqual(output.data, target.data)

        output = a*a*a
        target = Operator([[1.0, 0, 3]])
        self.assertEqual(output.data, target.data)

        output = a*a*a.dag()
        target = Operator([[1.0, 1, 2], [2.0, 0, 1]])
        self.assertEqual(output.data, target.data)

        output = a*a.dag()*a
        target = Operator([[1.0, 1, 2], [1.0, 0, 1]])
        self.assertEqual(output.data, target.data)

        output = a*a.dag()*a.dag()
        target = Operator([[1.0, 2, 1], [2.0, 1, 0]])
        self.assertEqual(output.data, target.data)

        output = a.dag()*a*a
        target = Operator([[1.0, 1, 2]])
        self.assertEqual(output.data, target.data)

        output = a.dag()*a*a.dag()
        target = Operator([[1.0, 2, 1], [1.0, 1, 0]])
        self.assertEqual(output.data, target.data)

        output = a.dag()*a.dag()*a
        target = Operator([[1.0, 2, 1]])
        self.assertEqual(output.data, target.data)

        output = a.dag()*a.dag()*a.dag()
        target = Operator([[1.0, 3, 0]])
        self.assertEqual(output.data, target.data)

