from normalorder.operator.boson import Operator

a = Operator([[1, 0, 1]], ['a'])

a2 = a**2
output = a2 * a.dag()

pass