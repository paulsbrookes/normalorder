from normalorder.operator.boson import Operator

a = Operator([[1, 0, 1]], ['a'])
delta = a + a.dag()
test = delta**10
pass