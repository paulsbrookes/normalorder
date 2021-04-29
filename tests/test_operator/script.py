from normalorder.operator.boson import Operator
import time

a = Operator([[1.0,0,1,0,0,0,0]])
b = Operator([[1.0,0,0,0,1,0,0]])
c = Operator([[1.0,0,0,0,0,0,1]])
delta = a + a.dag() + b + b.dag() + c + c.dag()

start = time.time()
delta**4
print(time.time()-start)