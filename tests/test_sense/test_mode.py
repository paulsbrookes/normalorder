from unittest import TestCase
import numpy as np
from normalorder.sense.mode import find_k


class Test(TestCase):
    def test_find_k(self):
        params = dict()
        params['l'] = 0.5
        params['L_0'] = 1.0
        params['L_J'] = 0.0001
        params['x_J'] = 0.0
        k_initial = 1.999*np.pi
        k = find_k(k_initial, params)
        print(k)
