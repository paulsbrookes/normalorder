import numpy as np
from sortedcontainers import SortedDict
import sympy
from scipy.special import factorial
import numbers


def commutator_func(x, y):
    if not isinstance(x, numbers.Integral) or not isinstance(y, numbers.Integral) or x < 0 or y < 0:
        raise Exception('This function only accepts exponents which are positive integers or zero.')

    def coeff_func(x, y, l):
        coeff = factorial(x) * factorial(y) / (factorial(x - l) * factorial(y - l) * factorial(l))
        return coeff

    commutator_op = Operator(n_modes=1)
    for l in range(1, min(x, y) + 1):
        commutator_op[(y - l, x - l)] = coeff_func(x, y, l)

    return commutator_op


def tensor(op_1, op_2):
    n_modes = op_1.n_modes + op_2.n_modes
    op_final = Operator(n_modes=n_modes)
    for key_1, coeff_1 in op_1.items():
        for key_2, coeff_2 in op_2.items():
            key_combined = key_1 + key_2
            coeff_combined = coeff_1 * coeff_2
            op_final[key_combined] = coeff_combined
    return op_final


class Operator(SortedDict):

    def __init__(self, spec=None, n_modes=None):

        if (spec is None) == (n_modes is None):
            raise Exception('Either spec or n_modes should be provided, but not both.')

        if n_modes is not None:
            self.n_modes = n_modes
            super().__init__()
        else:
            if isinstance(spec, list):
                spec = {tuple(element[1:]): element[0] for element in spec}
            if len(spec) == 0:
                raise Exception('Spec must not be empty.')
            for key in spec.keys():
                if not all(map(lambda el: isinstance(el, numbers.Integral), key)):
                    raise Exception('All elements of the key must be integers.')
            super().__init__(spec)

            all_tuples_bool = all(list(map(lambda key: isinstance(key, tuple), self.keys())))
            if not all_tuples_bool:
                raise Exception('All keys should be tuples.')

            unique_key_lengths = list(set(map(len, self.keys())))
            if len(unique_key_lengths) >= 2:
                raise Exception('All keys should be of identical length.')
            if len(unique_key_lengths) == 1:
                key_length = unique_key_lengths[0]
                if key_length % 2 != 0:
                    raise Exception('All keys should be of even length.')
                n_modes = key_length // 2

            self.n_modes = n_modes

    def __add__(self, other):
        if not isinstance(other, (float, numbers.Integral, complex, Operator, sympy.core.mul.Mul, sympy.core.symbol.Symbol)):
            raise Exception(
                'The only permitted types are: float, int, comlex, Operator, sympy.core.mul.Mul and sympy.core.mul.symbols.Symbol.')
        if not isinstance(other, Operator):
            other = Operator({2*self.n_modes*(0,): other})
        output_operator = Operator({x: self.get(x, 0) + other.get(x, 0) for x in set(self).union(other)})
        return output_operator

    def __mul__(self, other):
        if not isinstance(other, Operator):
            return self.__scalar_mul__(other)
        if self.n_modes != other.n_modes:
            raise Exception('Both operators must have the same number of modes.')
        if self.n_modes == 1:
            return self.__single_mode_mul__(other)
        else:
            return self.__multi_mode_mul__(other)

    def __scalar_mul__(self, scalar):
        output_op = Operator(n_modes=self.n_modes)
        for exponents, coeff in self.items():
            output_op[exponents] = coeff * scalar
        return output_op

    def __single_mode_mul__(self, other):
        op_final = Operator(n_modes=1)
        for exponents_1, coeff_1 in self.items():
            for exponents_2, coeff_2 in other.items():
                op_commutator = commutator_func(exponents_1[1], exponents_2[0])
                product_exponents = map(lambda key: (key[0] + exponents_1[0], key[1] + exponents_2[1]),
                                        op_commutator.keys())
                commutator_coeffs = op_commutator.values()
                op_product = Operator(n_modes=1)
                for key, value in zip(product_exponents, commutator_coeffs):
                    op_product[tuple(key)] = value
                op_product[(exponents_1[0] + exponents_2[0], exponents_1[1] + exponents_2[1])] = 1.0
                coeff_final = coeff_1 * coeff_2
                op_product *= coeff_final
                op_final += op_product
        return op_final

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        output_operator = self + (-1 * other)
        return output_operator

    def __rsub__(self, other):
        output_operator = other + (-1 * self)
        return output_operator

    def __radd__(self, other):
        return self.__add__(other)

    def __pow__(self, exponent):
        if not isinstance(exponent, numbers.Integral):
            raise Exception('Only integer exponents are allowed.')
        output_operator = Operator({2*self.n_modes*(0,): 1.0})
        for n in range(exponent):
            output_operator = output_operator * self
        return output_operator

    def copy(self):
        return Operator(spec=self)

    def dag(self):
        conjugated_dictionary = SortedDict()
        for exponents, coeff in self.items():
            conjugated_exponents = ()
            for mode_idx in range(len(exponents) // 2):
                conjugated_exponents += (exponents[2 * mode_idx + 1], exponents[2 * mode_idx])
            conjugated_dictionary[conjugated_exponents] = np.conjugate(coeff)
        return Operator(spec=conjugated_dictionary)

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            raise Exception('The key must be a tuple.')
        if len(key) != 2 * self.n_modes:
            raise Exception('The key length must be equal to twice the number of modes.')
        if not all(map(lambda el: isinstance(el, numbers.Integral), key)):
            raise Exception('All elements of the key must be integers.')
        super().__setitem__(key, value)

    def get_term(self, key):
        coeff = self[key]
        term = Operator({key: coeff})
        return term

    def __multi_mode_mul__(self, other):
        op_final = Operator(n_modes=self.n_modes)
        for exponents_1, coeff_1 in self.items():
            for exponents_2, coeff_2 in other.items():
                term_product = None
                for mode_idx in range(self.n_modes):
                    component_1 = Operator({exponents_1[2 * mode_idx:2 * (mode_idx + 1)]: 1.0})
                    component_2 = Operator({exponents_2[2 * mode_idx:2 * (mode_idx + 1)]: 1.0})
                    component_product = component_1 * component_2
                    if term_product is None:
                        term_product = component_product
                    else:
                        term_product = tensor(term_product, component_product)
                coeff_product = coeff_1 * coeff_2
                term_product *= coeff_product
                op_final += term_product
        return op_final
