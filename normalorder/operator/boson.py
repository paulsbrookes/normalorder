import numpy as np
from sortedcontainers import SortedDict
import functools
import sympy
from string import ascii_lowercase
from scipy.special import factorial


def multiply_components(exponents_1, exponents_2):
    dict_final = SortedDict()
    multiplied_components = multiply_components_raw(*exponents_1, *exponents_2)
    spec_dict = SortedDict()
    for element in multiplied_components:
        key = tuple(element[1:].astype(int))
        if key in spec_dict.keys():
            spec_dict[tuple(element[1:].astype(int))] += element[0]
        else:
            spec_dict[tuple(element[1:].astype(int))] = element[0]
    dict_final = add_dictionaries(dict_final, spec_dict)

    return dict_final


def tensor_product_dictionaries(dict_1, dict_2):
    output_dict = SortedDict()
    for key_1, coeff_1 in dict_1.items():
        for key_2, coeff_2 in dict_2.items():
            key_out = key_1 + key_2
            coeff_out = coeff_1 * coeff_2
            output_dict[key_out] = coeff_out
    return output_dict


def multiply_dictionaries(dict_1, dict_2):
    dict_final = SortedDict()
    for exponents_1, coeff_1 in dict_1.items():
        for exponents_2, coeff_2 in dict_2.items():
            component_product = None
            for subexponents_1, subexponents_2 in zip(np.array(exponents_1).reshape(-1, 2),
                                                      np.array(exponents_2).reshape(-1, 2)):
                test = multiply_components(subexponents_1, subexponents_2)
                if component_product is None:
                    component_product = test
                else:
                    component_product = tensor_product_dictionaries(component_product, test)
            dict_final = add_dictionaries(dict_final, component_product)
    return dict_final


def add_dictionaries(dict_1, dict_2):
    dict_final = SortedDict({x: dict_1.get(x, 0) + dict_2.get(x, 0)
                             for x in set(dict_1).union(dict_2)})
    return dict_final


def commutator_func(x, y):
    if not isinstance(x, int) or not isinstance(y, int) or x < 0 or y < 0:
        raise Exception('This function only accepts exponents which are positive integers or zero.')

    def coeff_func(x, y, l):
        coeff = factorial(x) * factorial(y) / (factorial(x - l) * factorial(y - l) * factorial(l))
        return coeff

    commutator_op = Operator(n_modes=1)
    for l in range(1, min(x, y) + 1):
        commutator_op[(y - l, x - l)] = coeff_func(x, y, l)

    return commutator_op


def tensor(op_1, op_2):
    op_final = Operator()
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
        if not isinstance(other, (float, int, complex, Operator, sympy.core.mul.Mul, sympy.core.symbol.Symbol)):
            raise Exception(
                'The only permitted types are: float, int, comlex, Operator, sympy.core.mul.Mul and sympy.core.mul.symbols.Symbol.')
        if not isinstance(other, Operator):
            other = Operator({tuple(0 for i in range(2 * self.n_modes)): other})
        output_operator = Operator({x: self.get(x, 0) + other.get(x, 0) for x in set(self).union(other)})
        return output_operator

    def __mul__(self, other):
        if not isinstance(other, Operator):
            return self.__scalar_mul__(other)
        if self.n_modes != other.n_modes:
            raise Exception('Both operators must have the same number of modes.')
        if self.n_modes == 1:
            return self.__single_mode_mul__(other)

        if isinstance(other, Operator):
            output_dictionary = multiply_dictionaries(self, other)
        else:
            output_dictionary = multiply_dictionary_by_scalar(self, other)
        output_operator = Operator(spec=output_dictionary)
        return output_operator

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
        assert isinstance(exponent, int)
        output_operator = 1
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
        super().__setitem__(key, value)


@functools.lru_cache(maxsize=512)
def comm_func_raw(m, n):
    if m == 0 or n == 0:
        return np.zeros([0, 3])
    else:
        term_1 = np.array([[n, n - 1, m - 1]])
        output_list = [term_1]
        term_2 = np.copy(comm_func_raw(m - 1, n - 1))
        term_2[:, 0] *= n
        output_list.append(term_2)
        term_3 = np.copy(comm_func_raw(m - 1, n))
        term_3[:, 2] += 1
        output_list.append(term_3)
        output_array = np.vstack(output_list)
        output_array.flags.writeable = False
        return output_array


def multiply_components_raw(k, l, m, n):
    output_list = [np.array([[1, k + m, l + n]])]
    commutator_terms = np.copy(comm_func_raw(l, m))
    commutator_terms[:, 1] += k
    commutator_terms[:, 2] += n
    output_list.append(commutator_terms)
    return np.vstack(output_list)