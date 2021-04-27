import numpy as np
from sortedcontainers import SortedDict
import functools
from string import ascii_lowercase


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


def multiply_dictionary_by_scalar(dictionary, scalar):
    output_dictionary = SortedDict()
    for key, value in dictionary.items():
        output_dictionary[key] = value * scalar
    return output_dictionary


class Operator:

    def __init__(self, spec=None, modes=None):

        if isinstance(modes, str):
            modes = [modes]
        if modes is None:
            n_modes = (np.array(spec).shape[1] - 1) // 2
            modes = [ascii_lowercase[i] for i in range(n_modes)]
        self.modes = modes

        ladder_op_names = []
        for l in modes:
            ladder_op_names += [l + '_dag', l]
        self.ladder_op_names = ladder_op_names

        columns = ['coeff'] + ladder_op_names

        if isinstance(spec, SortedDict):
            self.data = spec
        elif isinstance(spec, dict):
            self.data = SortedDict(spec)
        elif isinstance(spec, (np.ndarray, list, tuple)):
            self.data = SortedDict()
            for el in spec:
                self.data[tuple(el[1:])] = el[0]
        else:
            raise Exception('Cannot handle spec input of type ' + type(spec) + '.')

    def __add__(self, other):
        if not isinstance(other, Operator):
            other = Operator([[other] + [0, 0] * len(self.modes)], modes=self.modes)
        combined_data = add_dictionaries(self.data, other.data)
        output_operator = Operator(combined_data, modes=self.modes)
        return output_operator

    def __mul__(self, other):
        if isinstance(other, Operator):
            output_dictionary = multiply_dictionaries(self.data, other.data)
        else:
            output_dictionary = multiply_dictionary_by_scalar(self.data, other)
        output_operator = Operator(spec=output_dictionary, modes=self.modes)
        return output_operator

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
        return Operator(spec=self.data, modes=self.modes)

    def dag(self):
        conjugated_dictionary = SortedDict()
        for exponents, coeff in self.data.items():
            conjugated_exponents = ()
            for mode_idx in range(len(exponents) // 2):
                conjugated_exponents += (exponents[2 * mode_idx + 1], exponents[2 * mode_idx])
            conjugated_dictionary[conjugated_exponents] = np.conjugate(coeff)
        return Operator(spec=conjugated_dictionary, modes=self.modes)