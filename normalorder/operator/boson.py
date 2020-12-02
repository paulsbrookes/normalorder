import pandas as pd
import numpy as np
from string import ascii_lowercase
import functools


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


def multiply_components(component_1, component_2, mode_name):
    assert component_1.shape[0] == 1 and component_2.shape[0] == 1

    columns = ['coeff', mode_name + '_dag', mode_name]

    k = component_1[mode_name + '_dag'].iloc[0]
    l = component_1[mode_name].iloc[0]
    coeff_1 = component_1['coeff'].iloc[0]

    m = component_2[mode_name + '_dag'].iloc[0]
    n = component_2[mode_name].iloc[0]
    coeff_2 = component_2['coeff'].iloc[0]

    assert k >= 0 and l >= 0 and m >= 0 and n >= 0

    c = coeff_1 * coeff_2

    output_raw = multiply_components_raw(k, l, m, n)
    output_raw[:, 0] *= c
    output = pd.DataFrame(output_raw, columns=columns)

    return output


class Operator:

    def __init__(self, spec=None, modes=None):

        if isinstance(modes,str):
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

        if isinstance(spec, list):
            self.data = pd.DataFrame(spec, columns=columns)
        elif isinstance(spec, pd.DataFrame):
            self.data = spec.copy()
            #self.data.columns = spec.columns

        self.consolidate()

    def __add__(self, other):
        if not isinstance(other, Operator):
            other = Operator([[other] + [0,0]*len(self.modes)],modes=self.modes)
        combined_data = pd.concat([self.data, other.data])
        output_operator = Operator(combined_data,modes=self.modes)
        return output_operator

    def __sub__(self, other):
        output_operator = self + (-1 * other)
        return output_operator

    def __rsub__(self, other):
        output_operator = other + (-1 * self)
        return output_operator

    def __mul__(self, other):
        if isinstance(other, Operator):
            output_operator = multiply_operators(self, other)
        else:
            output_operator = self.copy()
            output_operator.data['coeff'] *= other
        return output_operator

    def __rmul__(self, other):
        assert not isinstance(other, Operator)
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __pow__(self, exponent):
        assert isinstance(exponent, int)
        output_operator = 1
        for n in range(exponent):
            output_operator = output_operator * self
        return output_operator

    def copy(self):
        return Operator(self.data.copy())

    def consolidate(self):
        self.data = self.data.groupby(self.ladder_op_names)['coeff'].sum().reset_index()
        self.data = self.data[['coeff']+self.ladder_op_names]
        pass
        #mask = ~np.isclose(self.data['coeff'], 0)
        #self.data = self.data[mask]

    def dag(self):
        output_operator = self.copy()
        output_operator.data['coeff'] = np.conjugate(output_operator.data['coeff'])
        swap_dict = {}
        for l in self.modes:
            swap_dict[l] = l + '_dag'
            swap_dict[l + '_dag'] = l
        output_operator.data.rename(columns=swap_dict, inplace=True)
        output_operator.data = output_operator.data[self.data.columns]
        return output_operator


def tensor(op_1, op_2):
    coeffs_out = []
    components_out = []
    for idx in range(op_1.data.shape[0]):
        for jdx in range(op_2.data.shape[0]):
            components_out.append(
                pd.concat([op_1.data.drop(columns='coeff').iloc[idx], op_2.data.drop(columns='coeff').iloc[jdx]]))
            coeffs_out.append(op_1.data['coeff'].iloc[idx] * op_2.data['coeff'].iloc[jdx])
    op_out_data = pd.concat(components_out, axis=1).T
    op_out_data.index = np.arange(op_out_data.shape[0])
    op_out_data['coeff'] = coeffs_out
    op_out = Operator(op_out_data, modes=op_1.modes + op_2.modes)
    return op_out


def multiply_operators(op_1, op_2):
    output_components = []
    for idx in range(op_1.data.shape[0]):
        for jdx in range(op_2.data.shape[0]):
            output_components.append(
                multiply_terms(op_1.data.iloc[idx:idx + 1], op_2.data.iloc[jdx:jdx + 1], op_1.modes).data)
    output_operator = Operator(spec=pd.concat(output_components), modes=op_1.modes)
    return output_operator


def multiply_terms(term_1, term_2, modes):
    component_products = []
    for l in modes:
        component_1 = term_1[['coeff']+[l + '_dag', l]].copy()
        component_1['coeff'] = 1.0
        component_2 = term_2[['coeff']+[l + '_dag', l]].copy()
        component_2['coeff'] = 1.0
        out = multiply_components(component_1, component_2, l)
        component_product = Operator(out, modes=[l])
        component_products.append(component_product)

    tensored_products = term_1['coeff'].iloc[0] * term_2['coeff'].iloc[0] * component_products[0]
    #tensored_products = component_products[0]
    for idx in range(len(modes) - 1):
        tensored_products = tensor(tensored_products, component_products[idx + 1])

    return tensored_products
