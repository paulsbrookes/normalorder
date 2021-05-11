import numpy as np
from normalorder.operator.boson import Operator
import itertools

#tools based on https://arxiv.org/pdf/quant-ph/0109020.pdf


def find_bogoliubov_representation(hamiltonian):
    n_modes = hamiltonian.n_modes
    xi = np.zeros([n_modes, n_modes])
    eta = np.zeros([n_modes, n_modes])

    def key_gen_xi(i, j, n_modes=1):
        key = [0 for i in range(2 * n_modes)]
        key[2 * i] += 1
        key[2 * j + 1] += 1
        key = tuple(key)
        return key

    def key_gen_eta(i, j, n_modes=1):
        key = [0 for i in range(2 * n_modes)]
        key[2 * i] += 1
        key[2 * j] += 1
        key = tuple(key)
        return key

    for i, j in itertools.product(range(n_modes), range(n_modes)):

        key = key_gen_xi(i, j, n_modes)
        xi[i, j] = 0.5 * hamiltonian[key]

        key = key_gen_eta(i, j, n_modes)
        eta[i, j] = hamiltonian[key]
        if i != j:
            eta[i, j] *= 0.5

    return xi, eta


def generate_hamiltonian_from_bogoliubov_representation(xi, eta):
    n_modes = xi.shape[0]
    ops = []
    for i in range(n_modes):
        spec = [0 for j in range(2 * n_modes)]
        spec[2 * i + 1] = 1
        spec = [[1] + spec]
        ops.append(Operator(spec))

    hamiltonian = 0
    for i, j in itertools.product(range(n_modes), range(n_modes)):
        hamiltonian += xi[i, j] * ops[i].dag() * ops[j]
        hamiltonian += np.conjugate(xi[i, j]) * ops[i] * ops[j].dag()
        hamiltonian += eta[i, j] * ops[i].dag() * ops[j].dag()
        hamiltonian += np.conjugate(eta[i, j]) * ops[i] * ops[j]

    return hamiltonian
