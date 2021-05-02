import scipy.optimize, scipy.integrate
from autograd import numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import pandas as pd


def root_func(k, params):
    x_J = params['x_J']
    l = params['l']
    L_0 = params['L_0']
    L_J = params['L_J']
    phi_i, phi_o = calc_boundary_phases(k,params)
    theta_l = k * (x_J + l) - phi_i
    theta_r = k * (x_J - l) + phi_o
    out = (np.sin(theta_r) * np.cos(theta_l) - np.sin(theta_l) * np.cos(theta_r)) * L_0 - k * L_J * np.cos(
        theta_r) * np.cos(theta_l)

    return out


def calc_boundary_phases(k, params):
    l = params['l']
    L_0 = params['L_0']
    C_0 = params['C_0']
    C_i = params['C_i']
    C_o = params['C_o']
    Z_0 = np.sqrt(L_0 / C_0)
    velocity = 1 / np.sqrt(L_0 * C_0)
    omega = velocity * k
    if not np.isclose(C_i / (2 * l * C_0), 0.0):
        phi_i = np.arctan(1 / np.abs(Z_0 * omega * C_i))
    else:
        phi_i = np.pi / 2
    if not np.isclose(C_o / (2 * l * C_0), 0.0):
        phi_o = np.arctan(1 / np.abs(Z_0 * omega * C_o))
    else:
        phi_o = np.pi / 2
    return phi_i, phi_o


class Mode:

    def __init__(self, eps=1e-10):
        self.k = None
        self.A = 1.0
        self.B = 1.0
        self.phi_i = None
        self.phi_o = None
        self.Delta = None
        self.params = None
        self.velocity = None
        self.frequency = None
        self.C_total = None
        self.L = None
        self.C_prime = None
        self.L_prime = None
        self.calc_u_l_grad = grad(self.calc_u_l)
        self.calc_u_r_grad = grad(self.calc_u_r)
        self.calc_u_grad = grad(self.calc_u)
        self.eps = eps

    def set_params(self, params):
        self.params = params

    def calc_B(self, k):
        current_left_scaled = self.calc_u_l_grad(self.params['x_J'])
        current_right_scaled = self.calc_u_r_grad(self.params['x_J'])

        if np.isclose(current_left_scaled, 0.0) and np.isclose(current_right_scaled, 0.0):
            B = -1.0
        else:
            B = current_left_scaled / current_right_scaled

        return B

    def solve(self, k_initial):
        self.k = self.find_k(k_initial)
        self.velocity = 1 / np.sqrt(self.params['L_0'] * self.params['C_0'])
        self.frequency = self.velocity*self.k / (2*np.pi)
        self.phi_i, self.phi_o = calc_boundary_phases(self.k, self.params)
        self.B = self.calc_B(self.k)
        C_total = self.params['C_0'] * 2 * self.params['l'] + self.params['C_i'] + self.params['C_o']
        integral = scipy.integrate.quad(lambda x: self.calc_u(x) ** 2, -self.params['l'], self.params['l'],
                                        epsrel=self.eps, epsabs=self.eps)[0]
        self.A *= np.sqrt(C_total / (integral*self.params['C_0']))
        self.Delta = self.calc_u_r(self.params['x_J'])
        self.Delta -= self.calc_u_l(self.params['x_J'])
        self.C_total = C_total
        self.L = 1/(C_total * (2*np.pi*self.frequency)**2)
        self.C_prime = C_total / self.Delta**2
        self.L_prime = self.L * self.Delta**2

    def find_k(self, k_initial):
        args = (self.params,)
        res = scipy.optimize.root(root_func, k_initial, args=args, method='hybr', tol=self.eps)
        k = res.x[0]
        return k

    def calc_u(self, x):
        u = np.greater(self.params['x_J'] - x, 0.0) * self.calc_u_l(x)
        u += np.greater_equal(x - self.params['x_J'], 0.0) * self.calc_u_r(x)
        return u

    def calc_u_l(self, x):
        u = self.A * np.sin(self.k * (x + self.params['l']) - self.phi_i)
        return u

    def calc_u_r(self, x):
        u = self.A * self.B * np.sin(self.k * (x - self.params['l']) + self.phi_o)
        return u

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        x_array = np.linspace(-self.params['l'], self.params['l'], 201)
        u_series = pd.Series(self.calc_u(x_array), index=x_array)
        u_series.plot(ax=ax)
        return ax
