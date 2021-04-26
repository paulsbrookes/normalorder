import scipy.optimize, scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calc_B(k, params):
    phi_i, phi_o = calc_boundary_phases(k, params)
    x_J = params['x_J']
    l = params['l']
    current_left = np.cos(k * (x_J + l) - phi_i)
    current_right = np.cos(k * (x_J - l) + phi_o)

    if np.isclose(current_left, 0.0) and np.isclose(current_right, 0.0):
        B = -1.0
    else:
        B = current_left / current_right

    return B


def find_k(k_initial, params):
    args = (params,)
    res = scipy.optimize.root(root_func, k_initial, args=args, method='hybr', tol=0.0)
    k = res.x[0]
    return k


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


def calc_u_l(x, k, l, phi_i):
    u = np.sin(k * (x + l) - phi_i)
    return u


def calc_u_r(x, k_prime, l, phi_o):
    u = np.sin(k_prime * (x - l) + phi_o)
    return u


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

    def __init__(self):
        self.k = None
        self.A = 1
        self.B = None
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

    def set_params(self, params):
        self.params = params

    def solve(self, k_initial):
        self.k = find_k(k_initial, self.params)
        self.velocity = 1 / np.sqrt(self.params['L_0'] * self.params['C_0'])
        self.frequency = self.velocity*self.k / (2*np.pi)
        self.phi_i, self.phi_o = calc_boundary_phases(self.k, self.params)
        self.B = calc_B(self.k, self.params)
        C_total = self.params['C_0'] * 2 * self.params['l']
        integral = scipy.integrate.quad(lambda x: self.calc_u(x) ** 2, -self.params['l'], self.params['l'])[0]
        self.A = np.sqrt(C_total / (integral*self.params['C_0']))
        self.Delta = self.B * calc_u_r(self.params['x_J'], self.k, self.params['l'], self.phi_o)
        self.Delta -= calc_u_l(self.params['x_J'], self.k, self.params['l'], self.phi_i)
        self.Delta *= self.A
        self.C_total = C_total
        self.L = 1/(C_total * (2*np.pi*self.frequency)**2)
        self.C_prime = C_total / self.Delta**2
        self.L_prime = self.L * self.Delta**2

    def calc_u(self, x):
        u = np.heaviside(self.params['x_J'] - x, 0.0) * calc_u_l(x, self.k, self.params['l'], self.phi_i)
        u += self.B * np.heaviside(x - self.params['x_J'], 1.0) \
             * calc_u_r(x, self.k, self.params['l'], self.phi_o)
        u *= self.A
        return u

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        x_array = np.linspace(-self.params['l'], self.params['l'], 201)
        u_series = pd.Series(self.calc_u(x_array), index=x_array)
        u_series.plot(ax=ax)
        return ax
