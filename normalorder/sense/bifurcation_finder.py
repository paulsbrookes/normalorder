from normalorder.sense.model import Model, convert_op_to_expr
import sympy
import numpy as np

np.random.seed(seed=0)
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import optimize, signal
from scipy.integrate import solve_ivp
import seaborn as sns


def fft_result(y, cutoff_time=0, columns=['a', 'b'], normalize=False):
    y_fft = np.fft.fft(y.loc[cutoff_time:].values.T).T
    y_fft = pd.DataFrame(y_fft, columns=columns)
    sample_timestep = y.index[1] - y.index[0]
    frequencies = np.fft.fftfreq(y_fft.shape[0], d=sample_timestep)
    y_fft.index = frequencies
    y_fft.sort_index(inplace=True)
    if normalize:
        y_fft /= y_fft.shape[0]
    return y_fft


def obtain_signal(sol_frame, cutoff_fraction=0.9):
    cutoff_time = sol_frame.index[-1] * cutoff_fraction
    fft_output = fft_result(sol_frame, cutoff_time=cutoff_time, columns=['a'], normalize=True)
    peak_1 = fft_output.iloc[fft_output.index.get_loc(0.01, method='nearest')].abs().iloc[0]
    peak_2 = fft_output.iloc[fft_output.index.get_loc(-0.01, method='nearest')].abs().iloc[0]
    combined = peak_1 ** 2 + peak_2 ** 2
    return combined


def c_4_objective_func(x, model):
    phi_ext = x[0]
    potential_params = {'phi_ext': phi_ext}
    model.set_potential_params(potential_params)
    delta_min_guess = 0.0
    model.find_delta_min(delta_min_guess=delta_min_guess)
    c_4 = model.c_func(4)
    return c_4


def f_J_func(phi_ext, f_J_1, f_J_2):
    f_J_tot = f_J_1 + f_J_2
    d = np.abs(f_J_1 - f_J_2) / f_J_tot
    f_J = f_J_tot * np.abs(np.cos(phi_ext * np.pi) * np.sqrt(1 + d ** 2 * np.tan(phi_ext * np.pi) ** 2))
    return f_J


class BifurcationFinder:

    def __init__(self, model):
        self.model = model
        self.phi_ext = None
        self.phi_ext_array = None
        self.c_4_array = None
        self.res = None
        self.spectrum = None
        self.epsilon_limits = None
        self.objective_series = None
        self.mode_name = self.model.mode_names[0]
        self.threshold = None
        self.epsilon_bifurcation = None
        self.f_d_bifurcation = None

    def c_4_objective_func(self, phi_ext, delta_min_guess=0.0):
        potential_params = self.model.potential_params
        potential_params['phi_ext'] = phi_ext[0]
        self.model.set_potential_params(potential_params)
        self.model.find_delta_min(delta_min_guess=delta_min_guess)
        c_4 = self.model.c_func(4)
        return c_4

    def eliminate_c_4(self, phi_ext_initial=0.1):
        x_initial = [phi_ext_initial]
        res = optimize.root(self.c_4_objective_func, x_initial)
        self.res = res
        potential_params = self.model.potential_params
        potential_params['phi_ext'] = self.res.x[0]
        self.model.set_potential_params(potential_params)

    def calculate_c_4_dependence(self, phi_ext_array=np.linspace(0.0, 1.0, 21), delta_min_guess=0.0):
        self.phi_ext_array = phi_ext_array
        self.c_4_array = np.zeros(self.phi_ext_array.shape)
        phi_ext_original = self.model.potential_params['phi_ext']
        potential_params = self.model.potential_params
        for idx, phi_ext in enumerate(self.phi_ext_array):
            potential_params['phi_ext'] = phi_ext
            self.model.set_potential_params(potential_params)
            self.model.find_delta_min(delta_min_guess=delta_min_guess)
            c_4 = self.model.c_func(4)
            self.c_4_array[idx] = c_4
        potential_params['phi_ext'] = phi_ext_original
        self.model.set_potential_params(potential_params)

    def plot_c_4_dependence(self, axes=None, phi_ext_array=None):
        if phi_ext_array is not None:
            self.calculate_c_4_dependence(phi_ext_array=phi_ext_array)
        elif self.c_4_array is None:
            self.calculate_c_4_dependence(phi_ext_array=np.linspace(0.0, 1.0, 21))
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.plot(self.phi_ext_array, self.c_4_array)
        if self.res:
            axes.axvline(self.res.x[0])
        axes.axhline(0)
        axes.set_xlabel(r'$\phi_{ext}$')
        axes.set_ylabel(r'$c_4$')

    def calculate_spectrum(self, f_d_array=None, epsilon=None):
        if len(self.model.modes) != 1:
            raise Exception('The bifurcation finder is only set up for models with a single mode.')
        mode_name = self.model.mode_names[0]
        kappa = self.model.decay_rates[mode_name]

        if f_d_array is None:
            f_d_array = self.model.modes[mode_name].frequency + np.linspace(-1, 1, 2001) * kappa
        if epsilon is None:
            epsilon = kappa

        results_array = np.zeros([f_d_array.shape[0], 2], dtype=complex)
        for idx, f_d in enumerate(f_d_array):
            results_array[idx] = self.model.dalpha_func(f_d, epsilon, kappa)

        self.spectrum = pd.DataFrame(results_array, index=f_d_array, columns=['dalpha', 'alpha'])

    def plot_spectrum(self, axes=None, f_d_array=None, epsilon=None):
        if (f_d_array is not None) or (epsilon is not None) or (self.spectrum is None):
            self.calculate_spectrum(f_d_array=f_d_array, epsilon=epsilon)
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        self.spectrum['alpha'].abs().plot(ax=axes)

    def plot_objective(self, epsilon_limits=[1.0, 50.0], n_points=21, axes=None):

        epsilon_array = np.linspace(*epsilon_limits, n_points)
        objective_array = np.zeros(epsilon_array.shape[0])

        x0 = self.model.modes[self.mode_name].frequency
        kappa = self.model.decay_rates[self.mode_name]

        for idx, epsilon in tqdm(enumerate(epsilon_array)):
            objective_array[idx] = self.objective_wrapper(epsilon, x0, kappa)

        self.objective_series = pd.Series(objective_array, index=epsilon_array)

        if axes is None:
            fig, self.axes = plt.subplots(1, 1, figsize=(10, 6))
        else:
            self.axes = axes

        self.objective_series.plot(ax=self.axes)
        self.axes.set_xlabel(r'$\epsilon$')
        self.axes.set_ylabel('Objective')

        if self.epsilon_bifurcation is not None:
            self.axes.axhline(self.threshold)
            self.axes.axvline(self.epsilon_bifurcation / kappa)

    def gradient_objective_func(self, x, epsilon, kappa):
        f_d = x[0]
        dalpha, alpha = self.model.dalpha_func(f_d, epsilon, kappa)
        return 1 / np.abs(dalpha)

    def objective_wrapper(self, epsilon, x0, kappa=1, offset=0):
        res = optimize.minimize(self.gradient_objective_func, x0=x0, args=(epsilon * kappa, kappa),
                                method='Nelder-Mead')
        return np.log10(res.fun) - offset

    def find_bifurcation(self, epsilon_limits=[1.0, 50.0], threshold=0):
        self.epsilon_limits = epsilon_limits
        self.threshold = threshold

        x0 = self.model.modes[self.mode_name].frequency
        kappa = self.model.decay_rates[self.mode_name]

        epsilon_bifurcation = optimize.bisect(self.objective_wrapper, *self.epsilon_limits,
                                              args=(x0, kappa, self.threshold)) * kappa

        self.epsilon_bifurcation = epsilon_bifurcation
        res = optimize.minimize(self.gradient_objective_func, x0=x0, args=(self.epsilon_bifurcation, kappa),
                                method='Nelder-Mead')

        self.f_d_bifurcation = res.x[0]

    def run_simulation(self):

        self.model.set_drive_params(self.f_d_bifurcation, self.epsilon_bifurcation)
        timescale = 1e-9  # time unit = 1 nanosecond
        self.model.generate_eom(timescale=timescale)

        f_s = 1e4 * timescale  # 1 kHz

        def delta_phi_squid_func(t):
            A_s = 50e-9
            return A_s * np.cos(2 * np.pi * f_s * t)

        def ode_wrapper(t, y):
            delta_phi_squid = delta_phi_squid_func(t)
            dy = self.model.eom(y, delta_phi_squid)
            return dy

        mode_name = self.model.mode_names[0]
        kappa = self.model.decay_rates[mode_name]
        t_span = [0, 10000 / (timescale * kappa)]
        t_eval = np.linspace(*t_span, 100001)
        y_0 = np.array([0.0], dtype=complex)
        tol = 1e-5
        sol = solve_ivp(ode_wrapper, t_span, y_0, t_eval=t_eval, rtol=tol, atol=tol)
        sol_frame = pd.DataFrame(sol.y.T, index=sol.t * timescale, columns=self.model.mode_names)
        sol_frame.index *= 1e6

        self.sol_frame = sol_frame


def task(potential_params, resonator_params, Q, phi_squid, phi_ext_initial=-0.2, delta_min_guess=0.5, phi_ext=None):
    threshold = 0.5
    order = 8
    n = 3
    delta_sym, f_J_sym, phi_ext_sym, f_J_1_sym, f_J_2_sym, phi_squid_sym = sympy.symbols(
        'delta f_J phi_ext f_J_1 f_J_2 phi_squid')
    potential_param_symbols = {'f_J': f_J_sym, 'phi_ext': phi_ext_sym, 'f_J_1': f_J_1_sym, 'f_J_2': f_J_2_sym,
                               'phi_squid': phi_squid_sym}
    potential_expr = - f_J_1_sym * sympy.cos(2 * sympy.pi * delta_sym) - f_J_2_sym * sympy.cos(
        2 * sympy.pi * (delta_sym + phi_squid_sym)) - f_J_sym * n * sympy.cos(
        2 * sympy.pi * (delta_sym - phi_ext_sym) / n)

    phi_ext_initial = [phi_ext_initial]

    f_J_1 = potential_params['f_J_1']
    f_J_2 = potential_params['f_J_2']
    f_J = potential_params['f_J']

    alpha = f_J_func(phi_squid, f_J_1, f_J_2) / f_J

    model = Model()
    model.set_order(order)
    model.set_potential(potential_expr, potential_param_symbols)

    if phi_ext is None:
        model.set_potential_params(potential_params)
        res = optimize.root(c_4_objective_func, phi_ext_initial, args=(model,))
        phi_ext = res.x[0]

    potential_params = {'phi_ext': phi_ext}
    model.set_potential_params(potential_params)
    model.find_delta_min(delta_min_guess=delta_min_guess)
    model.calculate_L_J()
    model.set_resonator_params(**resonator_params)
    harmonic_numbers = np.array([1])
    model.set_modes(names=['a'], harmonic_numbers=harmonic_numbers)
    model.generate_hamiltonian(drive=True, potential_variables=['phi_squid'])
    kappa_a = model.modes['a'].frequency / Q
    decay_rates = {'a': kappa_a}
    model.set_decay_rates(decay_rates)
    model.generate_lindblad_ops()
    model.generate_eom_ops()
    model.generate_eom_exprs()
    model.generate_dalpha_func()

    bf = BifurcationFinder(model)
    bf.Q = Q
    bf.alpha = alpha

    try:
        bf.find_bifurcation(epsilon_limits=[0.001, 50.0], threshold=threshold)
        bf.run_simulation()
    except:
        print('Failure.')

    return bf


def extract_results(bf):
    results = {'potential_params': bf.model.potential_params,
               'drive_params': bf.model.drive_params,
               'resonator_params': bf.model.resonator_params,
               'mode_frequencies': bf.model.mode_frequencies,
               'signal': obtain_signal(bf.sol_frame),
               'decay_rates': bf.model.decay_rates,
               'Q': bf.Q,
               'alpha': bf.alpha}
    return results
