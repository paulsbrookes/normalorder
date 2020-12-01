from normalorder.sense.model import Model
from sympy import symbols, cos
import numpy as np


def test_init():

    model = Model()

    assert model.order == 0
    assert isinstance(model.c_sym, tuple)
    assert len(model.c_sym) == 0
    assert isinstance(model.resonator_params,dict)


def test_set_order():

    model = Model()

    for order in [0, 1, 2, 1, 0]:
        model.set_order(order)
        assert model.order == order
        assert isinstance(model.c_sym, tuple)
        assert len(model.c_sym) == order + 2
        for m in range(order+1):
            model.g_expr_gen(m)


def test_set_resonator_params():

    model = Model()

    params = {'a': 1.0,
              'b': 'test'}
    model.set_resonator_params(params)
    assert model.resonator_params == params


def test_set_potential():

    model = Model()

    phi_sym, x, y = symbols('phi x y')
    potential_param_symbols = {'x': x,
                               'y': y}
    potential_expr = x*y*phi_sym
    model.set_potential(potential_expr, potential_param_symbols)
    assert isinstance(model.c_sym, tuple)
    assert model.potential_expr == potential_expr
    assert model.potential_param_symbols == potential_param_symbols

    potential_params = {'x': np.random.rand(),
                        'y': np.random.randn()}
    model.set_potential_params(potential_params)
    assert model.potential_params == potential_params


def test_cg_func():

    model = Model()

    phi_sym, phi_ext_sym, nu_sym, n_sym = symbols('phi phi_ext nu n')
    potential_param_symbols = {'phi_ext': phi_ext_sym,
                               'nu': nu_sym,
                               'n' : n_sym}
    potential_expr = -nu_sym * cos(phi_sym) - n_sym * cos((phi_ext_sym - phi_sym) / n_sym)
    n = 3
    phi_ext = 2*np.pi*np.random.rand()
    nu = np.random.rand()*0.9/n
    potential_params = {'phi_ext': phi_ext,
                        'n': n,
                        'nu': nu}
    order = 3

    model.set_order(order)
    model.set_potential(potential_expr, potential_param_symbols)
    model.set_potential_params(potential_params)

    for idx in range(order+2):
        phi = n*2*np.pi*np.random.rand()
        assert isinstance(model.c_func(idx, phi), float)

    for idx in range(order+1):
        phi = n*2*np.pi*np.random.rand()
        assert isinstance(model.g_func(idx, phi), float)


def test_find_phi_min():

    model = Model()

    phi_sym, phi_ext_sym, nu_sym, n_sym = symbols('phi phi_ext nu n')
    potential_param_symbols = {'phi_ext': phi_ext_sym,
                               'nu': nu_sym,
                               'n' : n_sym}
    potential_expr = -nu_sym * cos(phi_sym) - n_sym * cos((phi_ext_sym - phi_sym) / n_sym)
    n = 3
    phi_ext = 2*np.pi*np.random.rand()
    nu = np.random.rand()*0.9/n
    potential_params = {'phi_ext': phi_ext,
                        'nu': nu,
                        'n' : n}
    model.set_potential(potential_expr, potential_param_symbols)
    model.set_potential_params(potential_params)

    model.find_phi_min()
    assert np.isclose(model.c_func(1, model.phi_min), 0.0)

    x0 = n*2*np.pi*np.random.rand()
    model.find_phi_min(x0=x0)
    assert np.isclose(model.c_func(1, model.phi_min), 0.0)