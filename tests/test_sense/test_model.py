from normalorder.sense.model import Model


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
