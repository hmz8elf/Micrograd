import math
import pytest
import torch

from micrograd import Value
from micrograd.nn import MLP

TOL = 1e-6


def finite_diff(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

UNARY_CASES = [
    ("pow2", lambda v: v ** 2),
    ("pow3", lambda v: v ** 3),
    ("pow_neg", lambda v: v ** -1),
    ("neg", lambda v: -v),
    ("relu", lambda v: v.relu()),
    ("tanh", lambda v: v.tanh()),
    ("exp", lambda v: v.exp()),
]


@pytest.mark.parametrize("name,op", UNARY_CASES, ids=[c[0] for c in UNARY_CASES])
@pytest.mark.parametrize("x0", [-1.7, 0.3, 2.5])
def test_unary_against_torch(name, op, x0):
    x = Value(x0)
    y = op(x)
    y.backward()

    xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    yt = op(xt) if name != "relu" else torch.relu(xt)
    yt.backward()

    assert abs(y.data - yt.item()) < TOL, f"{name}: forward value differs"
    assert abs(x.grad - xt.grad.item()) < TOL, f"{name}: gradient differs"


BINARY_CASES = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b),
    ("radd", lambda a, b: 2.0 + a + b),
    ("rsub", lambda a, b: 2.0 - a * b),
    ("rmul", lambda a, b: 3.0 * a * b),
    ("rdiv", lambda a, b: 1.0 / (a * b)),
]


@pytest.mark.parametrize("name,op", BINARY_CASES, ids=[c[0] for c in BINARY_CASES])
def test_binary_against_torch(name, op):
    a, b = Value(-1.3), Value(2.4)
    y = op(a, b)
    y.backward()

    at = torch.tensor(-1.3, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(2.4, dtype=torch.float64, requires_grad=True)
    yt = op(at, bt)
    yt.backward()

    assert abs(y.data - yt.item()) < TOL
    assert abs(a.grad - at.grad.item()) < TOL
    assert abs(b.grad - bt.grad.item()) < TOL



def _big_expression(a, b, tanh, relu):
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + relu(b + a)
    d = d + 3 * d + relu(b - a)
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    return g + tanh(a * 0.1)


def test_composite_against_torch():
    a, b = Value(-4.0), Value(2.0)
    g = _big_expression(a, b, tanh=lambda v: v.tanh(), relu=lambda v: v.relu())
    g.backward()

    at = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    gt = _big_expression(at, bt, tanh=torch.tanh, relu=torch.relu)
    gt.backward()

    assert abs(g.data - gt.item()) < TOL
    assert abs(a.grad - at.grad.item()) < TOL
    assert abs(b.grad - bt.grad.item()) < TOL


def _f(x):
    if isinstance(x, Value):
        return (x * x + 3 * x).tanh() * x.exp() / (1 + x ** 2)
    return math.tanh(x * x + 3 * x) * math.exp(x) / (1 + x ** 2)


@pytest.mark.parametrize("x0", [-2.0, -0.5, 0.7, 1.9])
def test_finite_differences(x0):
    x = Value(x0)
    y = _f(x)
    y.backward()
    assert abs(x.grad - finite_diff(_f, x0)) < 1e-5

def test_repr_returns_string():
    assert isinstance(repr(Value(1.0)), str)


def test_op_is_stored_in_op_field():
    out = Value(1.0) + Value(2.0)
    assert out._op == "+", "the operation must go in _op, not in label"
    assert (Value(1.0) * Value(2.0))._op == "*"


def test_mlp_forward_and_backward():
    model = MLP(3, [4, 4, 1])
    out = model([1.0, -2.0, 0.5])
    out = out[0] if isinstance(out, list) else out
    assert isinstance(out, Value)
    out.backward()
    assert any(p.grad != 0 for p in model.parameters())


def test_zero_grad():
    model = MLP(2, [3, 1])
    out = model([1.0, 2.0])
    out = out[0] if isinstance(out, list) else out
    out.backward()
    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())