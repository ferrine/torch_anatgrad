import torch
import torch.nn as nn
from torch_anatgrad import KFACOptimizer


def test_linear_regression(linear_regression):
    linreg_ = nn.Linear(10, 1)
    linreg = nn.Sequential(linreg_)
    x, y = linear_regression['setup']['x'], linear_regression['setup']['y']
    x, y = torch.Tensor(x), torch.Tensor(y)
    a = torch.Tensor(linear_regression['solution']['a'])
    b = torch.Tensor(linear_regression['solution']['b'])
    loss = nn.MSELoss()

    def loss_closure():
        return loss(linreg(x), y)

    def fisher_closure():
        pred = linreg(x)
        ys = pred + linear_regression['setup']['s'] * pred.new(pred.size()).normal_()
        return loss(pred, ys.detach())

    opt = KFACOptimizer(linreg)
    for i in range(1000):
        opt.step(loss_closure, fisher_closure)

    assert torch.allclose(linreg_.weight.data, a.data, atol=1e-3)
    assert torch.allclose(linreg_.bias.data, b.data, atol=1e-3)
