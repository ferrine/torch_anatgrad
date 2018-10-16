import pytest
import numpy as np


@pytest.fixture('session')
def linear_regression():
    solution = dict(
        a=np.arange(1, 11, dtype='float32')[:, None],
        b=np.asarray(4., dtype='float32'),
    )
    np.random.seed(42)
    s = 0.01
    setup = dict(
        s=s,
        x=np.random.randn(1000, 10).astype(dtype='float32') * s,
    )
    setup['y'] = setup['x'] @ solution['a'] + solution['b']
    return dict(solution=solution, setup=setup)
