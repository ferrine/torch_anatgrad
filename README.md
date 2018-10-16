# torch_anatgrad
Accelerating Natural Gradient with Higher-Order Invariance https://arxiv.org/abs/1803.01273

# User installation
You can install latest version using pip+github
```
pip install git+https://github.com/ferrine/torch_anatgrad.git
```

# Development Installation
If you start to develop and contribute please follow the following instructions

1. Clone the repository
```
git clone git@github.com:ferrine/torch_anatgrad.git
cd torch_anatgrad
```

2. Create your personal copy of Makefile
```
cp Makefile.template Makefile
```

3. Install Development Environment
```
make install-dev
```
4. I've prepared codestyle precommit checks, you might find useful to use `make pep8-inplace` if codestyle diff is printed instead of successful commit
Some other useful commands are
```
make pep8       # just ckecks pep8 not inplace
make codestyle  # pylint + pep8
make tests      # runs tests
```

# Acknowledgments
Thanks to @ikostrikov who kindly provided us with base kfac implementation