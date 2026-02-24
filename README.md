# Ultro - Unsupervised Learning Through Robust Optimization

A Python package for training a neural network in the unsupervised learning paradigm leveraging the numerical optimization framework. The idea is to include the network's parameters in the decision variables of the optimization problem.

## Overview

Ultro provides tools for learning neural network policies that approximate optimal MPC controllers for nonlinear dynamical systems. The package implements various MPC formulations including:

- **Multiple shooting with RK4 integration**
- **Direct collocation**
- **Input-based approximation**
- **Consensus-based optimization**

## Supported Systems

- Cart-pole system
- Inverted pendulum
- Extensible to other dynamical systems

## Installation

### From source (development mode)

```bash
cd ultro
pip install -e .
```

### With development dependencies

```bash
pip install -e ".[dev]"
```

## Package Structure

```
ultro/
|
├── src/           # Training scripts for MPC approximation
├── models/        # Dynamical system models
├── test/          # Comparison and testing scripts
├── models_nn/         # Saved neural network parameters
└── images/           # Generated plots and figures
```

## Usage

### As a Python package

```python
from ultro.src.inv_pend_MPC_approx_ms import InvertedPendulumMPCInputVar
from ultro.test.cart_pole_MPC_comparison import CartPoleMPCComparison

# Train a policy
trainer = InvertedPendulumMPCInputVar(
    layer_sizes=[2, 10, 1],
    batch_size=40,
    horizon=20
)
trainer.setup()
trainer.train(num_iterations=1000)

# Test and compare
comparison = CartPoleMPCComparison(
    layer_sizes=[4, 20, 1],
    horizon=20
)
comparison.generate_test_states(n_test=20)
comparison.run_comparison()
comparison.print_results()
comparison.plot_trajectories()
comparison.show_plots()
```

<!-- ### Command-line interface

```bash
# Train cart-pole policy
ultro-train-cartpole --layers 4 20 1 --horizon 20 --batch-size 40

# Train inverted pendulum policy
ultro-train-invpend --layers 2 10 1 --horizon 20 --batch-size 40

# Run tests
ultro-test --system cartpole --n-tests 20 -->
```

## Features

- **RK4 Integration**: High-accuracy numerical integration
- **CasADi Integration**: Efficient symbolic computation and automatic differentiation
- **Neural Network Policies**: Learn control policies using feedforward neural networks
- **Multiple MPC Formulations**: Support for various MPC problem formulations
- **Visualization**: Built-in plotting for trajectory comparison and error analysis
- **Modular Design**: Easy to extend with new systems and controllers

## Dependencies

- NumPy
- SciPy
- Matplotlib
- CasADi
- csnn (CasADi neural networks)
- PyYAML

## Development

### Running tests

```bash
pytest tests/
```

### Code formatting

```bash
black ultro/
isort ultro/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{ultro2026,
  title={Ultro: Unsupervised Learning Through Robust Optimization},
  author={Pietro},
  year={2026},
  url={https://github.com/piergo98/ultro}
}
```
