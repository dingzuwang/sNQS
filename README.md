# sNQS: Smooth Neural Quantum States for Real-Time Evolution

This repository provides the implementation of the **s-NQS** (smooth Neural Quantum States) method. 

s-NQS introduces a continuous-time variational ansatz for real-time quantum dynamics using Chebyshev interpolation of neural network parameters. 
The method enables stable, global optimization of neural quantum states via Monte Carlo sampling.

## Features

- Real-time evolution of quantum many-body systems via NQS
- Smooth parametrization using Chebyshev basis with global optimization
- Monte Carlo sampling with Metropolis-Hastings algorithm
- Optimizer: AdamW with PyTorch backend

## Repository Structure

- `model.py` — Defines system Hamiltonians
- `rbm.py` — Restricted Boltzmann Machine architecture
- `snqs.py` — s-NQS evolution with Chebyshev basis
- `sampler.py` — MCMC sampler implementation
- `utils.py` — Utility functions
- `vmc.py` — Variational Monte Carlo utilities
- `test_*.py` — Unit tests

## Dependencies

- Python 3.12+
- NumPy
- PyTorch
- Matplotlib

## Running the Code

To train an s-NQS model:

```bash
python main.py
