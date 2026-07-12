# atlas

[![CI](https://github.com/Z-Denis/atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/Z-Denis/atlas/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust framework for variational methods on structured configuration spaces.

Atlas keeps a small core:

- `Space` defines admissible flat configurations.
- `LocalSpace` and `HomogeneousSpace` describe structured factors and their lift to a flat product space.
- `ViewSpace` exposes zero-copy structured views.
- `Model` exposes a single `log_value()` on backend tensors.
- `StateSpace` turns that model output into the sampler-facing `LogDensity`.
- `Operator` returns connected configurations and matrix elements for local estimators.
- `VariationalState` owns the model, sampler, state space, and collected samples.

Discrete and continuous families are treated on the same footing. `Simplex` and `Hilbert` are the initial `StateSpace` families. `Magnetization`, `IsingEnergy`, and `TransverseFieldIsing` are some example spin operators. Geometry will live above that layer and will be responsible for metrics and projection.

> **Status:** Early development. The API is not yet stable.
