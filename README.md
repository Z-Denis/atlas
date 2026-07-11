# atlas

[![CI](https://github.com/Z-Denis/atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/Z-Denis/atlas/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust framework for variational methods on structured configuration spaces.

Atlas provides composable abstractions for configuration spaces, local spaces, homogeneous product spaces, probabilistic models, wave functions, sampling algorithms, and variational optimisation. Configuration spaces define admissible configurations. Local spaces are the primitive structured factors, and homogeneous product spaces lift them to repeated configurations. Discrete and continuous families are treated on the same footing. Above the configuration space, Atlas uses a `StateSpace` layer to distinguish ambient model families such as `Simplex` and `Hilbert`. Geometry will live above that layer and will be responsible for metrics and projection.

> **Status:** Early development. The API is not yet stable.
