//! Atlas is a Rust framework for variational methods built around a minimal
//! space core.
//!
//! The core abstractions are:
//!
//! - `Space` for admissibility, flat sample size, and primitive scalar type
//! - `ViewSpace` for optional zero-copy structured access to flat samples
//! - `Samples<T>` for canonical contiguous storage with batch axes
//! - `LogDensity` for sampler-facing evaluation
//! - `Spin` and `SpinSpace` for homogeneous spin domains
//!
//! The crate keeps spaces as the domain of the models. Flat storage is
//! canonical, while structure is recovered only through borrowed views when a
//! concrete space provides one.

pub mod model;
pub mod observable;
pub mod optimizer;
pub mod layout;
pub mod space;
pub mod sampler;

pub use space::{
    HomogeneousProductSpace, HomogeneousSpace, LogDensity, Samples, SamplesError, Space, Spin,
    SpinSpace, ViewSpace,
};
