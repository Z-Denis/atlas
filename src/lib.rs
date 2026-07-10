//! Atlas is a Rust framework for variational methods built around a minimal
//! space core and Burn for backend execution.
//!
//! The core abstractions are:
//!
//! - `Space` for admissibility, flat sample size, and scalar type chosen by the space
//! - `ViewSpace` for optional zero-copy structured access to flat samples
//! - `Samples<B, D, K>` for Burn-backed contiguous storage with batch axes
//! - `LogDensity` for sampler-facing Burn tensor evaluation
//! - `Spin` and `SpinSpace` for homogeneous spin domains
//! - `random_state()` on homogeneous product spaces for chain initialization
//! - `sample()` and `samples` on `VariationalState` for NetKet-like sampling
//! - `Proposal`, `LogDensityBatch`, `LocalProposal`, `Metropolis`,
//!   `SamplerState`, and `VariationalState` for
//!   Burn-native sampling
//!
//! The crate keeps spaces as the domain of the models. Flat storage is
//! canonical, while structure is recovered only through borrowed views when a
//! concrete space provides one. Homogeneous product spaces may also seed
//! batched chain states through `random_state()`. Burn owns backend selection,
//! tensor parallelism, and device execution. Variational states expose a
//! `sample()` method and a `samples` buffer, mirroring the NetKet user
//! interface.

pub mod layout;
pub mod model;
pub mod observable;
pub mod optimizer;
pub mod sampler;
pub mod space;
mod utils;

pub use layout::Layout;
pub use sampler::{
    GaussianProposal, LocalProposal, LogDensityBatch, Metropolis, Proposal, SamplerState,
    VariationalState,
};
pub use space::{
    ContinuousSpace, HomogeneousProductSpace, HomogeneousSpace, LogDensity, Particles, Samples,
    Space, Spin, SpinSpace, ViewSpace,
};
