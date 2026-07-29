// Copyright 2026 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Rust port of the `dora-openarm-evaluation` orchestration repository.
//!
//! Upstream is not a dora node package -- it has no `pyproject.toml` and
//! ships no console scripts. It is dataflow YAML plus two loose scripts,
//! [`src/local_policy_server.py`] and [`src/docker_policy_server.py`],
//! that sit behind the `AF_UNIX` socket the already-published
//! `dora-openarm-local-policy-server-rust` and
//! `dora-openarm-docker-policy-server-rust` node crates talk to. Those
//! two scripts load a `LeRobot` ACT checkpoint with `PyTorch` and are the
//! only place in the whole `dora-openarm-*` tree that does real machine
//! learning inference.
//!
//! This crate ports everything Rust *can* port byte-for-byte -- the
//! socket roles, the NDJSON protocol, Arrow IPC observation loading,
//! resolution detection, and image/state preprocessing -- and isolates
//! the one thing it cannot behind the [`policy::PolicyModel`] trait. See
//! the [`policy`] module docs for exactly what that boundary means and
//! why. Ported dataflow YAML lives under `dataflows/` at the repository
//! root, not in this crate, since dora dataflows are not Rust source.
//!
//! [`src/local_policy_server.py`]: https://github.com/enactic/dora-openarm-evaluation/blob/main/src/local_policy_server.py
//! [`src/docker_policy_server.py`]: https://github.com/enactic/dora-openarm-evaluation/blob/main/src/docker_policy_server.py
//!
//! # Module map
//!
//! - [`protocol`] -- the NDJSON request/response types and line
//!   encode/decode functions, server side.
//! - [`resolution`] -- `detect_resolution`, the byte-count-to-`(h, w)`
//!   heuristic.
//! - [`observation`] -- Arrow IPC FILE loading and typed field
//!   extraction.
//! - [`image_prep`] -- resize and `HWC` `u8` → `CHW` `f32` normalization.
//! - [`camera_map`] -- the fixed observation-field to model-input-key
//!   mapping.
//! - [`batch`] -- assembly of one observation into a model-ready batch.
//! - [`policy`] -- the [`policy::PolicyModel`] inference trait and its
//!   deterministic [`policy::MockPolicy`].
//! - [`server`] -- the request/response loop shared by both binaries.
//! - [`socket_role`] *(Unix only)* -- the local (bind/listen/accept) and
//!   docker (connect) socket roles.
//! - [`cli`] -- socket path argument resolution for each binary.

pub mod batch;
pub mod camera_map;
pub mod cli;
pub mod image_prep;
pub mod observation;
pub mod policy;
pub mod protocol;
pub mod resolution;
pub mod server;

#[cfg(unix)]
pub mod socket_role;

pub use batch::{BatchError, ModelBatch, build_batch};
pub use camera_map::{CAMERA_KEY_MAP, STATE_FIELD, STATE_MODEL_KEY};
pub use image_prep::{ImagePrepError, PreparedImage, prepare_image};
pub use observation::{ObservationError, extract_f32_list, extract_u8_list, load_observation};
pub use policy::{MockPolicy, PolicyModel};
pub use protocol::{
    ActionsResponse, CUTOFF_HZ, INTERVAL_NS, InferenceRequest, read_request, write_response,
};
pub use resolution::{UnresolvableResolution, detect_resolution};
pub use server::{ServerError, serve_connection};
