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

//! The model inference boundary.
//!
//! Upstream loads a `LeRobot` ACT checkpoint
//! (`enactic/act-openarm-2-cell-pick_up_cube_mujoco`) with `PyTorch` and
//! calls `policy.predict_action_chunk(batch)`. **No `LeRobot` or `PyTorch`
//! runtime exists in Rust**, and this crate does not attempt to
//! reimplement one -- doing so would mean re-deriving ACT's architecture,
//! trained weights, and numerics from scratch, which is out of scope for
//! a protocol/transport port and would silently diverge from upstream's
//! actual policy.
//!
//! Instead, model inference is isolated behind the [`PolicyModel`] trait.
//! Everything else in this crate -- the socket roles, the NDJSON
//! protocol, Arrow IPC observation loading, and image/state
//! preprocessing -- is real, tested, behavior-compatible Rust. Only the
//! step in the middle, turning a [`ModelBatch`] into action positions, is
//! pluggable.
//!
//! The [`MockPolicy`] shipped here is a deterministic stand-in used by
//! this crate's own golden tests and by the two server binaries when no
//! other adapter is wired in. It is **not** a `LeRobot` ACT
//! implementation and produces results with no relationship to trained
//! policy weights. Running this crate's binaries against a real robot or
//! trusting their output for evaluation would be a mistake; they exist
//! to prove the transport and preprocessing pipeline is correct up to
//! the inference boundary. See `README.md`, "The `LeRobot` boundary", for
//! how a real adapter would be wired in (e.g. an HTTP or subprocess
//! shim to the original Python `local_policy_server.py` /
//! `docker_policy_server.py`).

use std::collections::HashMap;

use crate::batch::ModelBatch;

/// Adapts a policy model to this crate's [`ModelBatch`] input and
/// `Vec<Vec<f32>>` action-chunk output.
///
/// Implementations decide what image size each mapped camera should be
/// resized to (mirroring upstream's `policy.config.input_features`
/// lookup) and how a batch becomes an action chunk. Neither this trait
/// nor its [`MockPolicy`] implementation performs real `LeRobot` ACT
/// inference -- see the module docs.
pub trait PolicyModel {
    /// The `(height, width)` this policy expects for each camera model
    /// key in [`crate::camera_map::CAMERA_KEY_MAP`].
    fn image_sizes(&self) -> HashMap<String, (u32, u32)>;

    /// Infers an action chunk from `batch`.
    ///
    /// Returns one row per action step, matching upstream's
    /// `actions.squeeze(0).cpu().numpy().tolist()`.
    fn infer(&self, batch: &ModelBatch) -> Vec<Vec<f32>>;
}

/// A deterministic, non-ML stand-in for a real [`PolicyModel`].
///
/// Ignores every camera image and echoes the observation's state vector,
/// truncated or zero-padded to `action_dim`, as every row of a
/// `chunk_len`-row action chunk. This makes its output a pure,
/// deterministic function of the input state, which is what this
/// crate's golden tests assert against -- not a claim about what a real
/// policy would output.
#[derive(Debug, Clone)]
pub struct MockPolicy {
    image_sizes: HashMap<String, (u32, u32)>,
    chunk_len: usize,
    action_dim: usize,
}

impl MockPolicy {
    /// Builds a mock policy that expects `image_sizes` for its camera
    /// inputs and produces `chunk_len` rows of `action_dim` floats per
    /// inference.
    #[must_use]
    pub fn new(
        image_sizes: HashMap<String, (u32, u32)>,
        chunk_len: usize,
        action_dim: usize,
    ) -> Self {
        Self {
            image_sizes,
            chunk_len,
            action_dim,
        }
    }

    /// Builds a mock policy whose camera inputs all expect one of
    /// [`crate::resolution`]'s known resolutions (so preparing an image
    /// already at that resolution never triggers a resize), producing a
    /// 10-row, 16-column action chunk -- the shape upstream's ACT
    /// checkpoint uses.
    #[must_use]
    pub fn with_known_resolution(height: u32, width: u32) -> Self {
        let image_sizes = crate::camera_map::CAMERA_KEY_MAP
            .iter()
            .map(|&(_, model_key)| (model_key.to_owned(), (height, width)))
            .collect();
        Self::new(image_sizes, 10, 16)
    }
}

impl PolicyModel for MockPolicy {
    fn image_sizes(&self) -> HashMap<String, (u32, u32)> {
        self.image_sizes.clone()
    }

    fn infer(&self, batch: &ModelBatch) -> Vec<Vec<f32>> {
        let mut row = vec![0.0_f32; self.action_dim];
        for (index, slot) in row.iter_mut().enumerate() {
            *slot = batch.state.get(index).copied().unwrap_or(0.0);
        }
        vec![row; self.chunk_len]
    }
}
