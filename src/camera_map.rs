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

//! The observation-field to model-input camera key mapping.
//!
//! Matches upstream's `CAMERA_KEY_MAP` exactly, including iteration
//! order: only 3 of the observer's 5 camera fields are consumed --
//! `camera_ceiling` and one of the two wrist/head pairings are not
//! forwarded to the policy. This is upstream's behavior, not an
//! omission in this port; see `dora-openarm-evaluation/src/local_policy_server.py:32-36`.

/// `(observation_field, model_input_key)` pairs, in upstream dict
/// insertion order.
pub const CAMERA_KEY_MAP: &[(&str, &str)] = &[
    ("camera_head_left", "observation.images.head_left"),
    ("camera_wrist_left", "observation.images.wrist_left"),
    ("camera_wrist_right", "observation.images.wrist_right"),
];

/// The observation field carrying the arm state vector.
///
/// Matches upstream's `observation["position"]`.
pub const STATE_FIELD: &str = "position";

/// The model input key the state vector is published under.
///
/// Matches upstream's `batch["observation.state"]`.
pub const STATE_MODEL_KEY: &str = "observation.state";
