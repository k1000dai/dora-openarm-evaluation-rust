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
//! order. The policy consumes all five camera fields emitted by the
//! observer, including the ceiling and right head cameras introduced by
//! the current upstream evaluation policy.

/// `(observation_field, model_input_key)` pairs, in upstream dict
/// insertion order.
pub const CAMERA_KEY_MAP: &[(&str, &str)] = &[
    ("camera_ceiling", "observation.images.ceiling"),
    ("camera_head_left", "observation.images.head_left"),
    ("camera_head_right", "observation.images.head_right"),
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
