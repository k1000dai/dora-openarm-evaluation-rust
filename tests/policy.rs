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

//! Determinism and shape tests for `MockPolicy`.
//!
//! `MockPolicy` is not a claim about `LeRobot` ACT's behavior (see the
//! `policy` module docs) -- these tests only pin down its own documented
//! contract: a pure, deterministic function of the input state.

use dora_openarm_evaluation_rust::batch::ModelBatch;
use dora_openarm_evaluation_rust::camera_map::CAMERA_KEY_MAP;
use dora_openarm_evaluation_rust::policy::{MockPolicy, PolicyModel};

fn batch_with_state(state: Vec<f32>) -> ModelBatch {
    ModelBatch {
        state,
        images: Vec::new(),
    }
}

#[test]
fn with_known_resolution_declares_all_three_mapped_cameras() {
    let policy = MockPolicy::with_known_resolution(480, 640);
    let sizes = policy.image_sizes();

    assert_eq!(sizes.len(), CAMERA_KEY_MAP.len());
    for &(_, model_key) in CAMERA_KEY_MAP {
        assert_eq!(sizes.get(model_key), Some(&(480, 640)));
    }
}

#[test]
fn infer_produces_a_10_row_16_column_chunk() {
    let policy = MockPolicy::with_known_resolution(480, 640);
    let batch = batch_with_state(vec![1.0, 2.0, 3.0]);

    let positions = policy.infer(&batch);
    assert_eq!(positions.len(), 10);
    for row in &positions {
        assert_eq!(row.len(), 16);
    }
}

#[test]
fn infer_echoes_state_truncated_and_zero_padded_into_every_row() {
    let policy = MockPolicy::with_known_resolution(480, 640);
    let batch = batch_with_state(vec![1.0, 2.0, 3.0]);

    let positions = policy.infer(&batch);
    let mut expected = vec![0.0_f32; 16];
    expected[0] = 1.0;
    expected[1] = 2.0;
    expected[2] = 3.0;

    for row in &positions {
        assert_eq!(row, &expected);
    }
}

#[test]
fn infer_is_deterministic() {
    let policy = MockPolicy::with_known_resolution(480, 640);
    let batch = batch_with_state(vec![0.5, -0.5, 2.0]);

    assert_eq!(policy.infer(&batch), policy.infer(&batch));
}

#[test]
fn custom_chunk_len_and_action_dim_are_honored() {
    let image_sizes = MockPolicy::with_known_resolution(1, 1).image_sizes();
    let policy = MockPolicy::new(image_sizes, 2, 4);
    let batch = batch_with_state(vec![9.0, 9.0, 9.0, 9.0, 9.0]);

    let positions = policy.infer(&batch);
    assert_eq!(positions, vec![vec![9.0, 9.0, 9.0, 9.0]; 2]);
}
