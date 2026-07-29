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

//! Tests for `build_batch`: wiring `CAMERA_KEY_MAP` order and error
//! propagation from missing fields / missing declared image sizes.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, ListArray, StructArray};
use arrow::datatypes::Field;
use arrow::datatypes::{Float32Type, UInt8Type};
use dora_openarm_evaluation_rust::batch::{BatchError, build_batch};
use dora_openarm_evaluation_rust::camera_map::CAMERA_KEY_MAP;
use dora_openarm_evaluation_rust::observation::ObservationError;

/// A one-row observation `StructArray` with a `position` field and the
/// three camera fields `build_batch` reads, each holding a 3x4 (36-byte)
/// image -- the same tiny fixture `image_prep`'s tests use.
fn sample_observation(position: Vec<f32>) -> StructArray {
    let camera_bytes: Vec<u8> = (0_u8..36).collect();

    let position_array = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(
        position.into_iter().map(Some),
    )]);
    let camera_array = |bytes: &[u8]| {
        ListArray::from_iter_primitive::<UInt8Type, _, _>(vec![Some(
            bytes.iter().copied().map(Some).collect::<Vec<_>>(),
        )])
    };

    let mut fields: Vec<(Arc<Field>, ArrayRef)> = vec![(
        Arc::new(Field::new(
            "position",
            position_array.data_type().clone(),
            false,
        )),
        Arc::new(position_array),
    )];
    for &(arrow_key, _) in CAMERA_KEY_MAP {
        let array = camera_array(&camera_bytes);
        fields.push((
            Arc::new(Field::new(arrow_key, array.data_type().clone(), false)),
            Arc::new(array),
        ));
    }
    StructArray::from(fields)
}

fn image_sizes_at(height: u32, width: u32) -> HashMap<String, (u32, u32)> {
    CAMERA_KEY_MAP
        .iter()
        .map(|&(_, model_key)| (model_key.to_owned(), (height, width)))
        .collect()
}

#[test]
fn state_is_the_position_field_verbatim() {
    let observation = sample_observation(vec![1.0, 2.0, 3.0]);
    let image_sizes = image_sizes_at(3, 4);

    let batch = build_batch(&observation, &image_sizes).unwrap();
    assert_eq!(batch.state, vec![1.0, 2.0, 3.0]);
}

#[test]
fn images_are_built_in_camera_key_map_order() {
    let observation = sample_observation(vec![1.0]);
    let image_sizes = image_sizes_at(3, 4);

    let batch = build_batch(&observation, &image_sizes).unwrap();
    let model_keys: Vec<&str> = batch.images.iter().map(|(key, _)| key.as_str()).collect();
    assert_eq!(
        model_keys,
        vec![
            "observation.images.head_left",
            "observation.images.wrist_left",
            "observation.images.wrist_right",
        ]
    );
    for (_, image) in &batch.images {
        assert_eq!((image.height, image.width), (3, 4));
        assert_eq!(image.data.len(), 3 * 3 * 4);
    }
}

#[test]
fn missing_state_field_is_an_observation_error() {
    let camera_bytes: Vec<u8> = (0_u8..36).collect();
    let array = ListArray::from_iter_primitive::<UInt8Type, _, _>(vec![Some(
        camera_bytes.iter().copied().map(Some).collect::<Vec<_>>(),
    )]);
    let fields: Vec<(Arc<Field>, ArrayRef)> = vec![(
        Arc::new(Field::new(
            "camera_head_left",
            array.data_type().clone(),
            false,
        )),
        Arc::new(array),
    )];
    let observation = StructArray::from(fields);
    let image_sizes = image_sizes_at(3, 4);

    let error = build_batch(&observation, &image_sizes).unwrap_err();
    assert!(matches!(
        error,
        BatchError::Observation(ObservationError::MissingField("position"))
    ));
}

#[test]
fn missing_declared_image_size_is_a_batch_error() {
    let observation = sample_observation(vec![1.0]);
    let mut image_sizes = image_sizes_at(3, 4);
    image_sizes.remove("observation.images.wrist_right");

    let error = build_batch(&observation, &image_sizes).unwrap_err();
    assert!(matches!(
        error,
        BatchError::MissingImageSize("observation.images.wrist_right")
    ));
}
