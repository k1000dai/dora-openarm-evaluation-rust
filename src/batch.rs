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

//! Assembly of one observation into a model-ready batch.
//!
//! Ports upstream's `observation_to_batch`: extract the state vector and
//! the three mapped camera images, resizing each to the size the policy
//! declares for its corresponding input feature.

use std::collections::HashMap;
use std::fmt;

use arrow::array::StructArray;

use crate::camera_map::{CAMERA_KEY_MAP, STATE_FIELD};
use crate::image_prep::{ImagePrepError, PreparedImage, prepare_image};
use crate::observation::{ObservationError, extract_f32_list, extract_u8_list};

/// One observation, prepared for a [`crate::policy::PolicyModel`]: the
/// state vector plus every mapped camera image, resized to that model's
/// declared input size.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelBatch {
    /// The arm state vector, straight from the observation's `position`
    /// field. Matches upstream's `batch["observation.state"]` before it
    /// is turned into a tensor.
    pub state: Vec<f32>,
    /// `(model_key, image)` pairs, in [`CAMERA_KEY_MAP`] order.
    pub images: Vec<(String, PreparedImage)>,
}

/// An error building a [`ModelBatch`] from an observation.
#[derive(Debug)]
pub enum BatchError {
    /// Reading a field from the observation failed.
    Observation(ObservationError),
    /// Preparing a camera image failed.
    ImagePrep(ImagePrepError),
    /// The policy did not declare an input size for a mapped model key.
    MissingImageSize(&'static str),
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Observation(error) => write!(f, "{error}"),
            Self::ImagePrep(error) => write!(f, "{error}"),
            Self::MissingImageSize(model_key) => {
                write!(f, "no declared input size for model key '{model_key}'")
            }
        }
    }
}

impl std::error::Error for BatchError {}

impl From<ObservationError> for BatchError {
    fn from(error: ObservationError) -> Self {
        Self::Observation(error)
    }
}

impl From<ImagePrepError> for BatchError {
    fn from(error: ImagePrepError) -> Self {
        Self::ImagePrep(error)
    }
}

/// Builds a [`ModelBatch`] from `observation`, resizing each mapped
/// camera image to the `(height, width)` `image_sizes` declares for its
/// model key.
///
/// Mirrors upstream's `observation_to_batch`, including the fixed
/// [`CAMERA_KEY_MAP`] field/image ordering.
///
/// # Errors
///
/// Returns an error if the state field or any mapped camera field is
/// missing or malformed, if an image cannot be resized, or if
/// `image_sizes` has no entry for a mapped model key.
#[allow(clippy::implicit_hasher)]
pub fn build_batch(
    observation: &StructArray,
    image_sizes: &HashMap<String, (u32, u32)>,
) -> Result<ModelBatch, BatchError> {
    let state = extract_f32_list(observation, STATE_FIELD)?;

    let mut images = Vec::with_capacity(CAMERA_KEY_MAP.len());
    for &(arrow_key, model_key) in CAMERA_KEY_MAP {
        let raw = extract_u8_list(observation, arrow_key)?;
        let &(target_h, target_w) = image_sizes
            .get(model_key)
            .ok_or(BatchError::MissingImageSize(model_key))?;
        let prepared = prepare_image(&raw, target_h, target_w)?;
        images.push((model_key.to_owned(), prepared));
    }

    Ok(ModelBatch { state, images })
}
