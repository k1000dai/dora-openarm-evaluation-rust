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

//! Loading the observation `StructArray` from an Arrow IPC FILE.
//!
//! Mirrors upstream's
//! `pa.OSFile(...)` + `pa.ipc.open_file(...)` + `reader.get_batch(0).to_struct_array()[0]`:
//! the observation is always row 0 of a single-batch, single-row Arrow IPC
//! **file** (not the streaming format) written by the dora node side of
//! this protocol (`dora-openarm-local-policy-server-rust` /
//! `dora-openarm-docker-policy-server-rust`).

use std::fmt;
use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Float32Array, ListArray, StructArray, UInt8Array};
use arrow_ipc::reader::FileReader;

/// An error loading or reading fields from an observation.
#[derive(Debug)]
pub enum ObservationError {
    /// Opening or reading the Arrow IPC file failed.
    Io(std::io::Error),
    /// Decoding the Arrow IPC file failed.
    Arrow(arrow::error::ArrowError),
    /// The file contained no record batches.
    EmptyFile,
    /// The record batch had no rows.
    EmptyBatch,
    /// A required field was missing from the observation.
    MissingField(&'static str),
    /// A field existed but was not encoded as a `List<T>` array.
    NotAList(&'static str),
    /// A field's list values were not the expected element type.
    UnexpectedElementType(&'static str),
    /// A field's row-0 list value was null.
    NullValue(&'static str),
}

impl fmt::Display for ObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "failed to open observation file: {error}"),
            Self::Arrow(error) => write!(f, "failed to decode observation Arrow IPC file: {error}"),
            Self::EmptyFile => write!(f, "observation Arrow IPC file has no record batches"),
            Self::EmptyBatch => write!(f, "observation record batch has no rows"),
            Self::MissingField(field) => write!(f, "observation has no '{field}' field"),
            Self::NotAList(field) => write!(f, "observation field '{field}' is not a list array"),
            Self::UnexpectedElementType(field) => {
                write!(
                    f,
                    "observation field '{field}' has an unexpected element type"
                )
            }
            Self::NullValue(field) => write!(f, "observation field '{field}' row 0 is null"),
        }
    }
}

impl std::error::Error for ObservationError {}

impl From<std::io::Error> for ObservationError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<arrow::error::ArrowError> for ObservationError {
    fn from(error: arrow::error::ArrowError) -> Self {
        Self::Arrow(error)
    }
}

/// Opens `path` as an Arrow IPC FILE and returns row 0 of its first
/// record batch as a [`StructArray`].
///
/// # Errors
///
/// Returns an error if the file cannot be opened, is not a valid Arrow
/// IPC file, contains no batches, or the first batch has no rows.
pub fn load_observation(path: &Path) -> Result<StructArray, ObservationError> {
    let file = File::open(path)?;
    let mut reader = FileReader::try_new(file, None)?;
    let batch = reader.next().ok_or(ObservationError::EmptyFile)??;
    if batch.num_rows() == 0 {
        return Err(ObservationError::EmptyBatch);
    }
    Ok(StructArray::from(batch))
}

/// Extracts row 0 of the `field` column as a flat `f32` vector.
///
/// Mirrors upstream's `observation[field].values.to_numpy().astype(np.float32)`:
/// `field` must be a `List<Float32>` (or a numeric list upstream would
/// cast to `float32`) whose row-0 value is the returned vector.
///
/// # Errors
///
/// Returns an error if the field is missing, is not a list array, its
/// row-0 value is null, or its element type cannot be read as `f32`.
pub fn extract_f32_list(
    observation: &StructArray,
    field: &'static str,
) -> Result<Vec<f32>, ObservationError> {
    let column = observation
        .column_by_name(field)
        .ok_or(ObservationError::MissingField(field))?;
    let list = column
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or(ObservationError::NotAList(field))?;
    if list.is_null(0) {
        return Err(ObservationError::NullValue(field));
    }
    let values = list.value(0);
    let floats = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or(ObservationError::UnexpectedElementType(field))?;
    Ok(floats.values().to_vec())
}

/// Extracts row 0 of the `field` column as a flat `u8` vector.
///
/// Mirrors upstream's `observation[field].values.to_numpy().astype(np.uint8)`:
/// `field` must be a `List<UInt8>` whose row-0 value is the returned
/// vector -- the raw (possibly JPEG-encoded) camera bytes.
///
/// # Errors
///
/// Returns an error if the field is missing, is not a list array, its
/// row-0 value is null, or its element type is not `u8`.
pub fn extract_u8_list(
    observation: &StructArray,
    field: &'static str,
) -> Result<Vec<u8>, ObservationError> {
    let column = observation
        .column_by_name(field)
        .ok_or(ObservationError::MissingField(field))?;
    let list = column
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or(ObservationError::NotAList(field))?;
    if list.is_null(0) {
        return Err(ObservationError::NullValue(field));
    }
    let values = list.value(0);
    let bytes = values
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or(ObservationError::UnexpectedElementType(field))?;
    Ok(bytes.values().to_vec())
}
