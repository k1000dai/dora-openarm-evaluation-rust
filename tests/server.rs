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

//! Integration test for `serve_connection`: a full request/response
//! round trip over in-memory buffers, reading a real Arrow IPC FILE from
//! disk -- everything upstream's request loop does except for the
//! `AF_UNIX` socket itself, which `socket_role` covers separately.

use std::cell::Cell;
use std::collections::HashMap;
use std::io::{BufReader, Cursor};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, ListArray, StructArray};
use arrow::datatypes::Field;
use arrow::datatypes::{Float32Type, UInt8Type};
use arrow::record_batch::RecordBatch;
use arrow_ipc::writer::FileWriter;
use dora_openarm_evaluation_rust::batch::ModelBatch;
use dora_openarm_evaluation_rust::camera_map::CAMERA_KEY_MAP;
use dora_openarm_evaluation_rust::policy::{MockPolicy, PolicyModel};
use dora_openarm_evaluation_rust::server::serve_connection;

/// Writes a one-row observation Arrow IPC FILE at `path`: `position` set
/// to `position`, and all five mapped camera fields set to the same
/// 3x4 (36-byte) fixture `image_prep`'s and `batch`'s tests use.
fn write_observation_file(path: &std::path::Path, position: Vec<f32>) {
    let camera_bytes: Vec<u8> = (0_u8..36).collect();

    let position_array = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(
        position.into_iter().map(Some),
    )]);
    let mut fields: Vec<(Arc<Field>, ArrayRef)> = vec![(
        Arc::new(Field::new(
            "position",
            position_array.data_type().clone(),
            false,
        )),
        Arc::new(position_array),
    )];
    for &(arrow_key, _) in CAMERA_KEY_MAP {
        let array = ListArray::from_iter_primitive::<UInt8Type, _, _>(vec![Some(
            camera_bytes.iter().copied().map(Some).collect::<Vec<_>>(),
        )]);
        fields.push((
            Arc::new(Field::new(arrow_key, array.data_type().clone(), false)),
            Arc::new(array),
        ));
    }
    let struct_array = StructArray::from(fields);
    let batch = RecordBatch::from(&struct_array);

    let file = std::fs::File::create(path).unwrap();
    let mut writer = FileWriter::try_new(file, batch.schema().as_ref()).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
}

struct ResetCountingPolicy {
    resets: Cell<usize>,
}

impl PolicyModel for ResetCountingPolicy {
    fn image_sizes(&self) -> HashMap<String, (u32, u32)> {
        CAMERA_KEY_MAP
            .iter()
            .map(|&(_, model_key)| (model_key.to_owned(), (3, 4)))
            .collect()
    }

    fn reset(&self) {
        self.resets.set(self.resets.get() + 1);
    }

    fn infer(&self, _batch: &ModelBatch) -> Vec<Vec<f32>> {
        vec![vec![0.0]]
    }
}

#[test]
fn serve_connection_round_trips_one_request() {
    let dir = tempfile::tempdir().unwrap();
    let obs_path = dir.path().join("obs.arrow");
    write_observation_file(&obs_path, vec![1.0, 2.0, 3.0]);

    let request_line = format!(
        "{{\"name\":\"inference\",\"data_path\":{:?},\"reset\":true,\"metadata\":{{}}}}\n",
        obs_path.to_string_lossy()
    );
    let mut reader = BufReader::new(Cursor::new(request_line.into_bytes()));
    let mut writer = Vec::new();

    // Sized to the fixture's camera bytes, so no resize occurs.
    let policy = MockPolicy::with_known_resolution(3, 4);
    serve_connection(&mut reader, &mut writer, &policy).unwrap();

    let response_line = String::from_utf8(writer).unwrap();
    assert!(response_line.ends_with('\n'));
    let response: serde_json::Value = serde_json::from_str(response_line.trim_end()).unwrap();

    assert_eq!(response["interval"], 33_333_333);
    assert_eq!(response["cutoff_hz"], 15);

    let positions = response["positions"].as_array().unwrap();
    assert_eq!(positions.len(), 10);
    let first_row: Vec<f64> = positions[0]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    assert_eq!(&first_row[0..3], &[1.0, 2.0, 3.0]);
    assert!(first_row[3..].iter().all(|&value| value == 0.0));
}

#[test]
fn serve_connection_stops_cleanly_at_end_of_stream() {
    let mut reader = BufReader::new(Cursor::new(Vec::new()));
    let mut writer = Vec::new();
    let policy = MockPolicy::with_known_resolution(480, 640);

    serve_connection(&mut reader, &mut writer, &policy).unwrap();
    assert!(writer.is_empty());
}

#[test]
fn serve_connection_serves_multiple_requests_on_one_connection() {
    let dir = tempfile::tempdir().unwrap();
    let obs_path = dir.path().join("obs.arrow");
    write_observation_file(&obs_path, vec![9.0]);

    let single_request = format!("{{\"data_path\":{:?}}}\n", obs_path.to_string_lossy());
    let mut reader = BufReader::new(Cursor::new(single_request.repeat(3).into_bytes()));
    let mut writer = Vec::new();
    let policy = MockPolicy::with_known_resolution(3, 4);

    serve_connection(&mut reader, &mut writer, &policy).unwrap();

    let response_count = String::from_utf8(writer).unwrap().lines().count();
    assert_eq!(response_count, 3);
}

#[test]
fn serve_connection_resets_on_start_and_at_reset_requests() {
    let dir = tempfile::tempdir().unwrap();
    let obs_path = dir.path().join("obs.arrow");
    write_observation_file(&obs_path, vec![9.0]);

    let request = format!("{{\"data_path\":{:?}}}\n", obs_path.to_string_lossy());
    let reset_request = format!(
        "{{\"data_path\":{:?},\"reset\":true}}\n",
        obs_path.to_string_lossy()
    );
    let mut reader = BufReader::new(Cursor::new(
        format!("{request}{reset_request}").into_bytes(),
    ));
    let mut writer = Vec::new();
    let policy = ResetCountingPolicy {
        resets: Cell::new(0),
    };

    serve_connection(&mut reader, &mut writer, &policy).unwrap();

    assert_eq!(policy.resets.get(), 2);
    assert_eq!(String::from_utf8(writer).unwrap().lines().count(), 2);
}
