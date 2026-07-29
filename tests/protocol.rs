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

//! Golden tests for the NDJSON protocol: byte-exact fixtures matching
//! upstream's request/response dict shapes and field order.

use std::io::{BufReader, Cursor};

use dora_openarm_evaluation_rust::protocol::{ActionsResponse, read_request, write_response};
use serde_json::json;

#[test]
fn request_with_full_fields_decodes() {
    let line = r#"{"name":"inference","data_path":"/dev/shm/x.arrow","reset":true,"metadata":{"watermark":7}}
"#;
    let mut reader = BufReader::new(Cursor::new(line.as_bytes().to_vec()));
    let request = read_request(&mut reader)
        .unwrap()
        .expect("one request line");

    assert_eq!(request.name.as_deref(), Some("inference"));
    assert_eq!(request.data_path, "/dev/shm/x.arrow");
    assert!(request.reset);
    assert_eq!(request.metadata, json!({"watermark": 7}));
}

#[test]
fn request_missing_optional_fields_uses_defaults() {
    // Upstream never omits these, but the server should not depend on
    // that: `name`, `reset`, and `metadata` must not be required.
    let line = "{\"data_path\":\"/dev/shm/x.arrow\"}\n";
    let mut reader = BufReader::new(Cursor::new(line.as_bytes().to_vec()));
    let request = read_request(&mut reader)
        .unwrap()
        .expect("one request line");

    assert_eq!(request.name, None);
    assert_eq!(request.data_path, "/dev/shm/x.arrow");
    assert!(!request.reset);
    assert_eq!(request.metadata, serde_json::Value::Null);
}

#[test]
fn end_of_stream_returns_none() {
    let mut reader = BufReader::new(Cursor::new(Vec::new()));
    assert!(read_request(&mut reader).unwrap().is_none());
}

#[test]
fn response_serializes_with_upstream_field_order_and_values() {
    let response = ActionsResponse::new(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    let mut buffer = Vec::new();
    write_response(&mut buffer, &response).unwrap();

    let line = String::from_utf8(buffer).unwrap();
    assert_eq!(
        line,
        "{\"interval\":33333333,\"cutoff_hz\":15,\"positions\":[[0.1,0.2],[0.3,0.4]]}\n"
    );
}

#[test]
fn response_carries_the_fixed_contract_constants() {
    let response = ActionsResponse::new(vec![]);
    assert_eq!(
        response.interval,
        dora_openarm_evaluation_rust::protocol::INTERVAL_NS
    );
    assert_eq!(response.interval, 33_333_333);
    assert_eq!(response.cutoff_hz, Some(15));
    assert_eq!(
        response.cutoff_hz,
        Some(dora_openarm_evaluation_rust::protocol::CUTOFF_HZ)
    );
}
