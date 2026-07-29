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

//! Tests for socket path argument resolution.

use dora_openarm_evaluation_rust::cli::{
    DEFAULT_SOCKET, require_mock_flag, resolve_docker_socket_path, resolve_local_socket_path,
};

#[test]
fn mock_policy_requires_explicit_opt_in() {
    assert!(require_mock_flag(&[]).is_err());
    assert!(require_mock_flag(&["/tmp/custom.socket".to_owned()]).is_err());
    assert_eq!(
        require_mock_flag(&["--mock".to_owned(), "/tmp/custom.socket".to_owned()]).unwrap(),
        &["/tmp/custom.socket".to_owned()]
    );
}

#[test]
fn local_uses_the_given_argument() {
    let args = vec!["/tmp/custom.socket".to_owned()];
    assert_eq!(
        resolve_local_socket_path(&args),
        std::path::PathBuf::from("/tmp/custom.socket")
    );
}

#[test]
fn local_falls_back_to_the_default_when_no_argument_is_given() {
    assert_eq!(
        resolve_local_socket_path(&[]),
        std::path::PathBuf::from(DEFAULT_SOCKET)
    );
}

#[test]
fn docker_uses_the_given_argument() {
    let args = vec!["/tmp/custom.socket".to_owned()];
    assert_eq!(
        resolve_docker_socket_path(&args).unwrap(),
        std::path::PathBuf::from("/tmp/custom.socket")
    );
}

#[test]
fn docker_errors_when_no_argument_is_given() {
    assert!(resolve_docker_socket_path(&[]).is_err());
}
