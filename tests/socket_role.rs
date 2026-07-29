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

#![cfg(unix)]

//! Tests for the local (bind/listen/accept) and docker (connect) socket
//! roles, and for the local role's socket-file cleanup guard.

use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use dora_openarm_evaluation_rust::socket_role::{SocketCleanup, bind, bind_and_accept, connect};

fn temp_socket_path(name: &str) -> PathBuf {
    PathBuf::from(format!("/tmp/doe-{}-{name}.sock", std::process::id()))
}

#[test]
fn bind_and_accept_completes_a_connection() {
    let path = temp_socket_path("bind-accept");
    let _ = std::fs::remove_file(&path);

    let accept_path = path.clone();
    let handle = thread::spawn(move || bind_and_accept(&accept_path));

    let stream = loop {
        match UnixStream::connect(&path) {
            Ok(stream) => break stream,
            Err(_) => thread::sleep(Duration::from_millis(5)),
        }
    };
    drop(stream);

    let accepted = handle.join().unwrap();
    assert!(accepted.is_ok());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn bind_removes_a_stale_socket_file_first() {
    let path = temp_socket_path("stale");
    std::fs::write(&path, b"not a socket").unwrap();

    let listener = bind(&path).expect("bind should remove the stale file and succeed");
    drop(listener);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn connect_to_a_bound_socket_succeeds() {
    let path = temp_socket_path("connect-ok");
    let _ = std::fs::remove_file(&path);
    let listener = bind(&path).unwrap();

    let accept_path = path.clone();
    let handle = thread::spawn(move || connect(&accept_path));

    let (_stream, _peer_addr) = listener.accept().unwrap();
    let connected = handle.join().unwrap();
    assert!(connected.is_ok());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn connect_to_a_nonexistent_socket_fails_immediately() {
    let path = temp_socket_path("missing");
    let _ = std::fs::remove_file(&path);
    assert!(connect(&path).is_err());
}

#[test]
fn socket_cleanup_removes_the_file_on_drop() {
    let path = temp_socket_path("cleanup");
    std::fs::write(&path, b"placeholder").unwrap();
    assert!(path.exists());

    {
        let _guard = SocketCleanup::new(path.clone());
    }
    assert!(!path.exists());
}

#[test]
fn socket_cleanup_is_a_no_op_if_the_file_is_already_gone() {
    let path = temp_socket_path("already-gone");
    let _ = std::fs::remove_file(&path);
    assert!(!path.exists());

    let guard = SocketCleanup::new(path.clone());
    drop(guard);
    assert!(!path.exists());
}
