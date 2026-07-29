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

//! `dora-openarm-evaluation-local-policy-server` -- behavior-compatible
//! port of upstream's `src/local_policy_server.py`.
//!
//! Binds, listens, and accepts exactly one connection on a Unix domain
//! socket (the *local* role -- the counterpart
//! `dora-openarm-local-policy-server` dora node connects to it), then
//! serves inference requests until the connection closes.
//!
//! **This binary does not run `LeRobot` ACT inference.** It plugs in
//! [`dora_openarm_evaluation_rust::MockPolicy`], a deterministic stand-in.
//! See the crate's `policy` module docs and `README.md`, "The `LeRobot`
//! boundary", for what that means and why.

use std::io::BufReader;

use dora_openarm_evaluation_rust::cli::{require_mock_flag, resolve_local_socket_path};
use dora_openarm_evaluation_rust::policy::MockPolicy;
use dora_openarm_evaluation_rust::server::serve_connection;
use dora_openarm_evaluation_rust::socket_role::{SocketCleanup, bind};

/// The `(height, width)` this binary's [`MockPolicy`] declares for every
/// mapped camera input.
///
/// `480x640` is one of upstream's [`dora_openarm_evaluation_rust::resolution`]
/// known resolutions, so an observation camera frame already at this
/// size is prepared without a resize.
const MOCK_IMAGE_SIZE: (u32, u32) = (480, 640);

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let socket_path = resolve_local_socket_path(require_mock_flag(&args)?);

    println!("Listening on {}", socket_path.display());
    let listener = bind(&socket_path)?;
    // Installed only after a successful bind, mirroring upstream's
    // unguarded `sock.bind(...)` followed by a `try`/`finally` around
    // `sock.accept()` and the connection body.
    let _cleanup = SocketCleanup::new(socket_path.clone());

    let (stream, _peer_addr) = listener.accept()?;
    println!("Connected");
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;

    let policy = MockPolicy::with_known_resolution(MOCK_IMAGE_SIZE.0, MOCK_IMAGE_SIZE.1);
    serve_connection(&mut reader, &mut writer, &policy)?;

    Ok(())
}

#[cfg(not(unix))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "dora-openarm-evaluation-local-policy-server requires a Unix-like OS (AF_UNIX sockets)"
            .into(),
    )
}
