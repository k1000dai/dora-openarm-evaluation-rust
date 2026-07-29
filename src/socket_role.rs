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

//! Unix-domain-socket establishment, one function per server role.
//!
//! The two upstream scripts have **inverted** socket roles from each
//! other, and each is the inverse of its counterpart dora node:
//!
//! | Process | Role | Counterpart |
//! |---|---|---|
//! | `local_policy_server.py` / [`bind_and_accept`] | binds, listens, accepts | `dora-openarm-local-policy-server` node **connects** |
//! | `docker_policy_server.py` / [`connect`] | connects | `dora-openarm-docker-policy-server` node **binds, listens, accepts** |
//!
//! Getting either role backwards means the two processes deadlock
//! waiting for each other instead of talking.

use std::io;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

/// Removes a stale socket file at `path`, if any, and binds a listener
/// on it -- the first half of the **local** policy server's role.
///
/// Split out from [`bind_and_accept`] so a caller can install a
/// [`SocketCleanup`] guard between binding and accepting, matching
/// upstream's `sock.bind(...)` (unguarded) followed by a `try`/`finally`
/// around `sock.accept()`.
///
/// # Errors
///
/// Returns an error if removing a stale socket file or binding fails.
pub fn bind(path: &Path) -> io::Result<UnixListener> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    UnixListener::bind(path)
}

/// Binds and accepts exactly one connection on `path` -- the **local**
/// policy server's role.
///
/// Mirrors upstream's `local_policy_server.py`: removes a stale socket
/// file left over from a previous run before binding, then blocks on a
/// single `sock.accept()`. Only the first connection is served; this
/// function does not loop. A convenience composition of [`bind`] +
/// `accept` for callers that do not need a [`SocketCleanup`] guard
/// between the two steps (for example, tests).
///
/// # Errors
///
/// Returns an error if removing a stale socket file, binding, or
/// accepting a connection fails.
pub fn bind_and_accept(path: &Path) -> io::Result<UnixStream> {
    let listener = bind(path)?;
    let (stream, _peer_addr) = listener.accept()?;
    Ok(stream)
}

/// Connects to `path` -- the **docker** policy server's role.
///
/// Mirrors upstream's `docker_policy_server.py`: a single `sock.connect`
/// attempt with no retry. If nothing is listening on `path` yet, this
/// fails immediately rather than waiting.
///
/// # Errors
///
/// Returns an error if the connection attempt fails.
pub fn connect(path: &Path) -> io::Result<UnixStream> {
    UnixStream::connect(path)
}

/// Removes the socket file at `path` when dropped, if it still exists.
///
/// Mirrors upstream's local server's `try`/`finally`, which always
/// removes the socket file on the way out, whether the connection ended
/// cleanly or an exception propagated. Only [`bind_and_accept`]'s local
/// server role owns the socket file; [`connect`]'s docker role never
/// removes it, matching upstream (the *node* on that side owns the
/// file).
#[must_use = "the socket file is removed when this guard is dropped"]
pub struct SocketCleanup {
    path: PathBuf,
}

impl SocketCleanup {
    /// Creates a guard that removes `path` on drop.
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for SocketCleanup {
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
