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

//! Command-line socket path resolution for each server binary.
//!
//! Neither upstream script uses `argparse` or an environment variable --
//! both take the socket path as a single positional argument, resolved
//! directly from `sys.argv`. This is deliberately different from the
//! dora node side of this protocol (`dora-openarm-local-policy-server-rust`
//! / `dora-openarm-docker-policy-server-rust`), which does accept a
//! `--socket` flag and a `SOCKET` environment variable.

use std::fmt;
use std::path::PathBuf;

/// The local server's default socket path when no argument is given.
///
/// Matches upstream's `DEFAULT_SOCKET = "/dev/shm/policy-server.socket"`
/// in `local_policy_server.py`.
pub const DEFAULT_SOCKET: &str = "/dev/shm/policy-server.socket";

/// The docker server requires its socket path argument, but none was
/// given.
///
/// Mirrors upstream's `docker_policy_server.py`, which raises an
/// uncaught `IndexError` on `sys.argv[1]` when invoked with no
/// arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingSocketPath;

impl fmt::Display for MissingSocketPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "missing required socket path argument")
    }
}

impl std::error::Error for MissingSocketPath {}

/// The executable was started without explicitly opting into the mock policy.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MissingMockFlag;

impl fmt::Debug for MissingMockFlag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for MissingMockFlag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "real LeRobot inference is not implemented in this Rust binary; pass --mock only for protocol testing, or run docker/Dockerfile.lerobot"
        )
    }
}

impl std::error::Error for MissingMockFlag {}

/// Requires an explicit leading `--mock` and returns the remaining arguments.
///
/// The Rust executables contain only [`crate::policy::MockPolicy`]. Requiring
/// an opt-in prevents a test policy from silently producing actions in a real
/// evaluation dataflow.
///
/// # Errors
///
/// Returns [`MissingMockFlag`] unless `args[0]` is exactly `--mock`.
pub fn require_mock_flag(args: &[String]) -> Result<&[String], MissingMockFlag> {
    args.strip_prefix(&["--mock".to_owned()])
        .ok_or(MissingMockFlag)
}

/// Resolves the local server's socket path: `args[0]` if given, else
/// [`DEFAULT_SOCKET`].
///
/// Mirrors upstream's `sys.argv[1] if len(sys.argv) > 1 else
/// DEFAULT_SOCKET` (`args` here excludes the program name, so index 0
/// corresponds to upstream's `sys.argv[1]`).
pub fn resolve_local_socket_path(args: &[String]) -> PathBuf {
    args.first()
        .map_or_else(|| PathBuf::from(DEFAULT_SOCKET), PathBuf::from)
}

/// Resolves the docker server's socket path: `args[0]`, required.
///
/// Mirrors upstream's `socket_path = sys.argv[1]`.
///
/// # Errors
///
/// Returns [`MissingSocketPath`] if `args` is empty.
pub fn resolve_docker_socket_path(args: &[String]) -> Result<PathBuf, MissingSocketPath> {
    args.first().map(PathBuf::from).ok_or(MissingSocketPath)
}
