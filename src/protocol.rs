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

//! The NDJSON request/response protocol served on the policy server socket.
//!
//! This is the server-side mirror of the protocol already implemented from
//! the client (dora node) side in `dora-openarm-local-policy-server-rust`
//! and `dora-openarm-docker-policy-server-rust`. One JSON object per line,
//! in both directions, on a single long-lived connection.

use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The interval, in nanoseconds, between successive rows of a response's
/// `positions`.
///
/// Matches upstream's `INTERVAL_NS = 33_333_333` (30 Hz).
pub const INTERVAL_NS: i64 = 33_333_333;

/// The low-pass filter cutoff frequency, in Hz, that this server reports
/// alongside every response.
///
/// Matches upstream's `CUTOFF_HZ = 15`.
pub const CUTOFF_HZ: u32 = 15;

/// One inference request read from the socket, one JSON object per line.
///
/// Field names match upstream's request dict exactly: `name`, `data_path`,
/// `reset`, `metadata`. `name` and `metadata` are accepted but not
/// interpreted by this server, matching upstream, which never reads
/// either. `reset` marks an episode boundary and causes the policy
/// adapter to reset before that observation is processed.
#[derive(Debug, Clone, Deserialize)]
pub struct InferenceRequest {
    /// The request kind. Upstream always sends `"inference"`, but this
    /// server does not validate it, matching upstream.
    #[serde(default)]
    pub name: Option<String>,
    /// Path to the Arrow IPC FILE holding the observation's record batch.
    pub data_path: String,
    /// Whether this observation starts a new episode. When true, the
    /// policy adapter is reset before inference.
    #[serde(default)]
    pub reset: bool,
    /// The forwarded dora input metadata, opaque to this protocol.
    #[serde(default)]
    pub metadata: Value,
}

/// One inference response written to the socket, one JSON object per line.
///
/// Field order matches upstream's response dict exactly: `interval`,
/// `cutoff_hz`, `positions`.
#[derive(Debug, Clone, Serialize)]
pub struct ActionsResponse {
    /// The interval, in nanoseconds, between successive rows of
    /// `positions`. Always [`INTERVAL_NS`].
    pub interval: i64,
    /// The low-pass filter cutoff frequency, in Hz. Always
    /// `Some(`[`CUTOFF_HZ`]`)`.
    pub cutoff_hz: Option<u32>,
    /// The inferred motor positions, one row per action step.
    pub positions: Vec<Vec<f32>>,
}

impl ActionsResponse {
    /// Builds a response carrying `positions`, with `interval` and
    /// `cutoff_hz` set to the server's fixed contract values.
    #[must_use]
    pub fn new(positions: Vec<Vec<f32>>) -> Self {
        Self {
            interval: INTERVAL_NS,
            cutoff_hz: Some(CUTOFF_HZ),
            positions,
        }
    }
}

/// Reads and parses one NDJSON request line.
///
/// Returns `Ok(None)` on end of stream, matching upstream's `for line in
/// io:` loop ending when the client closes the connection.
///
/// # Errors
///
/// Returns an error if reading fails or the line is not a valid
/// [`InferenceRequest`].
pub fn read_request(reader: &mut impl BufRead) -> io::Result<Option<InferenceRequest>> {
    let mut line = String::new();
    let bytes_read = reader.read_line(&mut line)?;
    if bytes_read == 0 {
        return Ok(None);
    }
    let request = serde_json::from_str(line.trim_end())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok(Some(request))
}

/// Writes `response` as a single NDJSON line and flushes the writer.
///
/// # Errors
///
/// Returns an error if serialization or the underlying write fails.
pub fn write_response(writer: &mut impl Write, response: &ActionsResponse) -> io::Result<()> {
    let line = serde_json::to_string(response)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writeln!(writer, "{line}")?;
    writer.flush()
}
