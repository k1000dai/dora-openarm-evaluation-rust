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

//! The request/response loop shared by both policy server binaries.
//!
//! Upstream's `local_policy_server.py` and `docker_policy_server.py` are
//! ~95% identical: only how the socket is established differs (see
//! [`crate::socket_role`]). This module is their shared `for line in io:`
//! body -- one request in, one response out, on a single connection,
//! blocking, synchronous. It is deliberately *not* async: the actions
//! executor on the other end of the dataflow paces its output on the
//! arrival cadence of these responses (see the Rust port of
//! `dora-openarm-actions-executor`), so batching or reordering requests
//! here would change downstream timing.

use std::fmt;
use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::batch::{BatchError, build_batch};
use crate::observation::{ObservationError, load_observation};
use crate::policy::PolicyModel;
use crate::protocol::{ActionsResponse, read_request, write_response};

/// An error serving one connection.
#[derive(Debug)]
pub enum ServerError {
    /// Reading a request or writing a response failed.
    Io(io::Error),
    /// Loading the observation named by a request's `data_path` failed.
    Observation(ObservationError),
    /// Building the model batch from the observation failed.
    Batch(BatchError),
}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Observation(error) => write!(f, "{error}"),
            Self::Batch(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ServerError {}

impl From<io::Error> for ServerError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ObservationError> for ServerError {
    fn from(error: ObservationError) -> Self {
        Self::Observation(error)
    }
}

impl From<BatchError> for ServerError {
    fn from(error: BatchError) -> Self {
        Self::Batch(error)
    }
}

/// Serves requests from `reader`, writing responses to `writer`, until
/// the client closes the connection.
///
/// Resets the policy once for the connection, then for each request loads
/// the observation Arrow IPC file named by `data_path`, builds a
/// [`crate::batch::ModelBatch`] using `policy`'s declared image sizes,
/// infers an action chunk, and writes back an [`ActionsResponse`] carrying
/// it plus this server's fixed `interval` and `cutoff_hz`. A request with
/// `reset: true` resets the policy before its observation is processed,
/// matching upstream's episode-boundary behavior. Requests are run to
/// completion in arrival order.
///
/// # Errors
///
/// Returns an error and stops serving if reading, parsing, loading,
/// batching, or writing fails for any single request. Upstream has the
/// same behavior: an unhandled exception in the loop body propagates out
/// of `main`, ending the process.
pub fn serve_connection<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    policy: &dyn PolicyModel,
) -> Result<(), ServerError> {
    policy.reset();
    let image_sizes = policy.image_sizes();
    while let Some(request) = read_request(reader)? {
        if request.reset {
            policy.reset();
        }
        let observation = load_observation(Path::new(&request.data_path))?;
        let batch = build_batch(&observation, &image_sizes)?;
        let positions = policy.infer(&batch);
        write_response(writer, &ActionsResponse::new(positions))?;
    }
    Ok(())
}
