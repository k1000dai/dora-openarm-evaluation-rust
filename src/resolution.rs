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

//! Resolution detection for flat, channel-last `uint8` camera buffers.
//!
//! Ports upstream's `detect_resolution` exactly, including its search
//! order and its truncating (not rounding) integer square root.

use std::fmt;

/// Known `(height, width)` resolutions, keyed by their 3-channel byte
/// count (`height * width * 3`).
///
/// Order and values match upstream's `KNOWN_RESOLUTIONS` dict exactly.
const KNOWN_RESOLUTIONS: &[(usize, (u32, u32))] = &[
    (600 * 960 * 3, (600, 960)),
    (720 * 1280 * 3, (720, 1280)),
    (480 * 640 * 3, (480, 640)),
    (1080 * 1920 * 3, (1080, 1920)),
];

/// Aspect ratios tried, in order, when `n_bytes` is not one of the known
/// resolutions.
///
/// Matches upstream's `[(3, 4), (9, 16), (3, 5), (2, 3)]` exactly -- both
/// the ratios and their order are load-bearing, since the first ratio
/// that evenly factors `n_pixels` wins.
const FALLBACK_RATIOS: &[(u32, u32)] = &[(3, 4), (9, 16), (3, 5), (2, 3)];

/// The buffer length could not be resolved to a `(height, width)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnresolvableResolution {
    /// The byte count that could not be resolved.
    pub n_bytes: usize,
}

impl fmt::Display for UnresolvableResolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cannot determine resolution for {} bytes", self.n_bytes)
    }
}

impl std::error::Error for UnresolvableResolution {}

/// Detects the `(height, width)` of a flat 3-channel `uint8` buffer of
/// `n_bytes` bytes.
///
/// Checks [`KNOWN_RESOLUTIONS`] first, then tries each of
/// [`FALLBACK_RATIOS`] in order: `h = floor(sqrt(n_pixels * ratio_h /
/// ratio_w))`, `w = n_pixels / h` (integer division), accepting the first
/// ratio where `h * w == n_pixels` exactly. Mirrors upstream's
/// `detect_resolution` line for line, including its use of `float`
/// (`f64`) square root truncated by `int()`, not rounded.
///
/// # Errors
///
/// Returns [`UnresolvableResolution`] if no known resolution and no
/// fallback ratio produces an exact factorization.
pub fn detect_resolution(n_bytes: usize) -> Result<(u32, u32), UnresolvableResolution> {
    for &(bytes, resolution) in KNOWN_RESOLUTIONS {
        if bytes == n_bytes {
            return Ok(resolution);
        }
    }

    let n_pixels = n_bytes / 3;
    for &(ratio_h, ratio_w) in FALLBACK_RATIOS {
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let h = ((n_pixels * ratio_h as usize) as f64 / f64::from(ratio_w)).sqrt() as usize;
        // Deliberate deviation from upstream: for a degenerate `n_pixels`
        // small enough that `h` truncates to 0, upstream's `n_pixels //
        // h` raises an uncaught `ZeroDivisionError`, crashing the
        // request handler. No real camera frame is this small, so we
        // instead skip to the next ratio and, if every ratio fails,
        // return `UnresolvableResolution` -- strictly more robust, never
        // less correct for any input that upstream itself handles.
        if h == 0 {
            continue;
        }
        let w = n_pixels / h;
        if h * w == n_pixels {
            #[allow(clippy::cast_possible_truncation)]
            return Ok((h as u32, w as u32));
        }
    }

    Err(UnresolvableResolution { n_bytes })
}
