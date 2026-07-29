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

//! Golden tests for `detect_resolution`, cross-checked against a
//! faithful Python replica of upstream's `detect_resolution` (see the
//! commit description / README for how these were derived).

use dora_openarm_evaluation_rust::detect_resolution;

#[test]
fn known_resolution_600x960() {
    assert_eq!(detect_resolution(600 * 960 * 3), Ok((600, 960)));
}

#[test]
fn known_resolution_720x1280() {
    assert_eq!(detect_resolution(720 * 1280 * 3), Ok((720, 1280)));
}

#[test]
fn known_resolution_480x640() {
    assert_eq!(detect_resolution(480 * 640 * 3), Ok((480, 640)));
}

#[test]
fn known_resolution_1080x1920() {
    assert_eq!(detect_resolution(1080 * 1920 * 3), Ok((1080, 1920)));
}

#[test]
fn fallback_ratio_3_4_exact_match() {
    // 300x400 has an exact 3:4 aspect ratio and is not a known
    // resolution, so it is resolved by the first fallback ratio.
    assert_eq!(detect_resolution(300 * 400 * 3), Ok((300, 400)));
}

#[test]
fn fallback_ratio_9_16_exact_match() {
    // 225x400 fails the 3:4 ratio's floor-sqrt factorization but
    // succeeds on the second ratio, 9:16.
    assert_eq!(detect_resolution(225 * 400 * 3), Ok((225, 400)));
}

#[test]
fn unresolvable_byte_count_is_an_error() {
    // 97 pixels (a prime) does not factor exactly under any of the four
    // fallback ratios' floor-sqrt heuristic.
    let n_bytes = 97 * 3;
    assert_eq!(
        detect_resolution(n_bytes),
        Err(dora_openarm_evaluation_rust::UnresolvableResolution { n_bytes })
    );
}

#[test]
fn error_message_reports_the_byte_count() {
    let error = detect_resolution(291).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Cannot determine resolution for 291 bytes"
    );
}
