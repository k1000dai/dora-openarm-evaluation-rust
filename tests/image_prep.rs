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

//! Golden tests for `prepare_image`.
//!
//! The identity case (no resize needed) is asserted byte-exact, since it
//! involves only a deterministic reshape and `/255.0` normalization. The
//! resize case only asserts shape and value range: see `image_prep`'s
//! module docs for why byte-exact resize parity with Pillow is not
//! claimed.

use dora_openarm_evaluation_rust::image_prep::{ImagePrepError, prepare_image};
use dora_openarm_evaluation_rust::resolution::UnresolvableResolution;

/// A 3x4 raw `HWC` `u8` buffer (one of upstream's fallback ratios
/// resolves this exactly, at the smallest size that does) filled with a
/// sequential, uniquely identifiable pattern.
fn tiny_image() -> Vec<u8> {
    (0_u8..36).collect()
}

#[test]
fn identity_case_is_byte_exact() {
    let raw = tiny_image();
    let prepared = prepare_image(&raw, 3, 4).expect("3x4 resolves without a resize");

    assert_eq!(prepared.channels, 3);
    assert_eq!(prepared.height, 3);
    assert_eq!(prepared.width, 4);
    assert_eq!(prepared.data.len(), 36);

    // Corner pixels, hand-derived from the HWC source layout
    // (index = (y * 4 + x) * 3 + c) and the CHW target layout
    // (index = c * 12 + y * 4 + x).
    assert_eq!(prepared.data[0].to_bits(), (0.0_f32 / 255.0).to_bits()); // y=0, x=0, c=0
    assert_eq!(prepared.data[12].to_bits(), (1.0_f32 / 255.0).to_bits()); // y=0, x=0, c=1
    assert_eq!(prepared.data[24].to_bits(), (2.0_f32 / 255.0).to_bits()); // y=0, x=0, c=2
    assert_eq!(prepared.data[11].to_bits(), (33.0_f32 / 255.0).to_bits()); // y=2, x=3, c=0
    assert_eq!(prepared.data[23].to_bits(), (34.0_f32 / 255.0).to_bits()); // y=2, x=3, c=1
    assert_eq!(prepared.data[35].to_bits(), (35.0_f32 / 255.0).to_bits()); // y=2, x=3, c=2

    // Every element follows the same rule.
    for y in 0..3_usize {
        for x in 0..4_usize {
            for c in 0..3_usize {
                let hwc_index = (y * 4 + x) * 3 + c;
                let chw_index = c * 3 * 4 + y * 4 + x;
                let expected = f32::from(raw[hwc_index]) / 255.0;
                assert_eq!(
                    prepared.data[chw_index].to_bits(),
                    expected.to_bits(),
                    "y={y} x={x} c={c}"
                );
            }
        }
    }
}

#[test]
fn resize_case_produces_the_target_shape_and_normalized_range() {
    let raw = tiny_image();
    let prepared = prepare_image(&raw, 6, 8).expect("resize to double the resolution");

    assert_eq!(prepared.channels, 3);
    assert_eq!(prepared.height, 6);
    assert_eq!(prepared.width, 8);
    assert_eq!(prepared.data.len(), 3 * 6 * 8);
    for &value in &prepared.data {
        assert!((0.0..=1.0).contains(&value), "value {value} out of [0, 1]");
    }
}

#[test]
fn unresolvable_length_is_an_error() {
    let raw = vec![0_u8; 97 * 3];
    let error = prepare_image(&raw, 10, 10).unwrap_err();
    assert!(matches!(
        error,
        ImagePrepError::UnresolvableResolution(UnresolvableResolution { n_bytes: 291 })
    ));
}

#[test]
fn length_not_matching_its_own_detected_resolution_is_a_defensive_error() {
    // 37 bytes: `n_pixels = 37 / 3 = 12` (integer division), which
    // resolves to `(3, 4)` under the same ratio as `tiny_image`'s 36
    // bytes -- but 37 != 3 * 4 * 3. This is the defensive check in
    // `prepare_image`, not a case upstream's own arithmetic reaches
    // cleanly either.
    let raw = vec![0_u8; 37];
    let error = prepare_image(&raw, 3, 4).unwrap_err();
    assert!(matches!(error, ImagePrepError::BufferSizeMismatch));
}
