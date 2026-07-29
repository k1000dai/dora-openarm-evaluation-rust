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

//! Image resize and channel normalization.
//!
//! Ports upstream's `prepare_image`: detect the source resolution from the
//! buffer length, resize to the policy's expected input size if it
//! differs, then convert channel-last `uint8` `[0, 255]` to channel-first
//! (`CHW`) `f32` `[0.0, 1.0]` -- upstream's
//! `torch.from_numpy(img).permute(2, 0, 1).float() / 255.0`.
//!
//! **Resize fidelity boundary.** Upstream resizes with
//! `PIL.Image.fromarray(img).resize((target_w, target_h))`, whose default
//! resampling filter is Pillow's bicubic. This port resizes with the
//! `image` crate's [`image::imageops::FilterType::CatmullRom`], a
//! Catmull-Rom bicubic variant. The two are **not** guaranteed to produce
//! byte-identical pixels: exact parity would require re-implementing
//! Pillow's specific resampling kernel and rounding, and this crate does
//! not claim to do so. Only the identity case (source resolution already
//! matches the target, so no resize happens) is guaranteed byte-exact,
//! and is the case this crate's golden tests assert numerically.

use std::fmt;

use image::{ImageBuffer, Rgb, imageops::FilterType};

use crate::resolution::{UnresolvableResolution, detect_resolution};

/// The number of channels in every image this crate handles.
const CHANNELS: usize = 3;

/// A resized, `CHW`-layout, `[0.0, 1.0]`-normalized image, ready to hand
/// to a [`crate::policy::PolicyModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedImage {
    /// Always [`CHANNELS`].
    pub channels: usize,
    /// The image height in pixels, after resizing.
    pub height: u32,
    /// The image width in pixels, after resizing.
    pub width: u32,
    /// Pixel data in `CHW` order: channel-major, then row, then column.
    /// Length is `channels * height * width`.
    pub data: Vec<f32>,
}

/// An error preparing an image.
#[derive(Debug)]
pub enum ImagePrepError {
    /// The buffer length could not be resolved to a `(height, width)`.
    UnresolvableResolution(UnresolvableResolution),
    /// The buffer length did not match `height * width * 3` for the
    /// resolution [`detect_resolution`] reported (should not happen; kept
    /// as a defensive check rather than a panic).
    BufferSizeMismatch,
}

impl fmt::Display for ImagePrepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnresolvableResolution(error) => write!(f, "{error}"),
            Self::BufferSizeMismatch => {
                write!(
                    f,
                    "image buffer length does not match its detected resolution"
                )
            }
        }
    }
}

impl std::error::Error for ImagePrepError {}

impl From<UnresolvableResolution> for ImagePrepError {
    fn from(error: UnresolvableResolution) -> Self {
        Self::UnresolvableResolution(error)
    }
}

/// Detects `raw`'s resolution, resizes it to `(target_h, target_w)` if
/// needed, and converts it to a normalized `CHW` `f32` image.
///
/// # Errors
///
/// Returns an error if `raw`'s length cannot be resolved to a
/// `(height, width)`, or (defensively) if it does not match that
/// resolution's expected byte count.
///
/// # Panics
///
/// Never panics: the length check above guarantees the internal
/// `ImageBuffer::from_raw` call always succeeds.
pub fn prepare_image(
    raw: &[u8],
    target_h: u32,
    target_w: u32,
) -> Result<PreparedImage, ImagePrepError> {
    let (src_h, src_w) = detect_resolution(raw.len())?;
    let expected_len = src_h as usize * src_w as usize * CHANNELS;
    if raw.len() != expected_len {
        return Err(ImagePrepError::BufferSizeMismatch);
    }

    let buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(src_w, src_h, raw.to_vec())
        .expect("length checked above matches src_w * src_h * 3");

    let resized = if (src_h, src_w) == (target_h, target_w) {
        buffer
    } else {
        image::imageops::resize(&buffer, target_w, target_h, FilterType::CatmullRom)
    };

    Ok(to_chw_f32(&resized))
}

/// Converts an `HWC` `u8` image buffer to `CHW` `f32`, dividing every
/// sample by 255.
fn to_chw_f32(image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> PreparedImage {
    let (width, height) = (image.width(), image.height());
    let (w, h) = (width as usize, height as usize);
    let mut data = vec![0.0_f32; CHANNELS * h * w];
    for (x, y, pixel) in image.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        for channel in 0..CHANNELS {
            data[channel * h * w + y * w + x] = f32::from(pixel.0[channel]) / 255.0;
        }
    }
    PreparedImage {
        channels: CHANNELS,
        height,
        width,
        data,
    }
}
