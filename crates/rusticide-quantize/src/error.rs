// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error and format tag shared by every quantization format.

/// A supported quantization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    Q4_0,
    Q4K,
    Q6K,
}

/// Failure while quantizing or dequantizing.
#[derive(Debug, thiserror::Error)]
pub enum QuantizeError {
    #[error("{format:?} data too short: {got} bytes, need {needed}")]
    DataTooShort {
        format: QuantizationType,
        got: usize,
        needed: usize,
    },
    #[error("{format:?} needs a multiple of {block} elements, got {numel}")]
    UnalignedShape {
        format: QuantizationType,
        numel: usize,
        block: usize,
    },
    #[error(transparent)]
    Tensor(#[from] rusticide_core::Error),
}

pub(crate) const fn ensure_len(
    format: QuantizationType,
    data: &[u8],
    needed: usize,
) -> Result<(), QuantizeError> {
    if data.len() < needed {
        return Err(QuantizeError::DataTooShort {
            format,
            got: data.len(),
            needed,
        });
    }
    Ok(())
}

/// Rejects element counts that are not whole blocks, as ggml requires for
/// K-quants. A saturated (overflowing) element count is `usize::MAX`, which is
/// never a multiple of a power-of-two block, so it fails here first.
pub(crate) const fn ensure_aligned(
    format: QuantizationType,
    numel: usize,
    block: usize,
) -> Result<(), QuantizeError> {
    if !matches!(numel.checked_rem(block), Some(0)) {
        return Err(QuantizeError::UnalignedShape {
            format,
            numel,
            block,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_len_accepts_exact_and_longer_data() {
        assert!(ensure_len(QuantizationType::Q6K, &[0; 4], 4).is_ok());
        assert!(ensure_len(QuantizationType::Q6K, &[0; 5], 4).is_ok());
    }

    #[test]
    fn ensure_len_rejects_short_data() {
        let err = ensure_len(QuantizationType::Q4K, &[0; 3], 4).unwrap_err();
        assert!(matches!(
            err,
            QuantizeError::DataTooShort {
                format: QuantizationType::Q4K,
                got: 3,
                needed: 4
            }
        ));
    }

    #[test]
    fn ensure_aligned_checks_the_remainder() {
        assert!(ensure_aligned(QuantizationType::Q4K, 512, 256).is_ok());
        assert!(matches!(
            ensure_aligned(QuantizationType::Q4K, 1, 256).unwrap_err(),
            QuantizeError::UnalignedShape {
                format: QuantizationType::Q4K,
                numel: 1,
                block: 256
            }
        ));
    }

    #[test]
    fn display_names_the_format_and_sizes() {
        let err = QuantizeError::DataTooShort {
            format: QuantizationType::Q4_0,
            got: 5,
            needed: 36,
        };
        assert_eq!(err.to_string(), "Q4_0 data too short: 5 bytes, need 36");
    }
}
