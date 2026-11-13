// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_transformer::{Activation, Family, RopeStyle};

/// A model family sharing the llama forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Variant {
    Llama,
    Qwen2,
}

impl Variant {
    /// The `general.architecture` value of this family.
    #[must_use]
    pub const fn architecture(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
        }
    }

    #[must_use]
    pub const fn rope(self) -> RopeStyle {
        match self {
            Self::Llama => RopeStyle::Adjacent,
            Self::Qwen2 => RopeStyle::HalfSplit,
        }
    }

    #[must_use]
    pub const fn requires_attention_bias(self) -> bool {
        matches!(self, Self::Qwen2)
    }

    /// The shared decoder's view of this family: a gated `SiLU`
    /// feed-forward and no sub-norms.
    #[must_use]
    pub const fn family(self) -> Family {
        Family {
            architecture: self.architecture(),
            rope: self.rope(),
            activation: Activation::Silu,
            requires_attention_bias: self.requires_attention_bias(),
            sub_norms: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_name_their_architecture_and_rope() {
        assert_eq!(Variant::Llama.architecture(), "llama");
        assert_eq!(Variant::Qwen2.architecture(), "qwen2");
        assert_eq!(Variant::Llama.rope(), RopeStyle::Adjacent);
        assert_eq!(Variant::Qwen2.rope(), RopeStyle::HalfSplit);
        assert!(!Variant::Llama.requires_attention_bias());
        assert!(Variant::Qwen2.requires_attention_bias());
    }

    #[test]
    fn variants_describe_their_decoder_family() {
        let family = |architecture, rope, requires_attention_bias| Family {
            architecture,
            rope,
            activation: Activation::Silu,
            requires_attention_bias,
            sub_norms: false,
        };
        assert_eq!(
            Variant::Llama.family(),
            family("llama", RopeStyle::Adjacent, false)
        );
        assert_eq!(
            Variant::Qwen2.family(),
            family("qwen2", RopeStyle::HalfSplit, true)
        );
    }
}
