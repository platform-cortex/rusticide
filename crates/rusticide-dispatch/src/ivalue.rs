// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{DType, Device, Tensor};

/// A dynamically typed value on the dispatch stack.
///
/// The C++ original hand-rolls this as a tagged union with a manual `Tag` field
/// and hand-written payload management. In Rust it is an ordinary enum,
/// which is one of the places the port improves on the original.
#[derive(Debug, Clone)]
pub enum IValue {
    /// The absence of a value.
    None,
    /// A tensor.
    Tensor(Tensor),
    /// A 64-bit signed integer.
    Int(i64),
    /// A 64-bit float.
    Double(f64),
    /// A boolean.
    Bool(bool),
    /// A string.
    Str(String),
    /// A list of integers.
    IntList(Vec<i64>),
    /// A list of tensors.
    TensorList(Vec<Tensor>),
    /// A device.
    Device(Device),
    /// A scalar data type.
    DType(DType),
}

impl IValue {
    /// Borrows the tensor, or `None` if this is another variant.
    #[must_use]
    pub const fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// Returns the integer, or `None` if this is another variant.
    #[must_use]
    pub const fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the float, or `None` if this is another variant.
    #[must_use]
    pub const fn as_double(&self) -> Option<f64> {
        match self {
            Self::Double(d) => Some(*d),
            _ => None,
        }
    }

    /// Returns the boolean, or `None` if this is another variant.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use rusticide_core::{DType, Device, Tensor};

    use super::IValue;

    #[test]
    fn an_ivalue_holds_a_tensor_and_gives_it_back() {
        let t = Tensor::zeros(&[2], DType::F32, Device::cpu()).expect("allocation succeeds");
        let v = IValue::Tensor(t.clone());
        let got = v.as_tensor().expect("holds a tensor");
        assert!(got.shares_storage_with(&t));
    }

    #[test]
    fn asking_for_the_wrong_variant_yields_none() {
        assert!(IValue::Int(3).as_tensor().is_none());
        assert!(IValue::None.as_tensor().is_none());
        assert!(IValue::None.as_int().is_none());
        assert!(IValue::None.as_double().is_none());
        assert!(IValue::None.as_bool().is_none());
    }

    #[test]
    fn scalars_round_trip_through_their_accessors() {
        assert_eq!(IValue::Int(7).as_int(), Some(7));
        assert!(
            IValue::Double(0.5)
                .as_double()
                .is_some_and(|d| (d - 0.5).abs() < f64::EPSILON)
        );
        assert_eq!(IValue::Bool(true).as_bool(), Some(true));
    }
}
