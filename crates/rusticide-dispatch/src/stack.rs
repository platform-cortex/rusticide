// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::IValue;
use rusticide_core::Tensor;

/// The argument and return stack of the boxed calling convention.
///
/// Typed pops return `None` and leave the stack untouched when the top
/// value is of another variant, so a caller can try a different pop.
#[derive(Debug, Clone, Default)]
pub struct Stack {
    values: Vec<IValue>,
}

impl Stack {
    /// Creates an empty stack.
    #[must_use]
    pub const fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Pushes a value on top of the stack.
    pub fn push(&mut self, value: IValue) {
        self.values.push(value);
    }

    /// Pops the top value, or `None` if the stack is empty.
    pub fn pop(&mut self) -> Option<IValue> {
        self.values.pop()
    }

    /// Pops the top value if it is a tensor; otherwise leaves it in place.
    pub fn pop_tensor(&mut self) -> Option<Tensor> {
        let top = self.values.pop_if(|value| value.as_tensor().is_some())?;
        top.as_tensor().cloned()
    }

    /// Pops the top value if it is an integer; otherwise leaves it in place.
    pub fn pop_int(&mut self) -> Option<i64> {
        self.values
            .pop_if(|value| value.as_int().is_some())?
            .as_int()
    }

    /// Returns the number of values on the stack.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the stack holds no values.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use rusticide_core::{DType, Device, Tensor};

    use super::Stack;
    use crate::IValue;

    fn tensor() -> Tensor {
        Tensor::zeros(&[1], DType::F32, Device::cpu()).expect("allocation succeeds")
    }

    #[test]
    fn a_new_stack_is_empty_and_pops_nothing() {
        let mut stack = Stack::default();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
        assert!(stack.pop().is_none());
        assert!(stack.pop_int().is_none());
        assert!(stack.pop_tensor().is_none());
    }

    #[test]
    fn values_pop_in_last_in_first_out_order() {
        let mut stack = Stack::new();
        stack.push(IValue::Int(1));
        stack.push(IValue::Int(2));
        assert_eq!(stack.len(), 2);
        assert!(!stack.is_empty());
        assert_eq!(stack.pop_int(), Some(2));
        assert_eq!(stack.pop_int(), Some(1));
        assert!(stack.is_empty());
    }

    #[test]
    fn a_tensor_pops_back_sharing_storage() {
        let t = tensor();
        let mut stack = Stack::new();
        stack.push(IValue::Tensor(t.clone()));
        let got = stack.pop_tensor().expect("top is a tensor");
        assert!(got.shares_storage_with(&t));
        assert!(stack.is_empty());
    }

    #[test]
    fn a_typed_pop_on_the_wrong_variant_keeps_the_value() {
        let mut stack = Stack::new();
        stack.push(IValue::Int(5));
        assert!(stack.pop_tensor().is_none());
        stack.push(IValue::Tensor(tensor()));
        assert!(stack.pop_int().is_none());
        assert_eq!(stack.len(), 2);
        assert!(stack.pop_tensor().is_some());
        assert_eq!(stack.pop().and_then(|v| v.as_int()), Some(5));
    }
}
