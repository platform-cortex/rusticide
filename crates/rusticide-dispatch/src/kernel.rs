// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::Stack;

/// A registered kernel, called through the boxed convention.
///
/// A kernel pops its arguments from the [`Stack`] and pushes its results
/// back. Cloning is cheap: clones share one function.
#[derive(Clone)]
pub struct KernelFunction(Arc<dyn Fn(&mut Stack) + Send + Sync>);

impl KernelFunction {
    /// Wraps a function as a kernel.
    #[must_use]
    pub fn new<F>(function: F) -> Self
    where
        F: Fn(&mut Stack) + Send + Sync + 'static,
    {
        Self(Arc::new(function))
    }

    /// Runs the kernel against the stack.
    pub fn call(&self, stack: &mut Stack) {
        (self.0)(stack);
    }
}

impl core::fmt::Debug for KernelFunction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("KernelFunction(..)")
    }
}

#[cfg(test)]
mod tests {
    use super::KernelFunction;
    use crate::{IValue, Stack};

    fn doubling_kernel() -> KernelFunction {
        KernelFunction::new(|stack: &mut Stack| {
            if let Some(value) = stack.pop_int() {
                stack.push(IValue::Int(value * 2));
            }
        })
    }

    #[test]
    fn calling_a_kernel_transforms_the_stack() {
        let mut stack = Stack::new();
        stack.push(IValue::Int(21));
        doubling_kernel().call(&mut stack);
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.pop_int(), Some(42));
    }

    #[test]
    fn a_cloned_kernel_runs_the_same_function() {
        let kernel = doubling_kernel();
        let copy = kernel.clone();
        let mut stack = Stack::new();
        stack.push(IValue::Int(1));
        copy.call(&mut stack);
        kernel.call(&mut stack);
        assert_eq!(stack.pop_int(), Some(4));
    }

    #[test]
    fn debug_hides_the_closure() {
        assert_eq!(format!("{:?}", doubling_kernel()), "KernelFunction(..)");
    }
}
