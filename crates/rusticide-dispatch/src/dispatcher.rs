// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use rusticide_core::DispatchKeySet;

use crate::{DispatchError, KernelFunction, OperatorEntry, Stack};

/// The operator registry: routes a call to the kernel its arguments select.
///
/// Most code uses the process-wide [`Dispatcher::global`]; [`Dispatcher::new`]
/// builds an independent registry.
#[derive(Debug, Default)]
pub struct Dispatcher {
    operators: RwLock<HashMap<String, OperatorEntry>>,
}

impl Dispatcher {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The process-wide registry.
    #[must_use]
    pub fn global() -> &'static Self {
        static GLOBAL: OnceLock<Dispatcher> = OnceLock::new();
        GLOBAL.get_or_init(Self::new)
    }

    /// Installs a kernel for an operator under a keyset.
    ///
    /// Registering an existing (operator, keyset) slot replaces its kernel.
    pub fn register(&self, operator: &str, keys: DispatchKeySet, kernel: KernelFunction) {
        self.write()
            .entry(operator.to_owned())
            .or_insert_with(|| OperatorEntry::new(operator))
            .register(keys, kernel);
    }

    /// Calls an operator: unions the keysets of the tensors on the stack,
    /// picks that slot's kernel and runs it.
    ///
    /// The registry lock is released before the kernel runs, so a kernel may
    /// itself call or register operators.
    ///
    /// # Errors
    ///
    /// [`DispatchError::UnknownOperator`] if the operator was never
    /// registered; [`DispatchError::NoKernel`] if it has no kernel for the
    /// computed keyset (including a stack without tensors); whatever error
    /// the kernel itself returns.
    pub fn call(&self, operator: &str, stack: &mut Stack) -> Result<(), DispatchError> {
        let keys = stack.key_set();
        let operators = self.read();
        let found = operators
            .get(operator)
            .map(|entry| entry.lookup(keys).cloned());
        drop(operators);
        let kernel = match found {
            None => return Err(DispatchError::UnknownOperator(operator.to_owned())),
            Some(None) => {
                return Err(DispatchError::NoKernel {
                    operator: operator.to_owned(),
                });
            }
            Some(Some(kernel)) => kernel,
        };
        kernel.call(stack)
    }

    // A poisoned lock is recovered, as in `Storage::read`: a registry map
    // is never left half-updated by a panic elsewhere.
    fn read(&self) -> RwLockReadGuard<'_, HashMap<String, OperatorEntry>> {
        self.operators
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, HashMap<String, OperatorEntry>> {
        self.operators
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use rusticide_core::{BackendComponent, DType, Device, DispatchKey, DispatchKeySet, Tensor};

    use super::Dispatcher;
    use crate::{DispatchError, IValue, KernelFunction, Stack};

    fn cpu_dense() -> DispatchKeySet {
        DispatchKeySet::empty()
            .with_backend(BackendComponent::Cpu)
            .with_functionality(DispatchKey::Dense)
    }

    fn tensor_on(device: Device) -> Tensor {
        Tensor::zeros(&[1], DType::U8, device).expect("allocation succeeds")
    }

    fn pushes(value: i64) -> KernelFunction {
        KernelFunction::new(move |stack: &mut Stack| {
            stack.push(IValue::Int(value));
            Ok(())
        })
    }

    fn cpu_stack() -> Stack {
        let mut stack = Stack::new();
        stack.push(IValue::Tensor(tensor_on(Device::cpu())));
        stack
    }

    #[test]
    fn a_call_runs_the_kernel_for_the_stacks_keys() {
        let dispatcher = Dispatcher::new();
        dispatcher.register("op", cpu_dense(), pushes(7));
        let mut stack = cpu_stack();
        assert_eq!(dispatcher.call("op", &mut stack), Ok(()));
        assert_eq!(stack.pop_int(), Some(7));
    }

    #[test]
    fn registering_again_replaces_the_kernel() {
        let dispatcher = Dispatcher::new();
        dispatcher.register("op", cpu_dense(), pushes(1));
        dispatcher.register("op", cpu_dense(), pushes(2));
        let mut stack = cpu_stack();
        assert_eq!(dispatcher.call("op", &mut stack), Ok(()));
        assert_eq!(stack.pop_int(), Some(2));
    }

    #[test]
    fn registering_another_keyset_keeps_the_existing_kernels() {
        let dispatcher = Dispatcher::new();
        let gpu_dense = DispatchKeySet::empty()
            .with_backend(BackendComponent::Gpu)
            .with_functionality(DispatchKey::Dense);
        dispatcher.register("op", cpu_dense(), pushes(1));
        dispatcher.register("op", gpu_dense, pushes(2));
        let mut stack = cpu_stack();
        assert_eq!(dispatcher.call("op", &mut stack), Ok(()));
        assert_eq!(stack.pop_int(), Some(1));
    }

    #[test]
    fn an_unknown_operator_is_reported_by_name() {
        let mut stack = cpu_stack();
        assert_eq!(
            Dispatcher::new().call("nope", &mut stack),
            Err(DispatchError::UnknownOperator("nope".to_owned()))
        );
    }

    #[test]
    fn a_keyset_without_a_kernel_is_an_error() {
        let dispatcher = Dispatcher::new();
        dispatcher.register("op", cpu_dense(), pushes(1));
        let mut stack = Stack::new();
        stack.push(IValue::Tensor(tensor_on(Device::gpu(0))));
        let expected = Err(DispatchError::NoKernel {
            operator: "op".to_owned(),
        });
        assert_eq!(dispatcher.call("op", &mut stack), expected);
        assert_eq!(dispatcher.call("op", &mut Stack::new()), expected);
    }

    #[test]
    fn a_kernel_error_is_propagated_to_the_caller() {
        let dispatcher = Dispatcher::new();
        let failing = KernelFunction::new(|_stack: &mut Stack| {
            Err(DispatchError::KernelFailed("bad input".to_owned()))
        });
        dispatcher.register("op", cpu_dense(), failing);
        let mut stack = cpu_stack();
        assert_eq!(
            dispatcher.call("op", &mut stack),
            Err(DispatchError::KernelFailed("bad input".to_owned()))
        );
    }

    #[test]
    fn a_panic_while_holding_the_registry_lock_leaves_it_usable() {
        let dispatcher = std::sync::Arc::new(Dispatcher::new());
        let victim = std::sync::Arc::clone(&dispatcher);
        let outcome = std::thread::spawn(move || {
            let _guard = victim.write();
            panic!("simulated failure while holding the write guard");
        })
        .join();

        assert!(outcome.is_err(), "the writer thread must have panicked");
        assert!(dispatcher.operators.is_poisoned(), "precondition: poisoned");
        dispatcher.register("op", cpu_dense(), pushes(3));
        let mut stack = cpu_stack();
        assert_eq!(dispatcher.call("op", &mut stack), Ok(()));
        assert_eq!(stack.pop_int(), Some(3));
    }

    #[test]
    fn the_global_registry_is_one_shared_instance() {
        assert!(std::ptr::eq(Dispatcher::global(), Dispatcher::global()));
    }
}
