// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SP1's acceptance test: `add` dispatches through the real table.

use rusticide_core::{BackendComponent, DType, Device, DispatchKey, DispatchKeySet, Tensor};
use rusticide_dispatch::{DispatchError, Dispatcher, IValue, KernelFunction, Stack};

const fn cpu_dense() -> DispatchKeySet {
    DispatchKeySet::empty()
        .with_backend(BackendComponent::Cpu)
        .with_functionality(DispatchKey::Dense)
}

fn register_add() {
    Dispatcher::global().register(
        "add",
        cpu_dense(),
        KernelFunction::new(|stack: &mut Stack| {
            let (Some(rhs), Some(lhs)) = (stack.pop_tensor(), stack.pop_tensor()) else {
                return Err(DispatchError::KernelFailed(
                    "add expects two tensors".to_owned(),
                ));
            };
            let out = Tensor::zeros(lhs.shape().dims(), lhs.dtype(), lhs.device())?;
            lhs.storage().with_read_pair(rhs.storage(), |a, b| {
                let mut o = out.storage().write();
                for (slot, (x, y)) in o.iter_mut().zip(a.iter().zip(b.iter())) {
                    *slot = x.wrapping_add(*y);
                }
            });
            stack.push(IValue::Tensor(out));
            Ok(())
        }),
    );
}

#[test]
fn add_dispatches_to_the_cpu_kernel() {
    register_add();
    let a = Tensor::zeros(&[4], DType::U8, Device::cpu()).expect("allocation succeeds");
    let b = Tensor::zeros(&[4], DType::U8, Device::cpu()).expect("allocation succeeds");
    a.storage().write()[0] = 2;
    b.storage().write()[0] = 3;

    let mut stack = Stack::new();
    stack.push(IValue::Tensor(a));
    stack.push(IValue::Tensor(b));
    Dispatcher::global()
        .call("add", &mut stack)
        .expect("op is registered");

    let out = stack.pop_tensor().expect("kernel pushed a tensor");
    assert_eq!(out.storage().read()[0], 5);
}

#[test]
fn add_of_a_tensor_with_itself_does_not_lock_twice() {
    register_add();
    let a = Tensor::zeros(&[2], DType::U8, Device::cpu()).expect("allocation succeeds");
    a.storage().write()[0] = 21;
    let mut stack = Stack::new();
    stack.push(IValue::Tensor(a.clone()));
    stack.push(IValue::Tensor(a));
    Dispatcher::global()
        .call("add", &mut stack)
        .expect("op is registered");
    let out = stack.pop_tensor().expect("kernel pushed a tensor");
    assert_eq!(out.storage().read()[0], 42);
}

#[test]
fn a_kernel_reports_a_missing_argument_instead_of_returning_silently() {
    register_add();
    let only = Tensor::zeros(&[4], DType::U8, Device::cpu()).expect("allocation succeeds");
    let mut stack = Stack::new();
    stack.push(IValue::Tensor(only));
    assert_eq!(
        Dispatcher::global().call("add", &mut stack),
        Err(DispatchError::KernelFailed(
            "add expects two tensors".to_owned()
        ))
    );
}

#[test]
fn calling_an_unregistered_operator_is_an_error() {
    let mut stack = Stack::new();
    assert_eq!(
        Dispatcher::global().call("nonexistent", &mut stack),
        Err(DispatchError::UnknownOperator("nonexistent".to_owned()))
    );
}

#[test]
fn a_kernel_may_re_enter_the_dispatcher() {
    Dispatcher::global().register(
        "reentrant",
        cpu_dense(),
        KernelFunction::new(|_stack: &mut Stack| {
            Dispatcher::global().register(
                "registered_inside",
                cpu_dense(),
                KernelFunction::new(|_| Ok(())),
            );
            let mut inner = Stack::new();
            let result = Dispatcher::global().call("registered_inside", &mut inner);
            assert_eq!(
                result,
                Err(DispatchError::NoKernel {
                    operator: "registered_inside".to_owned()
                })
            );
            Ok(())
        }),
    );
    let t = Tensor::zeros(&[1], DType::U8, Device::cpu()).expect("allocation succeeds");
    let mut stack = Stack::new();
    stack.push(IValue::Tensor(t));
    Dispatcher::global()
        .call("reentrant", &mut stack)
        .expect("kernel re-enters without deadlock");
}
