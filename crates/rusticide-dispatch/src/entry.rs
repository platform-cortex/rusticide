// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::DispatchKeySet;

use crate::KernelFunction;

/// One operator's kernel table: a flat slot per dispatch key.
///
/// Slots are indexed by [`DispatchKeySet::table_index`] and the table holds
/// [`DispatchKeySet::table_size`] slots.
#[derive(Debug)]
pub struct OperatorEntry {
    name: String,
    kernels: Vec<Option<KernelFunction>>,
}

impl OperatorEntry {
    /// Creates an entry with no kernels.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            kernels: vec![None; DispatchKeySet::table_size()],
        }
    }

    /// The operator's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Installs a kernel for a keyset, replacing any kernel already there.
    ///
    /// A keyset with no functionality has no slot and is ignored.
    pub fn register(&mut self, keys: DispatchKeySet, kernel: KernelFunction) {
        if let Some(slot) = keys.table_index().and_then(|i| self.kernels.get_mut(i)) {
            *slot = Some(kernel);
        }
    }

    /// The kernel for a keyset, or `None` if its slot is empty.
    #[must_use]
    pub fn lookup(&self, keys: DispatchKeySet) -> Option<&KernelFunction> {
        keys.table_index()
            .and_then(|i| self.kernels.get(i))
            .and_then(Option::as_ref)
    }
}

#[cfg(test)]
mod tests {
    use rusticide_core::{BackendComponent, DispatchKey, DispatchKeySet};

    use super::OperatorEntry;
    use crate::{IValue, KernelFunction, Stack};

    fn keys(backend: BackendComponent) -> DispatchKeySet {
        DispatchKeySet::empty()
            .with_backend(backend)
            .with_functionality(DispatchKey::Dense)
    }

    fn constant(value: i64) -> KernelFunction {
        KernelFunction::new(move |stack: &mut Stack| {
            stack.push(IValue::Int(value));
            Ok(())
        })
    }

    fn run(entry: &OperatorEntry, keyset: DispatchKeySet) -> Option<i64> {
        let mut stack = Stack::new();
        entry.lookup(keyset)?.call(&mut stack).ok()?;
        stack.pop_int()
    }

    #[test]
    fn a_registered_kernel_is_found_only_under_its_own_keys() {
        let mut entry = OperatorEntry::new("add");
        assert_eq!(entry.name(), "add");
        entry.register(keys(BackendComponent::Cpu), constant(1));
        assert_eq!(run(&entry, keys(BackendComponent::Cpu)), Some(1));
        assert!(entry.lookup(keys(BackendComponent::Gpu)).is_none());
    }

    #[test]
    fn each_backend_registers_into_its_own_slot() {
        let mut entry = OperatorEntry::new("add");
        entry.register(keys(BackendComponent::Gpu), constant(2));
        assert!(entry.lookup(keys(BackendComponent::Cpu)).is_none());
        assert_eq!(run(&entry, keys(BackendComponent::Gpu)), Some(2));
        entry.register(keys(BackendComponent::Cpu), constant(1));
        assert_eq!(run(&entry, keys(BackendComponent::Gpu)), Some(2));
        assert_eq!(run(&entry, keys(BackendComponent::Cpu)), Some(1));
    }

    #[test]
    fn registering_the_same_slot_replaces_the_kernel() {
        let mut entry = OperatorEntry::new("add");
        entry.register(keys(BackendComponent::Cpu), constant(1));
        entry.register(keys(BackendComponent::Cpu), constant(2));
        assert_eq!(run(&entry, keys(BackendComponent::Cpu)), Some(2));
    }

    #[test]
    fn an_empty_keyset_has_no_slot() {
        let mut entry = OperatorEntry::new("add");
        entry.register(DispatchKeySet::empty(), constant(1));
        assert!(entry.lookup(DispatchKeySet::empty()).is_none());
        assert!(entry.lookup(keys(BackendComponent::Cpu)).is_none());
    }
}
