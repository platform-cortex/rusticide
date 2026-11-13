// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;
use rusticide_model::{Loader, Model, ModelError};

use super::load;
use crate::testing::{ATTN_SUB_NORM, FFN_SUB_NORM, VOCAB, ramp, tiny};

#[test]
fn load_returns_a_boxed_model() {
    let loader: Loader = load;
    let mut model = loader(&tiny().build()).unwrap();
    assert_eq!(model.vocab_size(), VOCAB);
    assert_eq!(model.context_length(), 16);
    assert_eq!(model.forward(1, 0).unwrap().len(), VOCAB);
}

#[test]
fn load_requires_both_sub_norms() {
    for name in [ATTN_SUB_NORM, FFN_SUB_NORM] {
        match load(&tiny().without_tensor(name).build()) {
            Err(ModelError::MissingTensor { name: missing }) => assert_eq!(missing, name),
            other => panic!("unexpected: {:?}", other.map(|_| ())),
        }
    }
}

#[test]
fn load_rejects_mis_sized_sub_norms() {
    let cases: [(&str, Tensor, usize); 2] =
        [(ATTN_SUB_NORM, ramp(8), 4), (FFN_SUB_NORM, ramp(4), 8)];
    for (name, tensor, len) in cases {
        let got = tensor.shape().dims().to_vec();
        match load(&tiny().with_tensor(name, tensor).build()) {
            Err(ModelError::ShapeMismatch {
                name: wrong,
                expected,
                got: found,
            }) => {
                assert_eq!(wrong, name);
                assert_eq!(expected, vec![len]);
                assert_eq!(found, got);
            }
            other => panic!("unexpected: {:?}", other.map(|_| ())),
        }
    }
}

fn prompt_errors(model: &mut dyn Model) -> [String; 2] {
    let empty = model.forward_batch(&[], 0).unwrap_err();
    let long = model.forward_batch(&[1; 17], 0).unwrap_err();
    [format!("{empty:?}"), format!("{long:?}")]
}

#[test]
fn empty_and_over_long_prompts_fail_as_in_llama() {
    let mut bitnet = load(&tiny().build()).unwrap();
    let mut llama = rusticide_llama::load(&rusticide_llama::testing::tiny_source()).unwrap();
    let errors = prompt_errors(bitnet.as_mut());
    assert_eq!(errors, prompt_errors(llama.as_mut()));
    assert_eq!(
        errors,
        [
            "InvalidConfig { reason: \"no tokens to forward\" }".to_owned(),
            "ContextFull { limit: 16 }".to_owned(),
        ]
    );
}
