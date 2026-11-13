<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Plano de Implementacao do BitNet b1.58

> **Para agentes de execucao:** SUB-SKILL OBRIGATORIA: use
> superpowers:subagent-driven-development (recomendado) ou
> superpowers:executing-plans para implementar este plano tarefa a tarefa. Os
> passos usam a sintaxe de checkbox (`- [ ]`) para acompanhamento.

**Objetivo:** `rusticide run` executa o BitNet b1.58 2B 4T (`i2_s`) e
reproduz os tokens gulosos do bitnet.cpp, com os blocos do decodificador
compartilhados com `llama` e `qwen2` por meio de um novo crate
`rusticide-transformer`.

**Arquitetura:** primeiro as fixtures de referencia (go/no-go). Depois uma
mudanca pura do decodificador (atencao, feed-forward, normas, helpers de
lote, o laco de camadas) de `rusticide-llama` para `rusticide-transformer`,
provada por logits identicos. Depois o `i2_s`: uma quantizacao int8 de
ativacao por linha, um formato com trailer de escala por tensor, um kernel
ternario portable atras de `Matrix`/`Weights`. Por fim dois ganchos no
decodificador compartilhado (ReLU ao quadrado, sub-normas) e uma familia
`rusticide-bitnet` fina, registrada na facade.

**Stack Tecnologica:** Rust 2024, rayon, `rusticide-bench`, bitnet.cpp (fork
legado, compilado fora do repositorio) como referencia.

**Especificacao:** `docs/plan/2026-11-12-rusticide-bitnet-design.pt-BR.md`
(leia-a; a secao "Fatos do reconhecimento" e o `recon.md` do ledger trazem o
layout de bytes e as formulas de que toda tarefa `i2_s` depende).

## Restricoes Globais

- MSRV 1.98, pin do toolchain inalterado; nenhuma API mais nova que 1.98.
- `unsafe` somente em `rusticide-alloc`/`rusticide-simd`; este plano nao
  adiciona nenhum.
- Nenhum `unwrap`/`expect` em codigo de producao; 100% de cobertura; nenhum
  panic diante de um arquivo GGUF hostil.
- Somente ASCII, arquivos com menos de 300 linhas (testes incluidos),
  cabecalhos SPDX, documentos bilingues, commits convencionais assinados
  (`git commit -S`), sem trailers de atribuicao.
- Nunca `target-cpu=native` na configuracao commitada.
- Ledger: `.superpowers/sdd/2026-11-12-rusticide-bitnet-plan/`
  (`progress.md`, `recon.md`, ignorados pelo git).
- Os arquivos de modelo ficam em `$RUSTICIDE_MODELS` (`~/models` nas duas
  maquinas): `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`,
  `qwen2.5-0.5b-instruct-q4_k_m.gguf` e (Tarefa 1)
  `bitnet-b1.58-2b-4t-i2_s.gguf`, SHA-256
  `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`.
- O iMac Pro e `ssh elio@192.168.18.146` (as ferramentas precisam de
  `zsh -lc '...'`). Sincronize o worktree com `rsync -a --delete --exclude
  target --exclude .git --exclude mise.local.toml ./
  elio@192.168.18.146:~/rusticide-wt/` e compile la. Nunca faca commit nem
  push a partir do iMac Pro.
- As tarefas de refatoracao (3-5) e a tarefa de ganchos (9) devem deixar os
  logits de TinyLlama e Qwen2.5 bit a bit identicos: o digest de logits
  (Tarefa 2) confere no M1.
- `bench:gate` no M1 apos toda tarefa que toque kernels, o decodificador ou o
  escalonamento; maquina quieta, na tomada.

## Foco da Revisao

1. Um tensor `i2_s` hostil: um intervalo de bytes menor que `numel/4 + 32`,
   ou uma largura de linha que nao e multiplo de 128: um erro tipado no
   load, nunca um panic nem uma leitura fora dos limites. Testes da Tarefa 7,
   alvo de fuzz estendido.
2. Uma linha de ativacao toda zero: o piso `1e-5` da uma escala finita, todo
   quant e 0 e o produto e exatamente 0.0, nunca NaN. Tarefa 6.
3. Uma ativacao exatamente no meio entre dois quants: arredonda metade para
   par (`2.5 -> 2`, `3.5 -> 4`), como o `nearest_int` legado do bitnet.cpp.
   Tarefa 6.
4. O codigo `3` (nao usado pelo formato) em um arquivo hostil: decodifica
   como `code - 1 = +2` tanto no dequantize quanto no kernel, sem panic.
   Tarefas 7-8.
5. Um arquivo `bitnet-b1.58` sem uma sub-norma, ou um prompt vazio ou longo
   demais: `MissingTensor`, `InvalidConfig` e `ContextFull` como em `llama`.
   Tarefa 10.

---

## Parte 1: a referencia (go/no-go)

### Tarefa 1: fixtures de referencia do BitNet

**Arquivos:**

- Criar: `crates/rusticide/tests/fixtures/bitnet-b1.58-2b-4t-i2_s.json`
- Ledger: `fixtures-capture.md` (comandos, versoes, saidas)

**Interfaces:**

- Produz: uma fixture com o esquema de
  `crates/rusticide-llama/tests/fixtures/*.json` (`model.repo`,
  `model.file` = `bitnet-b1.58-2b-4t-i2_s.gguf`, `model.sha256`,
  `reference.tool`/`version`/`settings`, `cases[].prompt`,
  `prompt_tokens`, `generated_tokens` de 32).
- [ ] **Passo 1:** No iMac Pro, clone `microsoft/BitNet` fora de qualquer
  repositorio (`~/bitnet-ref`), faca checkout de `3b04140` com seu submodulo
  (`Eddie-Wang1120/llama.cpp`) e compile conforme `setup_env.py` com
  `-q i2_s` e o clang de `/usr/local/opt/llvm/bin`. Aponte para o arquivo ja
  presente no cache do HF (copie ou crie um symlink como
  `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`); a conversao deve ser
  ignorada. Registre os dois commits.
- [ ] **Passo 2:** Descubra como os binarios legados imprimem os ids de
  token (por exemplo `llama-cli --verbose-prompt`, `llama-tokenize`, ou
  `llama-server` `/completion` com `n_probs`/saida de tokens). Capture, com
  `--temp 0 -n 32 -c 2048 -ngl 0` e BOS no inicio, ao menos estes prompts:
  `The capital of France is`, `def fibonacci(n):`, e um prompt em portugues
  com acentos, escrito no JSON como
  `"A capital do Brasil \u00e9 Bras\u00edlia, e a maior cidade \u00e9"`
  (os escapes mantem o arquivo ASCII; a ferramenta recebe o UTF-8 acentuado).
- [ ] **Passo 3:** Rode a mesma captura duas vezes: os ids de token devem se
  repetir. Depois baixe o arquivo para `~/models/bitnet-b1.58-2b-4t-i2_s.gguf`
  nas duas maquinas e confira o SHA-256.
- [ ] **Passo 4 (go/no-go):** Se a build legada falhar ou nao se repetir,
  tente o BitNet `765741d` no M1 (NEON, mesmo layout 128); se as duas
  falharem ou divergirem, pare e reporte ao dono. Nenhum outro oraculo.
- [ ] **Passo 5: Commit** `test(bitnet): capture the bitnet.cpp fixtures`.

## Parte 2: `rusticide-transformer` (mudanca pura)

### Tarefa 2: digest de logits e o teste de referencia no nivel da facade

**Arquivos:**

- Ledger: `logit_digest.rs` (nunca commitado)
- Mover: `crates/rusticide-llama/tests/reference.rs` e `fixtures/` ->
  `crates/rusticide/tests/reference.rs`, `crates/rusticide/tests/fixtures/`
- Criar: `crates/rusticide/src/models.rs`
- Modificar: `crates/rusticide/src/cli/cmd/run.rs:69`, `mise.toml`
  (`test:reference` executa `-p rusticide --features models --test reference`)

**Interfaces:**

- Produz: `rusticide::models::registry() -> rusticide_model::Registry`
  (feature `models`; por enquanto `rusticide_llama::registry()`), usado pela
  CLI e pelo teste de referencia. A Tarefa 10 adiciona o BitNet aqui.
- Produz: o harness de digest, um teste de integracao mantido no ledger e
  copiado para `crates/rusticide/tests/` apenas enquanto roda: para cada
  prompt de fixture do TinyLlama e do Qwen2.5, `forward_batch(prompt)` e
  depois 8 passos `forward` gulosos, FNV-1a 64 sobre o `to_bits()` de cada
  logit; `RUSTICIDE_DIGEST=record` grava `logit-digest-<machine>.txt` no
  ledger, `check` compara e falha em qualquer diferenca.
- [ ] **Passo 1:** Mova o teste de referencia e as fixtures; troque-o e o
  `run.rs` para `rusticide::models::registry()`; `test:reference` verde
  (2/2) no M1.
- [ ] **Passo 2:** Grave o digest no M1 duas vezes; os dois devem ser iguais
  (determinismo), senao pare: a prova seria nula.
- [ ] **Passo 3: Commit** `test: run the reference parity from the facade`.

### Tarefa 3: o crate e seus helpers folha

**Arquivos:**

- Criar: `crates/rusticide-transformer/` (`Cargo.toml` a partir do de
  `rusticide-llama`, `src/lib.rs`)
- Mover de `rusticide-llama/src`: `batch.rs` (`for_each_row`,
  `zip_rows`, `PARALLEL_MIN_TOKENS`), `layout.rs` (`to_token_major`),
  `RopeStyle` de `variant.rs`, `normalize`/`normalize_rows` de
  `model.rs` (novo `norm.rs`), com seus testes.

**Interfaces:**

- Produz: `rusticide_transformer::{for_each_row, zip_rows,
  PARALLEL_MIN_TOKENS, to_token_major, RopeStyle, normalize,
  normalize_rows}`, mesmas assinaturas de hoje, agora `pub`.
- [ ] **Passo 1:** Mova; `rusticide-llama` os importa. Nenhuma mudanca de
  corpo.
- [ ] **Passo 2:** `cargo test -p rusticide-transformer -p rusticide-llama`,
  clippy, digest `check` no M1.
- [ ] **Passo 3: Commit** `refactor(transformer): host the decoder helpers`.

### Tarefa 4: atencao e feed-forward

**Arquivos:**

- Mover: `attention.rs`, `feed_forward.rs` e seus testes para
  `rusticide-transformer`.
- Criar: `crates/rusticide-transformer/src/config.rs`,
  `src/weights.rs`

**Interfaces:**

- Produz: `pub struct DecoderConfig { embedding_length, block_count,
  feed_forward_length, head_count, head_count_kv, head_dim,
  context_length, vocab_size: usize; rope_freq_base, rms_norm_eps: f32;
  rope: RopeStyle; activation: Activation }` com `kv_dim()`.
- Produz: `pub enum Activation { Silu }` (a Tarefa 9 adiciona
  `ReluSquared`).
- Produz: `pub struct BlockWeights` (os campos de hoje, movidos) mais
  `attn_sub_norm: Option<Vec<f32>>`, `ffn_sub_norm: Option<Vec<f32>>`,
  ambos `None` ate a Tarefa 9 le-los.
- `Attention::new(&DecoderConfig)`, `FeedForward::new(&DecoderConfig)`,
  assinaturas de `forward` inalteradas no mais.
- [ ] **Passo 1:** Mova; `LlamaConfig` ganha
  `fn decoder(&self) -> DecoderConfig`. Nenhuma mudanca de corpo.
- [ ] **Passo 2:** Testes, clippy, digest `check`.
- [ ] **Passo 3: Commit** `refactor(transformer): host attention and the
  feed-forward`.

### Tarefa 5: o modelo decodificador

**Arquivos:**

- Mover: o struct `Llama`, `forward_on_pool`, `embed` e o impl de `Model`
  para `crates/rusticide-transformer/src/decoder.rs` como `Decoder`;
  `LlamaWeights` para `DecoderWeights`.
- Manter em `rusticide-llama`: `LlamaConfig` (parsing de metadados),
  `Variant`, carga de pesos por nome de tensor, `load`/`load_qwen2`,
  `registry`.

**Interfaces:**

- Produz: `pub struct Decoder`; `Decoder::new(config: DecoderConfig,
  weights: DecoderWeights) -> rusticide_model::Result<Self>`;
  `impl Model for Decoder`. `pub struct DecoderWeights { token_embd:
  Matrix, blocks: Vec<BlockWeights>, output_norm: Vec<f32>, output:
  Option<Matrix> }`.
- [ ] **Passo 1:** Mova; `rusticide_llama::load` constroi um `Decoder`.
- [ ] **Passo 2:** `mise run ci:quick` (100%), digest `check` no M1,
  `test:reference` 2/2 no M1 e (sincronizado) no iMac Pro, `bench:gate`
  com saida 0 nas duas maquinas.
- [ ] **Passo 3:** `rusticide-llama` nao tem codigo de atencao, norma nem
  lote: `grep -rn "rms_norm\|softmax\|matmul_batch"
  crates/rusticide-llama/src` nao imprime nada.
- [ ] **Passo 4: Commit** `refactor(transformer): host the decoder model`.

## Parte 3: `i2_s`

### Tarefa 6: quantizacao de ativacao por linha

**Arquivos:**

- Modificar: `crates/rusticide-kernels/src/q8_input.rs` (+ arquivo de testes)

**Interfaces:**

- Produz: `Q8Kind::Row`: uma escala para a linha inteira, a formula legada
  do bitnet.cpp; `scales` guarda `s` (um divisor) repetido por bloco de 32,
  `sums` por bloco como hoje; `dequantize` divide por `s` para `Row`.

```rust
// max in f64 from 1e-5; s = (127 / max) as f32; q = round-half-even(x * s)
// clamped to [-128, 127]
```

- [ ] **Passo 1: Testes que falham:** `row_scale_follows_the_legacy_formula`
  (`[0.5, -2.0, 1.0]` -> `s == (127.0_f64 / 2.0) as f32`, quants
  `[32, -127, 64]`); `an_all_zero_row_quantizes_to_zeros`
  (`s == (127.0 / 1e-5) as f32`, quants 0); `halfway_rounds_to_even`
  (uma entrada construida para que `x * s` seja exatamente `2.5`/`3.5` ->
  `2`/`4`); `row_clamps_to_i8` (`-128` alcancavel somente por clamp); as
  somas igualam as somas de quants por bloco.
- [ ] **Passo 2:** Implemente; arredonde metade para par com
  `f32::round_ties_even`.
- [ ] **Passo 3:** Testes, clippy; prove cada teste quebrando sua linha.
- [ ] **Passo 4: Commit** `feat(kernels): quantize activations per row`.

### Tarefa 7: o formato `i2_s`

**Arquivos:**

- Criar: `crates/rusticide-quantize/src/i2_s.rs` (+ testes)
- Modificar: `crates/rusticide-quantize/src/format.rs` (contrato),
  dimensionamento de extents em `crates/rusticide-io` se ainda nao passar
  por `byte_len`, `crates/rusticide-fuzz` (um alvo que carrega um tensor
  `i2_s`)

**Interfaces:**

- Produz: `BlockFormat` ganha `trailer_bytes: usize` (0 para todo formato
  existente, 32 para `I2_S`) e
  `apply_trailer: Option<fn(trailer: &[u8], values: &mut [f32])>`;
  `byte_len(numel) = blocks * block_bytes + trailer_bytes`;
  `pub const fn trailer_bytes(&self) -> usize`.
- Produz: entrada de tabela `I2_S`, id de tipo 36, tamanho de bloco 128,
  bytes de bloco 32, `dequantize_block` grava `code - 1` conforme o layout
  (o byte `k` -> pesos `k`, `k+32`, `k+64`, `k+96`, bits altos primeiro),
  `apply_trailer` multiplica pelo f32 no offset 0 do trailer.
- Produz: `pub fn i2_s_scale(trailer: &[u8]) -> Option<f32>`.
- [ ] **Passo 1: Testes que falham:** um bloco de 128 pesos montado a mao
  cujo byte `k` codifica os pesos `k..k+96` decodifica para a sequencia
  `-1/0/+1` esperada; o codigo `3` decodifica para `+2`; `byte_len(6_553_600)
  == 1_638_432`; um tensor um byte menor -> `DataTooShort`; numel que nao e
  multiplo de 128 -> `UnalignedShape`; dequantize aplica a escala do
  trailer.
- [ ] **Passo 2:** Implemente; os testes dos formatos existentes
  inalterados.
- [ ] **Passo 3:** O alvo de fuzz roda 60 s limpo (`mise run fuzz` conforme
  sua tarefa); testes, clippy, cobertura das novas linhas.
- [ ] **Passo 4: Commit** `feat(quantize): read the i2_s format`.

### Tarefa 8: o kernel ternario portable

**Arquivos:**

- Criar: `crates/rusticide-kernels/src/portable/ternary.rs` (+ testes)
- Modificar: `crates/rusticide-kernels/src/matmul.rs`,
  `crates/rusticide-model/src/weights.rs`, `crates/rusticide-bench`
  (linhas de kernel `kernel/I2_S/row`, `kernel/I2_S/rows8`)

**Interfaces:**

- Produz: `Weights::Ternary { data: &[u8], row_bytes: usize, scale:
  f32 }`; o escalonador quantiza cada token com `Q8Kind::Row` e grava
  `(dot as f32) / s * scale` por linha (a ordem legada: inteiro primeiro,
  depois divide por `s`, depois multiplica pela escala do peso).
- Produz: `pub(crate) fn ternary_dot(row: &[u8], input: &Q8Input) -> i32`,
  o inteiro exato `sum((code - 1) * q)`.
- `Matrix::load` constroi `Storage::Ternary` quando o formato tem trailer
  (escala lida uma vez no load), `copy_row` decodifica `code - 1` vezes a
  escala.
- [ ] **Passo 1: Testes que falham:** `ternary_dot` iguala um dequantize
  ingenuo seguido de produto inteiro em linhas aleatorias (os quatro
  codigos); uma `Matrix` de um tensor `i2_s` sintetico da um `matvec` igual a
  `(int / s) * scale` calculado no teste; uma linha zero produz 0.0; lotes
  de varios tokens igualam chamadas de um token bit a bit.
- [ ] **Passo 2:** Implemente (linhas em paralelo como os outros formatos).
- [ ] **Passo 3:** As linhas de bench aparecem; `bench:gate` no M1 nao
  mostra regressao em nenhuma linha existente (as novas ainda nao estao na
  baseline).
- [ ] **Passo 4: Commit** `feat(kernels): run i2_s products portably`.

## Parte 4: BitNet

### Tarefa 9: ReLU ao quadrado e sub-normas no decodificador

**Arquivos:**

- Modificar: `crates/rusticide-kernels/src/portable/elementwise.rs`,
  `facade.rs`; atencao e feed-forward de `crates/rusticide-transformer`.

**Interfaces:**

- Produz: `rusticide_kernels::relu_squared_assign(values: &mut [f32])`
  (`max(x, 0)^2`, `0.0` para negativo e `-0.0`).
- Produz: `Activation::ReluSquared`; o feed-forward a aplica no lugar da
  SiLU. `attn_sub_norm` (RMS com o eps do bloco, depois o peso) nas cabecas
  misturadas antes de `attn_output`; `ffn_sub_norm` na linha oculta
  com gate antes de `ffn_down`; cada uma somente quando `Some`.
- [ ] **Passo 1: Testes que falham:** `relu_squared_assign` em
  `[-2.0, -0.0, 0.0, 1.5, 3.0]` -> `[0.0, 0.0, 0.0, 2.25, 9.0]`; um
  decodificador pequeno com sub-normas iguala um forward composto a mao; com
  ambas `None` as saidas igualam as da Tarefa 5.
- [ ] **Passo 2:** Implemente; digest `check` no M1 (caminho llama
  intocado).
- [ ] **Passo 3: Commit** `feat(transformer): add squared ReLU and
  sub-norms`.

### Tarefa 10: a familia `bitnet-b1.58`

**Arquivos:**

- Criar: `crates/rusticide-bitnet/` (`config.rs`, `weights.rs`,
  `lib.rs`, testes, `testing.rs` para uma fonte `bitnet-b1.58` minuscula
  sintetica como a de `rusticide-llama`)
- Modificar: `crates/rusticide/src/models.rs`, `Cargo.toml` da facade

**Interfaces:**

- Produz: `rusticide_bitnet::load(source: &dyn ModelSource) ->
  Result<Box<dyn Model>>`, construindo um `Decoder` com `RopeStyle` NEOX
  (meio a meio), `Activation::ReluSquared`, sub-normas lidas de
  `blk.N.attn_sub_norm.weight` / `blk.N.ffn_sub_norm.weight`, `token_embd`
  amarrado (F16) como saida. Chaves de metadados sob `bitnet-b1.58.`.
- `rusticide::models::registry()` ganha
  `.with("bitnet-b1.58", rusticide_bitnet::load)`.
- [ ] **Passo 1: Testes que falham:** config a partir de uma fonte sintetica
  (chaves, padroes, erros); uma sub-norma ausente -> `MissingTensor`; o
  registry lista `bitnet-b1.58`; prompt vazio e prompt longo demais dao os
  mesmos erros que `llama`.
- [ ] **Passo 2:** Implemente.
- [ ] **Passo 3:** Com o arquivo real no M1: os prompts das fixtures
  tokenizam para os ids da fixture (todos).
- [ ] **Passo 4 (checkpoint):** `rusticide run` no M1, 64 tokens: registre
  tokens/s de decode. Abaixo de 2 tok/s: reporte ao dono (NEON `i2_s` neste
  marco ou nao) antes da Tarefa 11.
- [ ] **Passo 5: Commit** `feat(bitnet): add the bitnet-b1.58 family`.

### Tarefa 11: paridade

**Arquivos:**

- Modificar: `crates/rusticide/tests/reference.rs` (um terceiro caso),
  `mise.toml` (descricao de `test:reference`), `AGENTS.md`
  (`RUSTICIDE_MODELS` contem tres arquivos)
- [ ] **Passo 1:** `test:reference` 3/3 no M1 (`neon` e
  `RUSTICIDE_KERNELS=portable`) e no iMac Pro (`avx2`, `portable`).
  Uma divergencia alem da regra de quase empate e um bug: depure (camada a
  camada contra os valores intermediarios do bitnet.cpp se preciso), nao
  relaxe.
- [ ] **Passo 2: Commit** `test(bitnet): check greedy parity with
  bitnet.cpp`.

## Finalizacao

### Tarefa 12: bench, resultados, docs, release

- [ ] **Passo 1:** `rusticide-bench` ganha `e2e/bitnet/prefill` e
  `e2e/bitnet/decode` quando o arquivo esta presente; regrave as duas
  baselines (quieta, na tomada), cada uma em seu proprio commit
  `perf(bench): re-record the <machine> baseline`.
- [ ] **Passo 2:** Preencha os Resultados do design (EN e pt-BR): paridade
  por prompt, prefill/decode do BitNet por maquina contra os numeros do
  bitnet.cpp (`llama-bench` da build da Tarefa 1), a prova da refatoracao,
  follow-ups (`i2_s` SIMD, atencao/RoPE). Status concluido. O "Current
  plan" do `AGENTS.md` aponta para o proximo passo.
- [ ] **Passo 3:** `mise run ci:quick`, `security`, `markdownlint` limpos.
- [ ] **Passo 4: Commit** `docs: record the BitNet results`.
- [ ] **Passo 5:** `mise run release:prepare` (0.10.0) depois que o dono
  aprovar o merge; `chore(release): prepare for v0.10.0`.
