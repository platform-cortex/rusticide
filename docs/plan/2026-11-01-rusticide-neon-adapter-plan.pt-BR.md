<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Plano de Implementacao do marco do adaptador NEON

> **Para agentes de execucao:** SUB-SKILL OBRIGATORIA: use
> superpowers:subagent-driven-development (recomendado) ou
> superpowers:executing-plans para implementar este plano tarefa a tarefa. Os
> passos usam a sintaxe de checkbox (`- [ ]`) para acompanhamento.

**Objetivo:** Fechar a maior parte da diferenca de prefill em relacao ao
llama.cpp no M1 Pro, em tres partes com gates: trabalho paralelo por token,
amostragem top-k e um adaptador de kernels NEON.

**Arquitetura:** As partes 1 e 2 sao Rust seguro em `rusticide-kernels`,
`rusticide-llama`, `rusticide-sampling` e `rusticide-model`. A parte 3
adiciona o crate `rusticide-simd` (o unico novo `unsafe`), um adaptador `neon`
em `rusticide-kernels` selecionado em tempo de build, e um layout de pesos em
grupos de linhas que o adaptador prepara no proprio lugar durante o load.

**Stack Tecnologica:** Rust 2024 (MSRV 1.98), rayon, intrinsics NEON e dotprod
de `core::arch::aarch64`, bytemuck (`aarch64_simd`), cargo-tarpaulin, Miri,
GitHub Actions (`ubuntu-24.04-arm`).

**Especificacao:** `docs/plan/2026-11-01-rusticide-neon-adapter-design.pt-BR.md`

## Restricoes Globais

- Tudo o que esta em AGENTS.md: TDD estrito (veja o teste falhar primeiro),
  100% de cobertura (`mise run test`), arquivos com menos de 300 linhas
  incluindo testes, somente ASCII, o cabecalho SPDX de tres linhas em todo
  novo arquivo `.rs`, `-D warnings`, nenhum `unwrap`/`expect` fora de
  `#[cfg(test)]`, `cargo fmt` antes de todo commit.
- MSRV 1.98: nenhuma API da std mais nova que 1.98.
- `unsafe` somente em `crates/rusticide-simd`, sob `docs/UNSAFE.md`: um
  comentario `// SAFETY:` em todo bloco, API publica segura, uma justificativa
  no README, uma entrada em `docs/UNSAFE.md`, Miri no CI.
- Nunca adicione `target-cpu=native` nem `+dotprod` a configuracao commitada
  de build ou release; `+dotprod` aparece somente no env do job `arm` do CI.
- Dentro de um mesmo adaptador, um prefill em lote continua bit a bit
  identico a executar seus tokens um por vez (os testes existentes de
  `matmul_batch` e de lote do modelo continuam passando sem mudanca).
- `mise run test:reference` passa com NEAR_TIE 0.25 apos cada parte.
- `mise run bench:gate` passa antes do ponto de merge de cada parte; uma
  baseline so e regravada nas tarefas que dizem isso (uma mudanca revisada),
  em uma maquina ociosa na tomada.
- Commits: assinados com GPG (`git commit -S`), convencionais, assunto de no
  maximo 50 caracteres, linhas do corpo de no maximo 72, sem trailers de
  qualquer tipo.
- Docs bilingues: todo `.md` alterado em `docs/` altera seu irmao `.pt-BR.md`
  no mesmo commit (ASCII, sem acentos).

## Foco da Revisao

1. Um lote de um token (decode) nao deve pagar por novas regioes paralelas:
   todo caminho paralelo da parte 1 mantem um ramo serial para `tokens == 1`,
   e as linhas e2e de decode do `bench:gate` nao devem regredir.
2. A amostragem com aleatoriedade semeada deve continuar produzindo os mesmos
   tokens para a mesma semente e os mesmos logits: a parte 2 e fixada contra o
   algoritmo atual por um teste oraculo sobre logits aleatorios, empates
   incluidos.
3. Logits com NaN, todos `-inf`, ou `top_k` maior que o vocabulario devem se
   comportar como hoje (`NoValidTokens` ou o vocabulario inteiro), nunca
   entrar em panico.
4. Uma matriz quantizada cujas linhas nao sao multiplo do grupo do `neon`, ou
   a matriz de saida/embedding compartilhada (tied), ainda deve carregar,
   executar e decodificar as linhas corretamente (recorre ao `portable`, e
   `copy_row` le a linha original de qualquer layout).
5. `RUSTICIDE_KERNELS` com um valor desconhecido, ou `neon` em um build sem o
   modulo, nao deve entrar em panico: o valor e reportado uma vez no stderr e
   usa-se `portable` ou o padrao compilado.

---

## Parte 1: trabalho paralelo por token

### Tarefa 1: quantizar as ativacoes de um lote em paralelo

**Arquivos:**

- Modificar: `crates/rusticide-kernels/src/matmul.rs` (`Activations::new`)
- Teste: `crates/rusticide-kernels/src/matmul_tests.rs`

**Interfaces:**

- Consome: `Q8Input::quantize(&mut self, &[f32], Q8Kind)`.
- Produz: nenhuma API nova; `matmul_batch` mantem sua assinatura e sua
  garantia de identidade bit a bit.
- [ ] **Passo 1: Escrever o teste que falha.** Adicione
  `batched_quantization_matches_per_token_quantization`: para 1, 2, 7 e 64
  tokens de largura 256 com valores deterministicos, o `Q8Input` de cada
  token construido por `Activations::new` e igual (campo a campo: `scales`,
  `quants`, `sums`) a um `Q8Input::quantize` desse token isolado, para ambos
  os `Q8Kind`s. Ele passa hoje; prove-o quebrando o laco por token (quantize o
  token 0 para todo slot), vendo-o falhar e restaurando.
- [ ] **Passo 2: Implementar.** Em `Activations::new`, quantize os tokens com
  `par_iter` quando `tokens > 1` e serialmente caso contrario; colete na ordem
  dos tokens.
- [ ] **Passo 3: Verificar.** `cargo test -p rusticide-kernels` passa, e
  `cargo clippy -p rusticide-kernels --all-targets -- -D warnings` esta limpo.
- [ ] **Passo 4: Commit** `perf(kernels): quantize batch tokens in parallel`.

### Tarefa 2: executar o trabalho f32 por token do lote no pool

**Arquivos:**

- Modificar: `crates/rusticide-llama/src/model.rs` (`normalize_rows`, as duas
  chamadas de `add_assign`), `crates/rusticide-llama/src/layout.rs`
  (`to_token_major`), `crates/rusticide-llama/src/feed_forward.rs`
  (`silu_assign`, `mul_assign`)
- Criar: `crates/rusticide-llama/src/batch.rs` (os helpers paralelos)
- Teste: modulo de testes de `crates/rusticide-llama/src/batch.rs`, testes de
  `layout.rs`

**Interfaces:**

- Produz, em `batch.rs`:
  - `pub(crate) const PARALLEL_MIN_TOKENS: usize` (o construtor escolhe o
    valor a partir do gate; comece em 2).
  - `pub(crate) fn for_each_row(rows: &mut [f32], width: usize, tokens:
    usize, op: impl Fn(usize, &mut [f32]) + Sync)`: chama `op(token, row)`
    para cada linha de largura `width`, no pool quando
    `tokens >= PARALLEL_MIN_TOKENS`, serialmente caso contrario.
  - `pub(crate) fn zip_rows(rows: &mut [f32], other: &[f32], width:
    usize, tokens: usize, op: impl Fn(&mut [f32], &[f32]) + Sync)`: o mesmo
    sobre pares de linhas.
- `normalize_rows`, os `add_assign` residuais, os `silu_assign`/`mul_assign`
  do feed-forward e `to_token_major` usam esses helpers. `to_token_major`
  paraleliza sobre os tokens de saida (cada linha de saida coleta sua coluna),
  mantendo o caminho de copia para `tokens == 1`.
- [ ] **Passo 1: Escrever os testes que falham.** Em `batch.rs`:
  `for_each_row_visits_every_row_once_with_its_index` (registra os indices em
  um `Mutex<Vec>`; ordenado e igual a `0..tokens` para tokens 1, 2, 9) e
  `zip_rows_pairs_matching_rows`. Em `layout.rs`, mantenha os testes
  existentes e adicione `large_batches_match_the_serial_gather` (64 tokens x
  96 linhas contra uma referencia escrita no teste).
- [ ] **Passo 2: Executa-los** e ve-los falhar na compilacao (sem modulo).
- [ ] **Passo 3: Implementar** `batch.rs` com `rayon::prelude`, depois rotear
  os pontos de chamada acima por ele.
- [ ] **Passo 4: Verificar.** `cargo test -p rusticide-llama` passa, incluindo
  os testes inalterados de identidade bit a bit de lote em
  `model_batch_tests.rs`.
- [ ] **Passo 5: Commit** `perf(llama): run per-token batch work on the pool`.

### Tarefa 3: bias e RoPE por token em paralelo

**Arquivos:**

- Modificar: `crates/rusticide-llama/src/attention.rs` (`Attention::forward`)
- Teste: `crates/rusticide-llama/src/attention_tests.rs`

**Interfaces:**

- Consome: `batch::PARALLEL_MIN_TOKENS`.
- O laco por token se divide em dois: primeiro `add_bias` e `heads.rotate`
  para cada token (no pool quando `tokens >= PARALLEL_MIN_TOKENS`, usando
  `par_chunks_exact_mut` sobre as linhas de query/key/value combinadas com a
  posicao `start + offset`), depois as chamadas `cache.store` serialmente na
  ordem das posicoes.
- [ ] **Passo 1: Escrever o teste que falha.**
  `batched_rotation_matches_one_token_at_a_time`: um lote de 12 tokens por
  `Attention::forward` escreve a mesma saida e as mesmas linhas de cache que
  12 chamadas de um token (bit a bit identicas). Ele passa hoje; prove-o
  rotacionando todo token em `start` em vez de `start + offset`.
- [ ] **Passo 2: Implementar** a divisao.
- [ ] **Passo 3: Verificar** `cargo test -p rusticide-llama`.
- [ ] **Passo 4: Commit** `perf(llama): rotate batch tokens in parallel`.

### Tarefa 4: gate e baseline da parte 1

**Arquivos:**

- Modificar: `crates/rusticide-bench/baselines/apple-m1-pro-8.json`
- Modificar: `docs/plan/2026-11-01-rusticide-neon-adapter-design.md` e
  `.pt-BR.md` (acrescentar uma linha na tabela `## Results` para a parte 1)
- [ ] **Passo 1:** `mise run test:reference` passa.
- [ ] **Passo 2:** `mise run bench:gate` sai com 0 (as linhas de prefill
  melhoram, as de decode ficam dentro da tolerancia). Se uma linha de decode
  regredir, aumente `PARALLEL_MIN_TOKENS` e execute de novo; registre o valor
  escolhido.
- [ ] **Passo 3:** Meca a quantidade de threads: execucoes de `mise run bench`
  com `RAYON_NUM_THREADS=6` e `=8` com `RUSTICIDE_MODELS`; registre os numeros
  e2e de prefill e decode na tabela Results. Nao mude codigo.
- [ ] **Passo 4:** `mise run bench:baseline`, depois `mise run bench:gate`
  passa contra ela.
- [ ] **Passo 5:** `mise run ci:quick` passa.
- [ ] **Passo 6: Commit** `perf(bench): re-record the M1 baseline` com a
  baseline e as linhas de Results.

## Parte 2: amostragem sobre os sobreviventes do top-k

### Tarefa 5: um sampler de slice que trabalha sobre os sobreviventes

**Arquivos:**

- Criar: `crates/rusticide-sampling/src/candidates.rs` (o conjunto de
  sobreviventes e seus filtros)
- Modificar: `crates/rusticide-sampling/src/sampler.rs`,
  `crates/rusticide-sampling/src/filters.rs`
- Teste: `crates/rusticide-sampling/src/candidates_tests.rs`

**Interfaces:**

- Produz:
  - `pub fn Sampler::sample_logits(&mut self, logits: &[f32], params:
    &SamplingParams) -> Result<u32>` e `pub fn
    Sampler::greedy_logits(&self, logits: &[f32]) -> Result<u32>`: a API de
    slice, tratando NaN como `-inf` sem copiar a entrada.
  - `Sampler::sample` e `Sampler::sample_greedy` mantem suas assinaturas e
    delegam para a API de slice depois de `extract_logits`.
- Algoritmo (o construtor o escreve; estas sao as decisoes):
  - Quando `0 < top_k < len`, selecione os k maiores com
    `select_nth_unstable_by` sobre pares (indice, logit), mantenha todo logit
    igual ao k-esimo (empates, como hoje), depois ordene os sobreviventes por
    indice para que toda soma posterior rode em ordem de indice.
  - Temperatura, softmax, top-p e o sorteio multinomial rodam entao somente
    sobre os sobreviventes. Como os logits descartados contribuem com
    `exp(-inf) = 0` para as somas de hoje, somar os sobreviventes em ordem de
    indice da probabilidades bit a bit identicas.
  - Com `top_k == 0` ou `top_k >= len`, os sobreviventes sao todos os logits
    finitos, em ordem de indice.
- [ ] **Passo 1: Escrever os testes que falham** em `candidates_tests.rs`:
  `survivors_reproduce_the_full_vocabulary_sampler`: para 200 sementes, um
  vocabulario de 1000 logits aleatorios com empates, NaN e `-inf` injetados, e
  parametros de `{temperature 0.0, 0.7, 1.3} x {top_k 0, 1, 40, 1000, 5000} x
  {top_p 0.5, 0.95, 1.0}`, `sample_logits` retorna o mesmo token que uma copia
  do `Sampler::sample` de hoje mantida no arquivo de teste como
  `reference_sample`. Mais `all_non_finite_logits_are_no_valid_tokens` e
  `top_k_beyond_the_vocabulary_keeps_everything`.
- [ ] **Passo 2: Executar** e ve-los falhar na compilacao.
- [ ] **Passo 3: Implementar** `candidates.rs`, depois fazer `sample`
  delegar. Apague as funcoes de filtro que nenhum chamador usa mais.
- [ ] **Passo 4: Verificar** `cargo test -p rusticide-sampling` (os testes
  existentes e `non_finite_tests` inalterados).
- [ ] **Passo 5: Commit** `perf(sampling): filter only the top-k survivors`.

### Tarefa 6: gerar sem copiar logits, gate da parte 2

**Arquivos:**

- Modificar: `crates/rusticide-model/src/generate.rs` (`pick`)
- Modificar: `crates/rusticide-bench/baselines/apple-m1-pro-8.json`, as linhas
  de Results do design (ambos os idiomas)

**Interfaces:**

- Consome: `Sampler::sample_logits`, `Sampler::greedy_logits`.
- `pick(sampler, logits: &[f32], params)` os chama diretamente; nenhum
  `Tensor` e construido por token.
- [ ] **Passo 1:** Os testes existentes de generate passam sem mudanca apos a
  troca (`cargo test -p rusticide-model`); prove que um deles cobre `pick`
  trocando os ramos greedy e amostrado e vendo-o falhar.
- [ ] **Passo 2:** `mise run test:reference` passa.
- [ ] **Passo 3:** Meca o decode do Qwen com amostragem (`rusticide run ...
  --temperature 0.8 --seed 1 --max-tokens 256`, tres execucoes, antes e
  depois) e registre na tabela Results.
- [ ] **Passo 4:** `mise run bench:gate`, depois `mise run bench:baseline` e o
  gate de novo; `mise run ci:quick` passa.
- [ ] **Passo 5: Commit** `perf(model): sample logits without a copy`
  (codigo) e `perf(bench): re-record the M1 baseline` (baseline e Results).

## Parte 3: o adaptador `neon`

### Tarefa 7: o crate `rusticide-simd` e seu CI

**Arquivos:**

- Criar: `crates/rusticide-simd/{Cargo.toml,README.md,src/lib.rs,
  src/dot.rs}`
- Modificar: `Cargo.toml` da raiz (`[workspace.dependencies]`:
  `bytemuck = { version = ">=1,<2.0.0" }`, entrada de path para
  `rusticide-simd`), `docs/UNSAFE.md` (entrada), `deny.toml` somente se a
  licenca do bytemuck exigir, `.github/workflows/cicd.yml` (jobs `miri` e
  `arm`, ambos adicionados a `tag.needs`), `AGENTS.md` (a linha do `unsafe`
  nomeia ambos os crates)

**Interfaces:**

- `Cargo.toml` copia o `[lints]` de `rusticide-alloc` (com
  `unsafe_code = "allow"` e `undocumented_unsafe_blocks = "deny"`); depende de
  `bytemuck` com a feature `aarch64_simd`.
- `src/lib.rs`: o modulo `neon` existe somente sob
  `#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]`;
  `pub const AVAILABLE: bool` e verdadeiro exatamente nesse caso.
- Primeiro kernel, o molde que todo kernel posterior segue:
  `pub fn dot_i8x32(lhs: &[i8; 32], rhs: &[i8; 32]) -> i32` em `neon::dot`, um
  wrapper seguro em torno de um
  `#[target_feature(enable = "neon,dotprod")] fn` privado usando `vdotq_s32`,
  com a unica chamada `unsafe` e seu comentario `// SAFETY:` citando o cfg de
  tempo de build; sob `cfg(miri)` o wrapper chama um corpo escalar.
- Job `miri` do CI (em `cicd.yml`, ubuntu-latest, nome do job
  `Miri (rusticide-simd)`): toolchain `nightly-2026-08-03` com `miri` e
  `rust-src`, `cargo miri test -p rusticide-simd --target aarch64-apple-darwin`,
  `MIRIFLAGS=-Zmiri-strict-provenance`.
- Job `arm` do CI: `runs-on: ubuntu-24.04-arm`, os mesmos passos de `gates`,
  `env: RUSTFLAGS: "-D warnings -C target-feature=+dotprod"`, nome do job
  `Gates arm64 (mise run ci:quick)`.
- Novas actions sao fixadas por SHA com
  `uv run --script scripts/update-actions-pins.py --write`; sem edicoes
  manuais.
- [ ] **Passo 1: Escrever o teste que falha** `dot_matches_the_scalar_sum`
  (todos os 256 valores de borda de `i8` em cada posicao, mais 1000 pares
  aleatorios) e `available_matches_the_build_target`.
- [ ] **Passo 2: Executar** `cargo test -p rusticide-simd` e ve-lo falhar.
- [ ] **Passo 3: Implementar** o crate, o README e a entrada em
  `docs/UNSAFE.md` (e seu `.pt-BR.md` se existir).
- [ ] **Passo 4: Verificar:** `cargo test -p rusticide-simd`;
  `cargo +nightly-2026-08-03 miri test -p rusticide-simd --target
  aarch64-apple-darwin` localmente; `mise run security` (bytemuck passa no
  deny); `actionlint` via `mise run ci:quick`; `mise run ci:quick`.
- [ ] **Passo 5: Commit** `feat(simd): add the NEON kernel crate` (crate,
  docs) e `ci: gate Miri and arm64` (workflow).

### Tarefa 8: o adaptador `neon` e sua selecao

**Arquivos:**

- Criar: `crates/rusticide-kernels/src/neon/mod.rs`
- Modificar: `crates/rusticide-kernels/src/adapter.rs`,
  `crates/rusticide-kernels/Cargo.toml` (depender de `rusticide-simd`)
- Teste: `crates/rusticide-kernels/src/adapter_tests.rs` (novo; mova o teste
  existente do adaptador para la)

**Interfaces:**

- `neon::ADAPTER: Adapter` chamado `"neon"`, compilado somente quando
  `rusticide_simd::AVAILABLE` (mesmo cfg), comecando com os kernels do
  `portable` (campos `..portable::ADAPTER`) para que o comportamento nao mude.
- `fn choose(requested: Option<&str>, compiled: &'static Adapter) ->
  (&'static Adapter, Option<String>)`: `None` -> padrao compilado;
  `Some("portable")` -> portable; `Some("neon")` -> o padrao compilado se ele
  for `neon`, senao portable mais um aviso; qualquer outro valor -> padrao
  compilado mais o aviso `"RUSTICIDE_KERNELS=<value>: unknown adapter, using
  <name>"`. `active()` o chama uma vez com
  `std::env::var("RUSTICIDE_KERNELS").ok()` e imprime o aviso no stderr.
- [ ] **Passo 1: Escrever os testes que falham** para cada linha de `choose`
  acima (o texto do aviso exato) e `active_adapter_is_the_compiled_default`
  (neste build M1: `"neon"`; em outros lugares `"portable"`).
- [ ] **Passo 2: Executar** e ve-los falhar.
- [ ] **Passo 3: Implementar.**
- [ ] **Passo 4: Verificar** `cargo test --workspace`; no M1
  `RUSTICIDE_KERNELS=portable cargo test -p rusticide-kernels` passa.
- [ ] **Passo 5: Commit** `feat(kernels): select a neon adapter at build`.

### Tarefa 9: a porta do layout em grupos de linhas

**Arquivos:**

- Modificar: `crates/rusticide-kernels/src/quant_kernel.rs`,
  `crates/rusticide-kernels/src/matmul.rs`,
  `crates/rusticide-kernels/src/adapter.rs`,
  `crates/rusticide-model/src/weights.rs`
- Teste: `quant_kernel_tests.rs`, `matmul_tests.rs`, `weights_tests.rs`

**Interfaces:**

- `QuantKernel` ganha:
  - `pub(crate) group_rows: usize` (1 para todo kernel `portable`);
  - `pub(crate) prepare: Option<fn(data: &mut [u8], row_bytes: usize) ->
    bool>`: reempacota grupos inteiros no proprio lugar, `false` (dados
    intactos) quando `data.len()` nao e multiplo de
    `row_bytes * group_rows`;
  - `pub(crate) read_row: Option<fn(data: &[u8], row_bytes: usize, index:
    usize, out: &mut [u8]) -> bool>`: escreve os bytes de bloco originais da
    linha `index`;
  - `pub(crate) dot_group: Option<fn(group: &[u8], inputs: &[Q8Input],
    out: &mut [f32])>`: um grupo contra todas as entradas,
    `out[row * inputs.len() + token]`.
- `pub fn Adapter::prepare_matrix(&self, type_id: u32, data: &mut [u8],
  row_bytes: usize) -> Option<&'static QuantKernel>`: o kernel deste adaptador
  quando seu `prepare` tem sucesso (ou ele nao tem um), senao o kernel
  `portable` para `type_id`, senao `None`.
- `pub fn QuantKernel::read_row(&self, data: &[u8], row_bytes: usize,
  index: usize, out: &mut [u8]) -> bool`: a copia identidade para
  `group_rows == 1`.
- O escalonador arredonda `rows_per_task` para cima ate um multiplo de
  `group_rows` e, para `group_rows > 1`, chama `dot_group` por grupo.
- `Matrix::load` chama `active().prepare_matrix` sobre sua copia propria;
  `Matrix::copy_row` le a linha por `kernel.read_row` para um buffer de
  `row_bytes` antes de `dequantize_row`.
- [ ] **Passo 1: Escrever os testes que falham** com um kernel somente de
  teste em `test_support.rs` que "intercala" 2 linhas trocando-as no proprio
  lugar (`group_rows: 2`): `prepare_matrix_falls_back_on_an_odd_row_count`,
  `read_row_restores_every_original_row`,
  `grouped_kernels_match_row_kernels` (saida do escalonador identica para 1, 3
  e 64 tokens) e, em `weights_tests.rs`,
  `copy_row_reads_a_prepared_matrix`.
- [ ] **Passo 2: Executar** e ve-los falhar.
- [ ] **Passo 3: Implementar.**
- [ ] **Passo 4: Verificar** `cargo test --workspace` e
  `mise run test:reference` (caminho portable inalterado).
- [ ] **Passo 5: Commit** `feat(kernels): add the row-group layout port`.

### Tarefa 10: `Q4_K` em NEON

**Arquivos:**

- Criar: `crates/rusticide-simd/src/neon/q4_k.rs`,
  `crates/rusticide-kernels/src/neon/q4_k.rs` (layout: `prepare`,
  `read_row`; Rust seguro)
- Modificar: `crates/rusticide-kernels/src/neon/mod.rs` (registrar o kernel)
- Teste: ao lado de cada arquivo

**Interfaces:**

- Layout: grupos de 8 linhas (`group_rows = 8`), com os blocos de cada grupo
  intercalados como o `block_q4_Kx8` do llama.cpp (o construtor confirma a
  ordem de bytes a partir do `repack.cpp` do ggml; qualquer layout bijetivo
  que mantenha 8 x 144 bytes por grupo de blocos e aceitavel).
- `rusticide_simd::neon::q4_k_group(group: &[[u8; 1152]], inputs: &[
  Q8Superblocks], out: &mut [f32])` onde o construtor define `Q8Superblocks`
  como a visao de tamanho fixo de uma entrada `Q8Kind::SuperBlock` de que o
  kernel precisa (o crate de kernels a constroi a partir de `Q8Input`; todo
  fatiamento para arrays de tamanho fixo acontece em codigo seguro antes da
  chamada).
- Tolerancia: a saida f32 coincide com `q4_k_rows` do `portable` dentro de
  `1e-5` relativo (ou `1e-6` absoluto perto de zero); as somas parciais
  inteiras sao exatas.
- [ ] **Passo 1: Escrever os testes que falham:**
  `prepare_then_read_row_is_identity` sobre blocos aleatorios;
  `neon_q4_k_matches_portable` para 1, 2, 7 e 64 tokens sobre pesos
  aleatorios; em `rusticide-simd`, o corpo escalar do Miri e igual ao corpo
  NEON.
- [ ] **Passo 2: Executar** e ve-los falhar.
- [ ] **Passo 3: Implementar** layout, kernel e registro.
- [ ] **Passo 4: Verificar:** `cargo test --workspace`; Miri como na Tarefa 7;
  `mise run test:reference`; `mise run bench:gate` mostra
  `kernel/Q4_K/rows8` e a linha de prefill do TinyLlama melhorando (registre
  os numeros no relatorio da tarefa).
- [ ] **Passo 5: Commit** `feat(simd): run Q4_K on NEON`.

### Tarefa 11: `Q6_K` em NEON

Mesma forma da Tarefa 10, para `Q6_K` (analogo de `block_q6_Kx8`, 8 x 210
bytes por grupo). Testes `prepare_then_read_row_is_identity`,
`neon_q6_k_matches_portable`, a igualdade do corpo escalar do Miri. Commit
`feat(simd): run Q6_K on NEON`.

### Tarefa 12: `Q5_0` e `Q8_0` em NEON

Mesma forma para os dois formatos de bloco que o Qwen2.5 usa (blocos de 32
elementos, 22 e 34 bytes). O construtor pode usar um grupo mais simples de 4
linhas se ele atingir o Objetivo 1 para o Qwen2.5; registre a escolha. Commit
`feat(simd): run Q5_0 and Q8_0 on NEON`.

### Tarefa 13: gate da parte 3, baseline, resultados e checks obrigatorios

**Arquivos:**

- Modificar: a baseline do M1; a tabela Results e o status do design (ambos
  os idiomas); o ponteiro "Current plan" do `AGENTS.md`.
- [ ] **Passo 1:** `mise run test:reference` passa com `neon` ativo e com
  `RUSTICIDE_KERNELS=portable`.
- [ ] **Passo 2:** `RUSTICIDE_KERNELS=portable mise run bench:gate` passa
  contra a baseline da parte 2 (o fallback nao regrediu).
- [ ] **Passo 3:** `mise run bench:gate` com `neon`; registre o `pp128` do
  llama.cpp (`llama-bench -ngl 0 -t 6`) ao lado do nosso para ambos os modelos
  e verifique o Objetivo 1 (pelo menos 65%). Se nao for atingido, pare e
  reporte com o profile em vez de regravar.
- [ ] **Passo 4:** `mise run bench:baseline`, gate de novo,
  `mise run ci:quick`.
- [ ] **Passo 5: Commit** `perf(bench): re-record the M1 baseline` e
  `docs: record the NEON adapter results`.
- [ ] **Passo 6:** Apos o merge (controlador, nao um implementador): adicione
  `Miri (rusticide-simd)` e `Gates arm64 (mise run ci:quick)` aos checks
  obrigatorios do ruleset de `main` com `gh api`.
