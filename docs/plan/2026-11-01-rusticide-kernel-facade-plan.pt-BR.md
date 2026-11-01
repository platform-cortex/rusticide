<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Plano de implementacao do facade de kernels e da guarda de desempenho

> **Para agentes:** SUB-SKILL OBRIGATORIA: use
> superpowers:subagent-driven-development (recomendado) ou
> superpowers:executing-plans para implementar este plano tarefa a tarefa. Os
> passos usam a sintaxe de checkbox (`- [ ]`) para acompanhamento.

**Objetivo:** Fazer os documentos corresponderem a 0.6.0, adicionar uma guarda
local de desempenho com uma baseline commitada, e depois mover todo kernel para
tras de um unico facade em `rusticide-kernels` com um unico adaptador
`portable`, sem perder velocidade.

**Arquitetura:** Um novo crate `rusticide-bench` mede kernels (sem modelo) e
tokens/s de ponta a ponta (com `RUSTICIDE_MODELS`) e compara com
`crates/rusticide-bench/baselines/<machine>.json`. A baseline e capturada do
codigo da 0.6.0 antes de qualquer refatoracao. Depois `rusticide-kernels`
ganha um `Adapter` (uma struct de ponteiros de funcao simples, escolhida uma
vez por processo, a exemplo do `DispatchStub` do PyTorch), absorve
`Q8Input`/`int_dot` de `rusticide-quantize` e `matmul_batch` de
`rusticide-model`, e `BlockFormat` volta a ser apenas dados.

**Stack:** Rust 2024, rayon, serde/serde_json, clap, anyhow (somente no
binario), tarefas mise, GitHub Actions.

**Spec:** `docs/plan/2026-11-01-rusticide-kernel-facade-design.md`
Versao em ingles: [2026-11-01-rusticide-kernel-facade-plan.md](2026-11-01-rusticide-kernel-facade-plan.md).

## Restricoes globais

- O AGENTS.md vale: TDD estrito (red, depois green), gate de 100% de
  cobertura, nenhum `unwrap`/`expect` fora de `#[cfg(test)]`, arquivos com
  menos de 300 linhas, somente ASCII, o cabecalho SPDX de tres linhas em todo
  `.rs` novo, `-D warnings`, MSRV 1.96, nenhum `unsafe` em todo este plano.
- Commits: `git commit -S`, convencionais, primeira linha com no maximo 50
  caracteres, sem trailers de atribuicao. Rode `cargo fmt` antes de cada
  commit.
- Todo documento novo ou alterado tem um irmao `.pt-BR.md` mantido em sincronia;
  links dentro de um `.pt-BR.md` apontam para os irmaos `.pt-BR.md` quando eles
  existem.
- Anti-objetivo: nenhum numero de decode ou prefill no M1 Pro abaixo da
  baseline commitada alem da sua tolerancia. A partir da Tarefa 5, toda tarefa
  termina com `mise run bench:gate` passando.
- Anti-objetivo: nenhuma chamada indireta por bloco. As portas sao chamadas por
  sequencia de linhas ou por lote.
- Anti-objetivo: `mise run test:reference` passa com NEAR_TIE 0.25.
- O despachante (`rusticide-dispatch`) nao e tocado.

## Foco da revisao

1. Aplicar a guarda contra um arquivo de baseline de outra maquina, ou contra
   nenhum: um erro claro e saida diferente de zero, nunca uma aprovacao
   (Tarefa 2).
2. Rodar a guarda sem `RUSTICIDE_MODELS` quando a baseline tem metricas de
   ponta a ponta: cada metrica ausente e reportada e reprova a guarda
   (Tarefa 2).
3. Uma medicao degenerada (nenhuma execucao, tempo decorrido zero, trabalho
   zero): um erro, nunca uma vazao infinita ou NaN (Tarefa 2).
4. Um tensor GGUF cujo formato nao tem kernel no adaptador: `Matrix::load`
   retorna `ModelError::MissingKernel`, nao um panic (Tarefa 6).
5. Um produto de matrizes com zero tokens ou um job com zero linhas depois da
   mudanca: nenhum trabalho e nenhum panic, como hoje (Tarefa 7).

---

### Tarefa 1: Fazer os documentos corresponderem a 0.6.0

**Arquivos:**

- Modify: `docs/plan/2026-10-22-rusticide-run-design.md` (emenda depois de
  "Non-goals" e em "Memory" e "Quantization formats")
- Modify: `docs/plan/2026-08-21-rusticide-framework-design.md` (decisao sob as
  secoes do despachante/`rusticide-simd`)
- Create: `docs/plan/2026-10-31-rusticide-inference-performance.pt-BR.md`
- Modify: `docs/plan/2026-11-01-rusticide-kernel-facade-design.pt-BR.md`
  (linkar o novo `.pt-BR.md` das notas)
- Modify: `AGENTS.md` ("Current plan" aponta para este plano e seu design)

**Passos:**

- [ ] **Passo 1:** No design do run, adicione um bloco no estilo de emenda ja
  existente, `> **Amended 2026-11-01.**`, sob "Non-goals", "Memory" e
  "Quantization formats": kernels de matmul quantizado e pesos quantizados em
  memoria foram entregues na 0.6.0 (cite as notas); o contrato de formato e
  nome, type id, tamanho do bloco, bytes do bloco e dequantizador, com os
  kernels registrados em `rusticide-kernels` (cite o design do facade). Nao
  apague o texto original.
- [ ] **Passo 2:** No design do framework, registre a selecao em dois niveis
  (despachante por operador; adaptador no estilo `DispatchStub` por processo
  em `rusticide-kernels`) e que os adaptadores nativos se apoiam em
  `rusticide-simd`.
- [ ] **Passo 3:** Traduza as notas da 0.6.0 para
  `2026-10-31-rusticide-inference-performance.pt-BR.md` (ASCII, sem acentos,
  mesma estrutura) e aponte o link `.pt-BR.md` do design do facade para ele.
- [ ] **Passo 4:** Rode `rumdl check docs AGENTS.md` e
  `LC_ALL=C grep -nP '[^\x00-\x7F]' docs AGENTS.md -r`. Esperado: nenhum
  problema, nenhuma saida.
- [ ] **Passo 5:** Commit: `docs: record 0.6.0 decisions in the specs`.

### Tarefa 2: Resultados de benchmark, baselines e comparacao

**Arquivos:**

- Create: `crates/rusticide-bench/Cargo.toml` (`publish = false`; deps
  `serde`, `serde_json`, `thiserror`; lints do workspace)
- Create: `crates/rusticide-bench/src/lib.rs`, `src/error.rs`,
  `src/stats.rs`, `src/results.rs`, `src/baseline.rs`, mais irmaos
  `*_tests.rs` onde um arquivo passaria de 300 linhas
- O crate e uma folha: sem entrada em `[workspace.dependencies]`. Sua versao
  segue `version.workspace`, entao `release:prepare` o incrementa como os
  demais.

**Interfaces:**

- Produz (tudo `pub` em `rusticide_bench`):
  - `BenchError` (thiserror): `NoRuns`, `ZeroElapsed`, `ZeroWork`,
    `WrongMachine { expected: String, got: String }`, `Json(serde_json::Error)`.
  - `pub struct Sample { pub median: f64, pub spread: f64 }` (serde);
    `spread` e `(max - min) / median`.
  - `pub fn summarize(throughputs: &[f64]) -> Result<Sample, BenchError>`.
  - `pub fn throughput(work: f64, elapsed: Duration) -> Result<f64, BenchError>`.
  - `pub struct Results { pub machine: String, pub metrics: BTreeMap<String, Sample> }`
    (serde), `Results::to_markdown(&self) -> String` (uma linha de tabela por
    metrica: nome, mediana, dispersao em percentual).
  - `pub const TOLERANCE_FLOOR: f64 = 0.05;`
  - `pub struct Expected { pub median: f64, pub tolerance: f64 }` e
    `pub struct Baseline { pub machine: String, pub metrics: BTreeMap<String, Expected> }`
    (serde); `Baseline::from_results(&Results) -> Baseline` com
    `tolerance = spread.max(TOLERANCE_FLOOR)`.
  - `pub enum Status { Within, Regressed, Missing, New }`,
    `pub struct Row { pub metric: String, pub status: Status, pub baseline: Option<f64>, pub measured: Option<f64> }`,
    `pub fn compare(baseline: &Baseline, results: &Results) -> Result<Vec<Row>, BenchError>`
    (`WrongMachine` quando as maquinas diferem), `pub fn passed(rows: &[Row]) -> bool`
    (false em qualquer `Regressed` ou `Missing`), `pub fn rows_to_markdown(rows: &[Row]) -> String`.

**Passos:**

- [ ] **Passo 1: Escreva os testes que falham** (nos arquivos de teste dos
  modulos):
  - `summarize_takes_the_median_and_relative_spread`:
    `summarize(&[9.0, 10.0, 12.0])` e `Sample { median: 10.0, spread: 0.3 }`.
  - `summarize_of_no_runs_is_an_error`: `NoRuns`.
  - `throughput_rejects_zero_elapsed_and_zero_work`: `ZeroElapsed`, `ZeroWork`.
  - `baseline_tolerance_is_the_spread_with_a_floor`: dispersoes 0.02 e 0.12
    dao tolerancias 0.05 e 0.12.
  - `compare_flags_a_drop_beyond_tolerance`: baseline 100.0 / 0.05;
    medido 96.0 e `Within`, 94.0 e `Regressed`.
  - `compare_reports_missing_and_new_metrics`: baseline `{a, b}`, resultados
    `{a, c}` dao `b` `Missing`, `c` `New`; `passed` e false.
  - `compare_refuses_another_machine`: `WrongMachine`.
  - `results_round_trip_through_json` e `baseline_round_trip_through_json`.
- [ ] **Passo 2:** `cargo test -p rusticide-bench`. Esperado: FALHA ao compilar
  (itens ausentes).
- [ ] **Passo 3:** Implemente as interfaces acima. `New` nunca reprova a
  guarda.
- [ ] **Passo 4:** `cargo test -p rusticide-bench` e
  `cargo clippy -p rusticide-bench --all-targets -- -D warnings`. Esperado:
  PASSA, sem warnings.
- [ ] **Passo 5:** Commit: `feat(bench): compare results with a baseline`.

### Tarefa 3: Medicoes de kernel e de ponta a ponta

**Arquivos:**

- Modify: `crates/rusticide-quantize/src/format.rs` (+ testes): adicione
  `BlockFormat::synthetic_blocks(&self, blocks: usize, seed: u32) -> Vec<u8>`,
  bytes pseudoaleatorios de faixa completa com todo campo de escala f16
  definido como um valor finito (offsets: 0 para Q4_0/Q5_0/Q8_0, 0 e 2 para
  Q4_K, 208 para Q6_K). Substitua `random_blocks` em `format_tests.rs` por ele.
- Create: `crates/rusticide-bench/src/kernels.rs`, `src/e2e.rs` (novas deps:
  `rusticide-quantize`, `rusticide-model`)
- Modify: `crates/rusticide-llama/src/lib.rs`: adicione
  `pub fn registry() -> Registry` (movida de
  `crates/rusticide/src/cli/registry.rs`, que agora a chama; seu teste vai
  junto)

**Interfaces:**

- Consome: `summarize`, `throughput`, `Sample` da Tarefa 2.
- Produz:
  - `pub struct Plan { pub runs: usize, pub budget: Duration }`; uma execucao
    de aquecimento e descartada antes de `runs`.
  - `pub fn kernel_metrics(plan: &Plan) -> Result<BTreeMap<String, Sample>, BenchError>`
    com chaves `kernel/<format>/row` (uma linha de 4096 colunas contra uma
    entrada, elementos/s, uma thread), `kernel/<format>/rows8` (as mesmas
    linhas contra 8 entradas, produtos de elementos/s) para `Q4_0`, `Q5_0`,
    `Q8_0`, `Q4_K`, `Q6_K`, e `kernel/q8/block`, `kernel/q8/superblock`
    (quantizacao de ativacao, elementos/s). Matriz: 256 linhas x 4096 colunas
    por formato. Cada execucao repete seu trabalho ate `budget` esgotar.
  - `pub const DECODE_PROMPT: &str = "The history of the Roman Empire";`
  - `pub const DECODE_TOKENS: usize = 64;`
  - `pub const PREFILL_PROMPT: &str` -- um paragrafo fixo em ingles de cerca
    de 120 palavras (o construtor o escreve; muda-lo depois invalida as
    baselines).
  - `pub fn e2e_metrics(name: &str, model: &mut dyn Model, decode_prompt: &[u32], prefill_prompt: &[u32], plan: &Plan) -> Result<BTreeMap<String, Sample>, BenchError>`
    com chaves `e2e/<name>/decode` (argmax guloso por `DECODE_TOKENS` depois
    do prompt, tokens/s somente dos passos de decode) e
    `e2e/<name>/prefill` (um `forward_batch` do prompt de prefill,
    tokens/s). `model.reset()` antes de cada execucao. Erros do modelo viram
    um novo `BenchError::Model(ModelError)`.

**Passos:**

- [ ] **Passo 1: Escreva os testes que falham:**
  - `synthetic_blocks_dequantize_to_finite_values` para todo formato.
  - `kernel_metrics_cover_every_format_and_both_activations`: com
    `Plan { runs: 2, budget: Duration::from_millis(1) }`, as 12 chaves
    esperadas existem e toda mediana e positiva e finita.
  - `e2e_metrics_time_decode_and_prefill_on_a_model`: um `Model` de teste
    (vocab 4, logits favorecendo o token 1) gera as duas chaves, medianas
    positivas, e foi resetado uma vez por execucao, aquecimento incluido.
  - `e2e_metrics_propagate_model_errors`: um modelo cujo `forward` falha
    gera `BenchError::Model`.
  - `registry_lists_llama_and_qwen2` em `rusticide-llama`.
- [ ] **Passo 2:** Rode os novos testes. Esperado: FALHA ao compilar.
- [ ] **Passo 3:** Implemente. As metricas de kernel chamam a API de hoje
  (`BlockFormat::dot_row_q8`, `dot_rows_q8`, `Q8Input::quantize`) em uma
  unica thread, sem rayon.
- [ ] **Passo 4:** `cargo test -p rusticide-bench -p rusticide-quantize -p rusticide-llama -p rusticide`.
  Esperado: PASSA.
- [ ] **Passo 5:** Commit: `feat(bench): measure kernels and tokens/s`.

### Tarefa 4: Binario de bench, tarefas, resumo no CI e a primeira baseline

**Arquivos:**

- Create: `crates/rusticide-bench/src/main.rs` (clap + anyhow; o unico arquivo
  sem teste, excluido pela regra existente `crates/*/src/main.rs` em
  `tarpaulin.toml`; contem a cola de carga do GGUF e do tokenizador e nada
  mais)
- Modify: `mise.toml` (tarefas abaixo, usando o template de tarefa)
- Modify: `.github/workflows/cicd.yml` (job `bench`)
- Create: `crates/rusticide-bench/baselines/apple-m1-pro-8.json`
- Modify: `AGENTS.md` (comandos, a regra da guarda)

**Interfaces:**

- Consome: Tarefas 2 e 3.
- Produz: binario `rusticide-bench` com subcomandos
  - `run --machine <id> [--models <dir>] [--runs N] [--budget-ms N] --out <file>`
    (kernels sempre; ponta a ponta para cada arquivo de modelo de referencia
    encontrado em `<dir>`, nomes `tinyllama` e `qwen2.5`; padroes 5
    execucoes, 200 ms)
  - `report --results <file>` (tabela markdown na saida padrao)
  - `gate --results <file> --baseline <file>` (tabela na saida padrao, saida 1
    em caso de falha, saida 2 com uma mensagem que nomeia o caminho esperado
    quando o arquivo de baseline nao existe)
  - `baseline --results <file> --out <file>`
- tarefas mise (cada uma com alias, description, `usage` com `--verbose`):
  - `bench` (alias `bn`): build release, `run` e depois `report`.
  - `bench:gate` (alias `bng`): `run` e depois `gate` contra
    `crates/rusticide-bench/baselines/<machine>.json`.
  - `bench:baseline` (alias `bnb`): `run` e depois `baseline`, escrevendo esse
    arquivo.
  - Id da maquina: marca da CPU em minusculas com nao alfanumericos como `-`,
    mais `-<logical cpus>` (`sysctl -n machdep.cpu.brand_string` e
    `sysctl -n hw.logicalcpu` no macOS, `/proc/cpuinfo` e `nproc` no Linux);
    o M1 Pro resulta em `apple-m1-pro-8`. `RUSTICIDE_MODELS` e passado como
    `--models` quando definido.
- Job de CI `bench` (id em kebab-case, `needs: [version]`, mesma action de
  setup do `gates`): roda `mise run bench` sem modelos e anexa o relatorio a
  `$GITHUB_STEP_SUMMARY`. Falha somente se a ferramenta quebrar.

**Passos:**

- [ ] **Passo 1:** Implemente `main.rs` e as tarefas; `mise run bench`
  localmente. Esperado: uma tabela com 12 linhas de kernel e, com
  `RUSTICIDE_MODELS` definido, 4 linhas de ponta a ponta.
- [ ] **Passo 2:** `mise run bench:gate` sem arquivo de baseline. Esperado:
  saida 2 e o caminho esperado na mensagem.
- [ ] **Passo 3:** No M1 Pro, na tomada, outros apps pesados fechados:
  `RUSTICIDE_MODELS=~/models mise run bench:baseline`, depois
  `mise run bench:gate` duas vezes. Esperado: ambas passam. Registre as
  medianas de ponta a ponta no corpo do commit e confira com as notas da 0.6.0
  (decode cerca de 117 e 82 tokens/s); uma diferenca acima de 10% interrompe a
  tarefa para investigacao.
- [ ] **Passo 4:** AGENTS.md: adicione os tres comandos em "Commands" e a regra
  "run `mise run bench:gate` before merging any change to kernels,
  scheduling or the forward pass; rewriting a baseline is a reviewed
  change".
- [ ] **Passo 5:** `mise run ci:quick`. Esperado: PASSA com 100%.
- [ ] **Passo 6:** Commit: `feat(bench): gate tokens/s on a baseline`.

### Tarefa 5: O adaptador e as portas f32

**Arquivos:**

- Create: `crates/rusticide-kernels/src/adapter.rs` (tipos e `active`)
- Create: `crates/rusticide-kernels/src/portable/` (`mod.rs`, e os atuais
  `dot.rs`, `elementwise.rs`, `norm.rs`, `rope.rs` movidos para ca sem
  alteracao, com seus testes)
- Modify: `crates/rusticide-kernels/src/lib.rs`

**Interfaces:**

- Produz:
  - `pub(crate) struct F32Kernels` (privado ao crate: adaptadores sao modulos
    deste crate, entao os nativos depois nao precisam de construtor publico)
    com um campo de ponteiro `fn` por funcao livre atual: `dot`,
    `add_assign`, `mul_assign`, `silu`, `rms_norm`, `softmax`,
    `rope_adjacent`, `rope_half_split` (mesmas assinaturas de hoje).
  - `pub struct Adapter { name: &'static str, f32: F32Kernels }` (campos
    privados; `pub const fn name(&self) -> &'static str`).
  - `pub fn active() -> &'static Adapter` -- escolhido uma vez por um
    `std::sync::OnceLock`; retorna `&portable::ADAPTER` (`name` e
    `"portable"`).
  - As funcoes livres existentes mantem nomes e assinaturas e chamam por
    `active()`, entao nenhum chamador muda.

**Passos:**

- [ ] **Passo 1: Escreva os testes que falham:** `active_adapter_is_portable`
  (`active().name() == "portable"`, e duas chamadas retornam o mesmo
  endereco); `free_functions_route_through_the_adapter` (um teste no proprio
  modulo monta um `Adapter` cujo `dot` retorna uma constante e confere o
  auxiliar de roteamento usado pelas funcoes livres com ele).
- [ ] **Passo 2:** `cargo test -p rusticide-kernels`. Esperado: FALHA.
- [ ] **Passo 3:** Implemente; as funcoes livres obtem o adaptador por um
  unico `fn adapter() -> &'static Adapter` privado, para que o roteamento seja
  testavel.
- [ ] **Passo 4:** `cargo test --workspace`, `mise run bench:gate`.
  Esperado: PASSA.
- [ ] **Passo 5:** Commit: `refactor(kernels): route f32 kernels via adapter`.

### Tarefa 6: Os kernels quantizados se movem para o facade

**Arquivos:**

- Move: `crates/rusticide-quantize/src/q8_input.rs`, `int_dot.rs`,
  `int_dot/rows.rs` para `crates/rusticide-kernels/src/` (`q8_input.rs`
  compartilhado; `int_dot*` sob `portable/`), com seus testes e os testes de
  kernel de `format_tests.rs`
- Modify: `crates/rusticide-quantize/src/format.rs`, `lib.rs`
- Modify: `crates/rusticide-kernels/Cargo.toml` (deps `half`,
  `rusticide-quantize`), `src/adapter.rs`, `src/lib.rs`
- Modify: `crates/rusticide-model/src/weights.rs`, `batch.rs`,
  `error.rs`; `crates/rusticide-bench/src/kernels.rs`

**Interfaces:**

- Produz:
  - `rusticide_kernels::{Q8_BLOCK, Q8Input, Q8Kind}` (movidos; mesma API).
    `Q8Input::quantize` chama o slot `quantize` de `active()`; o tipo fica em
    `q8_input.rs`, seu corpo portable vai para `portable/q8.rs`.
  - `pub struct QuantKernel { type_id, activation: Q8Kind, dot, dot_rows }`
    com `pub const fn type_id(&self) -> u32`,
    `pub const fn activation(&self) -> Q8Kind`,
    `pub fn dot_row(&self, row: &[u8], input: &Q8Input) -> f32`,
    `pub fn dot_rows(&self, row: &[u8], inputs: &[Q8Input], out: &mut [f32])`
    (corpos dos atuais `dot_row_q8`/`dot_rows_q8`).
  - `Adapter` ganha `quantize: fn(&mut Q8Input, &[f32], Q8Kind)` e
    `quantized: &'static [QuantKernel]`;
    `pub fn quant_kernel(&self, type_id: u32) -> Option<&'static QuantKernel>`.
  - `BlockFormat` perde `dot_q8`, `dot_rows`, `activation` e seus metodos;
    `rusticide-quantize` deixa de exportar `Q8*`.
  - `ModelError::MissingKernel { format: &'static str }`, mensagem
    `no kernel for format {format}`.

**Passos:**

- [ ] **Passo 1: Escreva os testes que falham:**
  `portable_has_a_kernel_for_every_block_format` (toda entrada de
  `rusticide_quantize::formats()` tem um `quant_kernel` com a ativacao que a
  tabela antiga tinha: `Block` para Q4_0/Q5_0/Q8_0, `SuperBlock` para
  Q4_K/Q6_K); os testes de kernel movidos contra a referencia dequantizada
  usando `synthetic_blocks`; `a_format_without_a_kernel_fails_to_load` em
  `rusticide-model` (o auxiliar privado de busca, recebendo `None`, retorna
  `MissingKernel { format: "Q4_K" }`).
- [ ] **Passo 2:** Rode-os. Esperado: FALHA.
- [ ] **Passo 3:** Mova o codigo; `Matrix` guarda `&'static QuantKernel` ao
  lado dos bytes, resolvido uma vez em `Matrix::load`.
- [ ] **Passo 4:** `cargo test --workspace`, `mise run test:reference`,
  `mise run bench:gate`. Esperado: PASSA.
- [ ] **Passo 5:** Commit: `refactor(kernels): own the quantized kernels`.

### Tarefa 7: O agendador de matrizes se move para o facade

**Arquivos:**

- Move: `crates/rusticide-model/src/batch.rs` e `batch_tests.rs` para
  `crates/rusticide-kernels/src/matmul.rs` e `matmul_tests.rs`
- Modify: `crates/rusticide-kernels/Cargo.toml` (dep `rayon`),
  `crates/rusticide-model/src/{weights.rs,lib.rs}`,
  `crates/rusticide-model/Cargo.toml` (remover `rayon`),
  `crates/rusticide-llama/src/{attention.rs,feed_forward.rs,layout.rs}`

**Interfaces:**

- Produz:
  - `pub enum Weights<'a> { Dense { data: &'a [f32], cols: usize }, Blocks { kernel: &'static QuantKernel, data: &'a [u8], row_bytes: usize } }`
    com `pub fn rows(&self) -> usize`.
  - `pub fn matmul_batch(inputs: &[f32], tokens: usize, jobs: &mut [(Weights<'_>, &mut [f32])])`
    e `pub fn matvec_batch(input: &[f32], jobs: &mut [(Weights<'_>, &mut [f32])])`,
    mesma semantica e mesmo layout de saida (`out[row * tokens + token]`) de
    hoje; `TASKS_PER_THREAD` continua 16.
  - `Matrix::weights(&self) -> Weights<'_>`; `rusticide-model` deixa de
    exportar `matmul_batch`/`matvec_batch`.

**Passos:**

- [ ] **Passo 1: Escreva os testes que falham:** os testes de batch movidos,
  reescritos contra `Weights`, mais `a_job_with_no_rows_does_no_work` e o caso
  existente de zero tokens.
- [ ] **Passo 2:** Rode-os. Esperado: FALHA.
- [ ] **Passo 3:** Mova; llama monta os jobs com `matrix.weights()`.
- [ ] **Passo 4:** `cargo test --workspace`, `mise run test:reference`,
  `mise run bench:gate`. Esperado: PASSA.
- [ ] **Passo 5:** Commit: `refactor(kernels): schedule matrix products`.

### Tarefa 8: Fechar o milestone

**Arquivos:**

- Modify: os dois arquivos de design (status concluido, resultados medidos),
  os dois arquivos de notas da 0.6.0 (locais dos kernels agora em
  `rusticide-kernels`), `crates/rusticide-kernels/Cargo.toml` e `README.md`
  (descricao: o facade de kernels), a descricao em
  `crates/rusticide-quantize/Cargo.toml` se ela mencionar kernels,
  `AGENTS.md` (current plan)

**Passos:**

- [ ] **Passo 1:** `mise run ci:quick`, `mise run test:reference`,
  `mise run bench:gate`, `mise run security`. Esperado: tudo PASSA, cobertura
  100%.
- [ ] **Passo 2:** Atualize os documentos com a tabela final da guarda.
- [ ] **Passo 3:** `rumdl check` e o grep de ASCII da Tarefa 1. Esperado:
  limpo.
- [ ] **Passo 4:** Commit: `docs: close the kernel facade milestone`.
