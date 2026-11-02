<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Plano de Implementacao dos adaptadores de kernel selecionados em tempo de execucao

> **Para agentes de execucao:** SUB-SKILL OBRIGATORIA: use
> superpowers:subagent-driven-development (recomendado) ou
> superpowers:executing-plans para implementar este plano tarefa a tarefa. Os
> passos usam a sintaxe de checkbox (`- [ ]`) para acompanhamento.

**Objetivo:** um binario de release por alvo que escolhe, em tempo de
execucao, o adaptador de kernel mais rapido que sua CPU consegue executar:
`neon` em aarch64 com `dotprod`, `avx2` em x86_64 com AVX2, senao `portable`.

**Arquitetura:** `rusticide-simd` ganha tokens de prova de tamanho zero
(`Dotprod`, `Avx2`) que somente a deteccao constroi; todo ponto de entrada de
kernel recebe um, e seu unico `unsafe` o cita. `rusticide-kernels` detecta uma
vez, monta a lista ordenada de adaptadores executaveis e resolve
`RUSTICIDE_KERNELS` contra ela. O adaptador AVX2 espelha a estrutura do NEON:
um layout em grupos de linhas preparado no load e um kernel de grupo por
formato.

**Stack Tecnologica:** Rust 2024, `std::arch` (aarch64 NEON, x86_64
AVX2/AVX-512), `is_*_feature_detected!`, Miri, cargo-tarpaulin,
`rusticide-bench`, `llama-bench` do llama.cpp como referencia.

**Especificacao:** `docs/plan/2026-11-02-rusticide-runtime-adapters-design.pt-BR.md`
(leia-a; sua secao Findings traz fatos dos quais toda tarefa depende).

## Restricoes Globais

- MSRV 1.98, pin do toolchain inalterado; nenhuma API mais nova que 1.98.
- `unsafe` somente em `rusticide-simd`: um bloco por ponto de entrada de
  kernel, um comentario `// SAFETY:` que nomeia o token; API publica segura.
- Os kernels AVX2 (e AVX-512) sao bit a bit identicos ao portable para toda
  entrada que a API segura aceita (igualdade de `to_bits` nos testes); sem
  FMA onde o portable multiplica e soma separadamente.
- Nunca `target-cpu=native`, nunca um build por maquina na configuracao
  commitada.
- Custo da deteccao: no maximo um load em cache por chamada de kernel,
  nenhum por bloco.
- Nenhum `unwrap`/`expect` em codigo de producao; 100% de cobertura; Miri
  passa.
- Somente ASCII, arquivos com menos de 300 linhas, cabecalhos SPDX, docs
  bilingues, commits convencionais assinados (`git commit -S`), sem
  trailers de atribuicao.
- Pisos bloqueantes no iMac Pro: cada kernel AVX2 com pelo menos 2x o seu
  kernel portable (rows8, uma thread); prefill com pelo menos 65% do backend
  de CPU do llama.cpp nos dois modelos de referencia. Duas rodadas de
  otimizacao apos uma falha, depois pare e reporte ao dono.
- O iMac Pro e `ssh elio@192.168.18.146` (sudo sem senha; as ferramentas
  precisam de um login shell: `zsh -lc '...'`). Seu clone fica em
  `~/Development/github.com/platform-cortex/rusticide`; a Tarefa 4 adiciona
  nele um worktree em `~/rusticide-wt` (qualquer branch; so seus arquivos
  importam). Sincronize o worktree do M1 com ele usando `rsync -a --delete
  --exclude target --exclude .git --exclude mise.local.toml ./
  elio@192.168.18.146:~/rusticide-wt/` e compile la. Nunca faca commit nem
  push a partir do iMac Pro; as medicoes ali sao da arvore sincronizada.
- `bench:gate` no M1 apos toda tarefa que toque kernels ou selecao: sem
  regressao.

## Foco da Revisao

1. `RUSTICIDE_KERNELS=avx2` (ou `neon`) em uma CPU sem a feature: um aviso de
   uma linha e o melhor adaptador executavel; o kernel SIMD nunca e
   alcancado. Testado na Tarefa 3 pela lista de adaptadores injetada.
2. Um `Q8` de ativacao contendo `-128` construido pela API publica: as views o
   recusam e o resultado ainda e igual ao do portable. Tarefa 1.
3. Uma matriz cujo numero de linhas nao e multiplo do grupo (ou com linhas de
   zero bytes) sob o layout AVX2: o resto permanece em ordem de linha e o
   kernel portable o le, como no NEON. Tarefa 5.
4. Um wrapper de kernel alcancado antes de a selecao guardar seu token:
   devolve o resultado do portable, nunca uma chamada `unsafe`. Tarefa 3
   (NEON) e Tarefa 4 (AVX2), pelos testes dentro do modulo.
5. Ativacoes e pesos em enderecos impares: somente loads nao alinhados, as
   verificacoes de endereco impar que o `test_support` do NEON ja tem.
   Tarefas 5-7.

---

## Parte 1: selecao em tempo de execucao

### Tarefa 1: views de grupo neutras de arquitetura que recusam `-128`

**Arquivos:**

- Criar: `crates/rusticide-simd/src/group.rs` (movido de `neon/group.rs`),
  testes ao lado.
- Modificar: `crates/rusticide-simd/src/lib.rs`, todo usuario de
  `neon::group` em `neon/*`, `crates/rusticide-kernels/src/neon/group_dot.rs`.

**Interfaces:**

- Produz `rusticide_simd::group::{ROWS, TOKENS, GroupKernel, Q8Blocks,
  Q8Superblocks}`, compilado em todo alvo.
- `Q8Blocks::new(scales, quants) -> Option<Self>` e
  `Q8Superblocks::new(scales, quants, sums) -> Option<Self>`: `None` quando
  qualquer quant coberto e `-128`; do contrario como hoje (a menor fatia
  limita a view).
- `group_dot::dot_group` calcula um token cuja view e `None` com o `dot` do
  portable do formato para cada linha do grupo (mesmos valores).
- [ ] **Passo 1: Escrever os testes que falham:**
  `blocks_refuse_minus_128`, `superblocks_refuse_minus_128`,
  `views_accept_minus_127` (simd); em kernels
  `a_minus_128_activation_matches_portable` (um `Q8Input` montado a mao com
  um `-128`, todo formato NEON, `to_bits` igual ao do portable).
- [ ] **Passo 2: Executar** `cargo test -p rusticide-simd -p
  rusticide-kernels`; veja-os falhar (erros de compilacao contam).
- [ ] **Passo 3: Mover e implementar.** Mantenha a documentacao dos modulos.
- [ ] **Passo 4: Verificar** `cargo test --workspace`, `mise run lint`, Miri
  (a task `mise run` que o job `miri` do CI executa, ou `cargo
  +nightly-2026-08-03 miri test -p rusticide-simd`).
- [ ] **Passo 5: Commit** `refactor(simd): share the group views`.

### Tarefa 2: o token `Dotprod` e um modulo `neon` por arquitetura

**Arquivos:**

- Criar: `crates/rusticide-simd/src/token.rs` (tokens e deteccao).
- Modificar: `crates/rusticide-simd/src/lib.rs`, todo ponto de entrada em
  `neon/*.rs`, `crates/rusticide-simd/README.md`, `docs/UNSAFE.md`, a task
  `test` do `mise.toml`, os pontos de chamada de `rusticide-kernels`.

**Interfaces:**

- `#[derive(Clone, Copy, Debug)] pub struct Dotprod(())` em
  `target_arch = "aarch64"`; `Dotprod::detect() -> Option<Dotprod>` usa
  `is_aarch64_feature_detected!("dotprod")` (NEON e baseline ali); sob
  `cfg(miri)` devolve `Some` (os stand-ins nao precisam de hardware).
- `rusticide_simd::neon` compila em todo alvo `aarch64`; cada kernel publico
  recebe `token: Dotprod` primeiro, p. ex.
  `q4_k_group(token: Dotprod, group, inputs, out)`. `AVAILABLE` e removido.
- A task `test` exclui `crates/*/src/neon/*` quando `rustc --print cfg` (com
  `RUSTFLAGS`) nao tem `target_arch="aarch64"`, e `crates/*/src/avx2/*`
  quando nao tem `target_arch="x86_64"` (substitui a checagem de `dotprod`). O
  ramo `None` da deteccao e coberto por
  `Dotprod::from_detected(bool) -> Option<Dotprod>` (privado, chamado por
  `detect`, testado nos dois sentidos).
- Entrada `rusticide-simd` em `docs/UNSAFE.md`: o argumento de solidez passa
  a ser "um token so existe depois que a deteccao provou a feature; todo
  ponto de entrada recebe um".
- [ ] **Passo 1: Testes que falham:** `from_detected_true_yields_a_token`,
  `from_detected_false_yields_none` e `detect_matches_std` (igual a
  `is_aarch64_feature_detected!` fora do Miri).
- [ ] **Passo 2: Executar**, veja-os falhar.
- [ ] **Passo 3: Implementar**; atualize cada comentario `// SAFETY:` para
  citar o token.
- [ ] **Passo 4: Verificar** testes do workspace, lint, Miri e, no M1,
  `mise run test` (100%); tambem `cargo check --target x86_64-apple-darwin
  --workspace` (instale o alvo com rustup).
- [ ] **Passo 5: Commit** `feat(simd): gate NEON on a detected token`.

### Tarefa 3: selecao de adaptador em tempo de execucao

**Arquivos:**

- Modificar: `crates/rusticide-kernels/src/adapter.rs`, `adapter_tests.rs`,
  `neon/mod.rs` e os wrappers `neon/*_dot.rs`, `lib.rs`;
  `crates/rusticide/src/cli/cmd/version.rs`; resultados de `rusticide-bench`;
  `.github/workflows/cicd.yml` (job `arm`); secao de CI do `AGENTS.md`.

**Interfaces:**

- `fn available() -> Vec<&'static Adapter>`: o melhor primeiro, `portable` por
  ultimo; em aarch64 `neon` quando `Dotprod::detect()` e `Some`. Guarda o
  token no `static TOKEN: OnceLock<Dotprod>` do `neon` antes de devolver.
- `fn choose(requested: Option<&str>, available: &[&'static Adapter]) ->
  (&'static Adapter, Option<String>)`: `None` -> `available[0]`; um nome na
  lista -> ele; `"portable"` -> portable; um nome conhecido fora da lista ->
  `available[0]` mais `"RUSTICIDE_KERNELS=<value>: not supported by this
  CPU, using <name>"`; desconhecido -> `available[0]` mais
  `"RUSTICIDE_KERNELS=<value>: unknown adapter, using <name>"`. Nomes
  conhecidos: `portable`, `neon`, `avx2`.
- Os wrappers NEON leem `TOKEN`; quando ele nao esta definido, calculam com o
  kernel portable (Foco da Revisao 4).
- `rusticide version` imprime uma segunda linha `kernels: <name>`.
- Os resultados do bench registram `"adapter": "<name>"`; o gate compara
  somente resultados do mesmo adaptador (uma baseline de outro adaptador e
  "ausente", saida 2).
- Job `arm`: `RUSTFLAGS` perde `-C target-feature=+dotprod`; um passo apos
  `ci:quick` executa `cargo run --release -p rusticide --features cli --
  version` e falha a menos que uma linha seja exatamente `kernels: neon`.
- [ ] **Passo 1: Testes que falham** para cada linha de `choose` (textos
  exatos), `available_ends_with_portable`, `version_names_the_kernels`, o
  round-trip do campo adapter do bench e a saida do gate entre adaptadores, e
  `wrapper_without_token_matches_portable`.
- [ ] **Passo 2: Executar**, veja-os falhar.
- [ ] **Passo 3: Implementar.**
- [ ] **Passo 4: Verificar** testes do workspace, `mise run test`, lint,
  `mise run test:reference` no M1 (e com `RUSTICIDE_KERNELS=portable`),
  `mise run bench:gate` no M1 (sem regressao; regrave a baseline do M1 somente
  se o campo adapter o forcar, como seu proprio commit `perf(bench):
  re-record the M1 baseline`).
- [ ] **Passo 5: Commit** `feat(kernels): select the adapter at run time`.

## Parte 2: o adaptador AVX2

### Tarefa 4: o token `Avx2`, um adaptador `avx2` vazio, a baseline do iMac Pro

**Arquivos:**

- Modificar: `crates/rusticide-simd/src/token.rs`, `lib.rs` (novo modulo
  `avx2`, somente x86_64); criar `crates/rusticide-kernels/src/avx2/mod.rs`;
  `adapter.rs`; `.github/workflows/cicd.yml` (job `miri`: um segundo passo
  para `x86_64-unknown-linux-gnu`); criar
  `crates/rusticide-bench/baselines/<imac machine>.json`.

**Interfaces:**

- `Avx2` espelha `Dotprod` (`detect`, `from_detected`), detectando `avx2`
  (acrescente `fma` somente se um kernel o usar -- veja as Restricoes
  Globais).
- `avx2::ADAPTER` chamado `"avx2"`, `..portable::ADAPTER` (ainda sem
  kernels), classificado primeiro em x86_64 quando detectado;
  `static TOKEN: OnceLock<Avx2>` como no NEON.
- Preparacao do iMac Pro: o worktree `~/rusticide-wt` (Restricoes Globais),
  `mise trust` e `mise install` ali; copie
  `qwen2.5-0.5b-instruct-q4_k_m.gguf` do `~/models` do M1 com `scp`; execute
  com `RUSTICIDE_MODELS=~/models`.
- [ ] **Passo 1: Testes que falham** espelhando os testes de token da Tarefa
  2 e o teste `available` da Tarefa 3 para x86_64 (executados no iMac Pro por
  ssh apos o rsync).
- [ ] **Passo 2: Executar**, veja-os falhar.
- [ ] **Passo 3: Implementar.**
- [ ] **Passo 4: Verificar** no iMac Pro: `cargo test --workspace`,
  `rusticide version` diz `kernels: avx2`, `mise run bench:baseline` (grava a
  baseline do iMac: os kernels portable sob o nome `avx2`, a referencia para
  os pisos de 2x). Meca o llama.cpp ali, somente backend de CPU, threads 10 e
  20, `-r 5`, pp128/pp512/tg64 nos dois modelos; tambem com BLAS. Registre as
  duas linhas de comando e os numeros no ledger (eles vao para os Results na
  Tarefa 11). No M1: testes do workspace e bench:gate.
- [ ] **Passo 5: Commit** `feat(kernels): add the avx2 adapter shell` e,
  separadamente, `perf(bench): record the iMac Pro baseline`.

### Tarefa 5: `Q4_K` em AVX2

**Arquivos:**

- Criar: `crates/rusticide-simd/src/avx2/{q4_k.rs, q4_k/vector.rs,
  q4_k/scalar.rs, q4_k/tests.rs}` e helpers compartilhados conforme preciso;
  `crates/rusticide-kernels/src/avx2/{q4_k.rs, q4_k_dot.rs, tests}`.

**Interfaces:**

- `rusticide_simd::avx2::q4_k::{UNIT, q4_k_group(token: Avx2, group:
  &[[u8; UNIT]], inputs: &[Q8Superblocks], out: &mut [[f32; ROWS]])}`; a
  ordem de bytes da unidade e propria do AVX2 e documentada no modulo.
- `avx2::q4_k_dot::KERNEL: QuantKernel` com `layout: Some(Layout {
  group_rows: ROWS, prepare, read_row, dot_group })`, registrado em
  `avx2::ADAPTER`. Os helpers genericos de layout e dot passam de `neon/` para
  um modulo compartilhado de kernels quando o AVX2 tambem precisar deles
  (sem copias).
- [ ] **Passo 1: Testes que falham:** identidade bit a bit com o portable em
  entradas aleatorias, todas-extremas e de endereco impar, grupos parciais e
  linhas de zero bytes (Foco da Revisao 3, 5), round-trip de `read_row` apos
  `prepare`; o Miri executa os stand-ins escalares.
- [ ] **Passo 2: Executar** no iMac Pro, veja-os falhar.
- [ ] **Passo 3: Implementar.**
- [ ] **Passo 4: Verificar** no iMac Pro: testes, `mise run test` (100%: os
  arquivos `avx2` so sao medidos em x86_64), `test:reference` (TinyLlama e
  Qwen usam `Q4_K`), linhas de kernel de `mise run bench`: `Q4_K` rows8 com
  pelo menos 2x o numero portable da baseline (bloqueante). No M1: testes,
  lint, Miri, bench:gate.
- [ ] **Passo 5: Commit** `feat(simd): run Q4_K on AVX2`.

### Tarefa 6: `Q6_K` em AVX2

Mesmo formato da Tarefa 5 para `Q6_K` (`q6_k_group`, `avx2::q6_k_dot::KERNEL`),
reaproveitando os helpers da Tarefa 5. Piso: `Q6_K` rows8 com pelo menos 2x o
portable. Commit `feat(simd): run Q6_K on AVX2`.

### Tarefa 7: `Q5_0` e `Q8_0` em AVX2

Mesmo formato para ambos, sobre `Q8Blocks`. O `Q8_0` usa o truque do sinal
com `|w|` sem sinal (exato porque as views recusam `-128`, Tarefa 1); seus
testes incluem blocos de pesos todos `-128` contra ativacoes `+-127`. Pisos:
ambos com pelo menos 2x o portable. Commit `feat(simd): run Q5_0 and Q8_0 on
AVX2`.

### Tarefa 8: o gate do AVX2

- [ ] **Passo 1:** No iMac Pro, ocioso, na tomada: `mise run bench` de
  prefill e decode para os dois modelos (threads como o bench escolhe),
  contra os numeros do backend de CPU do llama.cpp da Tarefa 4.
- [ ] **Passo 2:** Se o prefill estiver em pelo menos 65% nos dois modelos e
  todo piso de kernel se mantiver, registre e va para a Tarefa 9. Do
  contrario execute uma rodada de otimizacao (perfile no iMac Pro com `cargo
  flamegraph` ou `sample`, corrija o maior custo, meca de novo); no maximo
  duas rodadas. Ainda abaixo: pare, escreva os numeros e o resumo do perfil
  no ledger e reporte ao dono. Nao inicie a Tarefa 9.
- [ ] **Passo 3:** `test:reference` no iMac Pro com o adaptador `avx2` e com
  `RUSTICIDE_KERNELS=portable`.
- [ ] **Passo 4: Commit** de qualquer otimizacao como `perf(simd): <what>`.

## Parte 3: AVX-512, condicional

### Tarefa 9: o spike de AVX-512

Descartavel, em `.worktrees/avx512-spike` numa branch a partir de
`feat/runtime-adapters`; nunca integrado. Porte o kernel de grupo `Q4_K` para
512 bits (`avx512bw`, `avx512vl`) e meca no iMac Pro: rows8 do kernel e
prefill nos dois modelos contra os numeros AVX2 da Tarefa 8. Decida tambem
como o CI poderia executar testes AVX-512 a cada execucao (um runner que
garanta AVX-512, ou um emulador como o Intel SDE; anote sua licenca e seu
custo de cadeia de suprimentos). Registre no ledger os numeros e um parecer:
construir (ganho de prefill de pelo menos 10% nos dois modelos E uma resposta
de CI; ou o AVX2 falhou um piso e isto o fecha, resposta de CI ainda
obrigatoria) ou registrar como follow-up. Remova o worktree e a branch do
spike apos o registro no ledger.

### Tarefa 10: o adaptador AVX-512 (somente se a Tarefa 9 decidiu "construir")

Um adaptador `avx512` classificado acima do `avx2`, com seu proprio token
(`avx512f`, `avx512bw`, `avx512vl`), kernels para os formatos em que o spike
mostrou ganho (o resto vem do `avx2`), o passo de CI que a Tarefa 9 escolheu,
e os mesmos testes e pisos das Tarefas 5-7. Pulada caso contrario; o ledger
diz isso.

## Encerramento

### Tarefa 11: baselines, resultados, docs

- [ ] **Passo 1:** Regrave as duas baselines (`bench:baseline` no M1 e no
  iMac Pro; ocioso, na tomada), cada uma em seu proprio commit
  `perf(bench): re-record the <machine> baseline`.
- [ ] **Passo 2:** `test:reference` nas duas maquinas com o adaptador padrao e
  com `RUSTICIDE_KERNELS=portable`.
- [ ] **Passo 3:** Preencha os Results do design (EN e pt-BR): linhas de
  kernel portable vs SIMD por formato e maquina, prefill/decode por modelo e
  maquina, numeros do backend de CPU e do BLAS do llama.cpp com suas linhas de
  comando, o parecer do AVX-512, follow-ups. Defina o Status como done.
- [ ] **Passo 4:** `AGENTS.md`: secao de CI (`arm` sem `+dotprod`, a
  assercao da versao, Miri em dois alvos), a linha do `unsafe` se mudou, e o
  ponteiro "Current plan" de volta ao proximo passo da especificacao do
  framework. Mantenha `docs/UNSAFE.md` em sincronia.
- [ ] **Passo 5:** `mise run ci:quick`, `mise run security`,
  `mise run markdownlint` limpos.
- [ ] **Passo 6: Commit** `docs: record the runtime adapters results`.
