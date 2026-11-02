<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Adaptadores de kernels escolhidos em tempo de execucao (Design)

Status: rascunho, 2026-11-02. Segue o
[adaptador NEON](2026-11-01-rusticide-neon-adapter-design.pt-BR.md)
(release 0.8.0). Branch `feat/runtime-adapters`.

## Intencao

A release 0.8.0 escolhe seu adaptador de kernels na compilacao. Isso e
certo para `aarch64-apple-darwin`, cujo alvo garante `dotprod`, e errado em
todo o resto:

- Os binarios de release Linux aarch64 compilam sem `+dotprod`, entao
  rodam os kernels portaveis, cerca de 3x mais lentos que NEON, nos
  servidores Graviton e Ampere que todos tem o recurso.
- Nenhum binario x86_64 tem kernels SIMD. O servidor do dono, um iMac Pro
  com Xeon W-2150B (Skylake-SP, 10 nucleos / 20 threads, AVX2 e AVX-512
  F/BW/VL/DQ, sem VNNI), roda os kernels portaveis.

Um binario de release precisa rodar em toda CPU do seu alvo (AGENTS.md:
nunca `target-cpu=native`). Entao o proprio binario tem de descobrir, uma
vez por processo, o que a CPU oferece, e escolher o adaptador mais rapido
que ela consegue rodar.

O dono escolheu (2026-11-02): deteccao em tempo de execucao para todo
adaptador SIMD, com o argumento de solidez do `unsafe` passando de "o alvo
da compilacao garante o recurso" para "a deteccao provou o recurso antes
que o kernel pudesse ser alcancado"; um adaptador AVX2 para x86_64;
AVX-512 somente se merecer seu lugar por medicao (veja
[Sequencia](#sequencia)).

## Sequencia

1. **Selecao em tempo de execucao.** O NEON passa da selecao na compilacao
   para a selecao na execucao. Nada fica mais rapido no M1; o binario de
   release Linux aarch64 ganha NEON. Provado pelo job `arm` do CI largando
   sua flag `+dotprod` e ainda selecionando `neon`.
2. **Adaptador AVX2.** Kernels para `Q4_K`, `Q6_K`, `Q5_0` e `Q8_0`, um
   formato por tarefa, como o NEON fez; medido no iMac Pro.
3. **AVX-512, condicional.** Primeiro um spike. Ele so vira adaptador se
   as duas condicoes valerem:
   - o prefill no W-2150B supera o AVX2 em pelo menos 10% (o dobro da
     tolerancia do gate) nos dois modelos de referencia;
   - o CI consegue rodar seus testes em toda execucao, nao apenas quando o
     runner por acaso tem AVX-512.

   Caso contrario, os numeros do spike vao para [Resultados](#resultados) e
   o AVX-512 vira um acompanhamento registrado. A excecao: se o AVX2 nao
   atingir um piso de [Objetivos](#objetivos) e o AVX-512 for o que fecha a
   diferenca, ele deixa de ser opcional (as duas condicoes de CI continuam
   valendo).
4. **Release 0.9.0** depois do merge, como na 0.8.0.

## Objetivos

- Um binario por alvo seleciona, no primeiro uso de kernel, o melhor
  adaptador que a CPU suporta: `neon` em aarch64 com `dotprod`, `avx2` em
  x86_64 com AVX2 (e FMA somente se um kernel a usar; veja Restricoes),
  senao `portable`.
- `RUSTICIDE_KERNELS=portable|neon|avx2` ainda forca uma escolha. Pedir um
  adaptador que a CPU nao consegue rodar avisa e recai no melhor que ela
  consegue, nunca o executa.
- O adaptador escolhido e observavel: o `rusticide` o informa (quem
  constroi escolhe onde: saida de `--version`, uma linha de `-v`, ou
  `rusticide info`), e os resultados do bench o registram.
- Os kernels AVX2 sao identicos bit a bit aos portaveis, como os do NEON,
  para toda entrada, inclusive valores de byte que um arquivo bem formado
  nunca contem.
- No iMac Pro, cada kernel AVX2 e pelo menos 2x seu kernel portavel
  (rows8, uma thread), e o prefill atinge pelo menos 65% do llama.cpp em
  CPU (`llama-bench`, ja instalado la) nos dois modelos de referencia. O
  piso de 65% e o do milestone NEON; igualar ou superar o llama.cpp e a
  meta, nao o gate. A referencia e apenas o backend de CPU do llama.cpp: a
  build do Homebrew la tambem carrega um backend BLAS, que assume os
  produtos em lote no prefill; esse numero e registrado ao lado, nao
  comparado. Quem constroi fixa como o BLAS e mantido de fora e registra a
  linha de comando do `llama-bench`.
- **Os dois pisos bloqueiam** (dono, 2026-11-02). Uma falha significa mais
  otimizacao dentro do milestone, AVX-512 inclusive se for isso que a
  fecha. Depois de duas rodadas de otimizacao que ainda falhem, o trabalho
  para e os numeros vao ao dono antes de qualquer outra decisao.

## Nao-objetivos

- Kernels `f32` (dot, softmax, RoPE, normas): atencao e RoPE sao o
  Milestone B. Todo adaptador mantem o conjunto `f32` portavel.
- Kernels SIMD `Q4_0` (nenhum modelo de referencia o usa).
- AVX-512 VNNI, AMX, SVE: o hardware do dono nao tem nenhum deles.
- Deteccao em tempo de execucao em qualquer SO alem de Linux, macOS e
  Windows que o std suporta; as macros de deteccao do std decidem isso, nao
  nos.
- Novos alvos em `publish.yml`.

## Anti-objetivos

- Um binario que executa uma instrucao que sua CPU nao tem: SIGILL e uma
  queda para o usuario, e comportamento indefinido nos termos do Rust. Este
  e o unico resultado que faz do milestone um fracasso mesmo que todos os
  numeros sejam bons.
- Resultados diferentes em maquinas diferentes. A identidade bit a bit com
  o portavel e o que mantem a paridade gulosa, a amostragem com semente e
  os testes iguais em todo lugar.
- Custo de deteccao por chamada no laco quente.
- `target-cpu=native` ou uma build por maquina em qualquer configuracao
  commitada.

## Arquitetura

**Tokens de prova (`rusticide-simd`).** Cada conjunto de instrucoes ganha
um tipo `Copy` de tamanho zero, construivel apenas por deteccao:
`Dotprod::detect() -> Option<Dotprod>` (aarch64), `Avx2::detect() ->
Option<Avx2>` (x86_64). Todo ponto de entrada publico de kernel recebe o
token por valor como primeiro argumento. Ter um token e a prova; a unica
chamada `unsafe` ao corpo `#[target_feature]` o cita. A API publica
continua segura, e nenhum chamador seguro alcanca um kernel sem que a
deteccao tenha tido sucesso. A deteccao usa
`is_aarch64_feature_detected!` / `is_x86_feature_detected!` do std, que
guardam a resposta em cache.

Os modulos agora compilam por arquitetura (`neon` em todo `aarch64`,
`avx2` em todo `x86_64`), nao mais por recurso do alvo.

**Selecao (`rusticide-kernels`).** `compiled_default()` vira uma etapa de
deteccao que produz a lista ordenada de adaptadores que esta CPU consegue
rodar (melhor primeiro, `portable` por ultimo). `choose(requested,
available)` mantem sua forma pura e injetavel: os testes passam qualquer
lista, entao todo ramo e alcancavel em qualquer maquina. `active()` ainda
resolve uma vez pelo `OnceLock`.

**Como o adaptador alcanca o token.** As funcoes `QuantKernel` de cada
adaptador SIMD sao ponteiros `fn` simples, entao nao podem capturar um
token. Quem constroi escolhe como elas o obtem, dentro destas regras:
nenhuma deteccao por chamada de kernel alem de uma leitura em cache, e
nenhum caminho pelo qual um kernel rode antes de a selecao do `Adapter`
provar o recurso. Uma forma aceitavel: o token fica guardado em um
`OnceLock` no modulo do adaptador, definido pela etapa de deteccao, e os
wrappers o leem (um token ausente e um caminho de bug que devolve o
resultado portavel, nunca uma chamada `unsafe`).

**Kernels AVX2.** Mesma estrutura do NEON: um layout de grupos de linhas
preparado uma vez no carregamento, um kernel de grupo por formato, caminhos
de decodificacao de um token onde o milestone NEON achou que valiam a pena.
O layout e proprio do AVX2 (o do NEON e moldado para as lanes do `sdot`);
`Layout`/`prepare_matrix` ja permitem isso.

## Achados que moldam o design

- **`vpmaddubsw` satura.** Ele multiplica bytes sem sinal por bytes com
  sinal e soma pares em um `i16` saturante. Para `Q4_K` (0..15), `Q5_0` e
  `Q6_K` (codificado com offset, no maximo 63) contra ativacoes `Q8`, a
  soma do par fica dentro de `i16`. Para `Q8_0`, o truque usual de sinal
  (`|w|` sem sinal, `a` carregando o sinal de `w`) e exato enquanto a
  ativacao nunca for `-128`: `2 x 128 x 127 = 32512`. Os pesos vem do
  arquivo e podem conter `-128`, o que esta ordem tolera; as ativacoes vem
  do nosso quantizador, que mapeia a maior magnitude para `+-127`. Minha
  decisao (2026-11-02): o invariante vale na API publica do
  `rusticide-simd`, o crate com o `unsafe`, nao apenas no
  `rusticide-kernels`. Suas visoes de ativacao (`Q8Blocks::new` e sua
  contraparte AVX2) recusam quants que contenham `-128`, uma varredura por
  matriz de ativacoes, de modo que todo valor que a API segura aceita tem
  resultado exato e o kernel `Q8_0` nao precisa de ramo por bloco. A
  identidade bit a bit e exigida para toda entrada aceita. Os testes de
  cada kernel incluem blocos de pesos todos `-128` e ativacoes extremas.
- **AVX-512 no Skylake-SP** nao tem VNNI: o dot int8 e o mesmo par
  `maddubs` + `madd` com o dobro da largura, e trabalho pesado de 512 bits
  reduz o clock de todos os nucleos. Seu ganho sobre o AVX2 e desconhecido,
  e por isso e condicional.
- **Miri.** O padrao do NEON vale: substitutos escalares com `cfg(miri)` e
  os mesmos nomes, com o job do Miri interpretando o crate para cada
  arquitetura (adicionar um alvo x86_64). Sob o Miri, a deteccao deve
  produzir um token (os substitutos nao precisam de hardware); quem
  constroi decide como.
- **Cobertura.** Cada modulo agora compila em todo runner da sua
  arquitetura, entao a exclusao da tarefa `test` passa de "sem `dotprod`"
  para "nao esta arquitetura": `neon/` fora de aarch64, `avx2/` fora de
  x86_64. Todos os runners x86_64 do GitHub tem AVX2, entao o job `Gates`
  x86 mede os kernels AVX2 em 100%. O ramo "ausente" da deteccao e
  alcancado pelo `choose` injetavel.
- **O iMac Pro** (`ssh elio@192.168.18.146`, sudo sem senha) tem rustup,
  mise e `llama-bench`, e apenas o modelo TinyLlama; o Qwen2.5 precisa ser
  copiado para la. Seu baseline e um novo arquivo
  `crates/rusticide-bench/baselines/<machine>.json` gerado por
  `mise run bench:baseline` naquela maquina.

## O gate

Os pisos em [Objetivos](#objetivos) bloqueiam, como dito la.

Por tarefa, como antes: testes e 100% de cobertura, Miri, `bench:gate` no
M1 (sem regressao pela mudanca de selecao), e no iMac Pro para as tarefas
AVX2 assim que seu baseline existir. A tarefa final regrava os dois
baselines, roda `test:reference` nas duas maquinas (NEON no M1, AVX2 no
iMac Pro, e `RUSTICIDE_KERNELS=portable` em cada uma), e registra as
razoes contra o llama.cpp.

## Restricoes

- MSRV 1.98, pin do toolchain inalterado. Os intrinsecos AVX2 e AVX-512 e
  `#[target_feature]` em funcoes seguras sao estaveis bem antes da 1.98.
- O `unsafe` fica dentro do `rusticide-simd`, um bloco por ponto de entrada
  de kernel, cada um com um comentario `// SAFETY:` nomeando o token;
  `docs/UNSAFE.md` atualiza sua entrada do `rusticide-simd` para o
  argumento de tempo de execucao.
- FMA muda o arredondamento. Um kernel que deve continuar identico bit a
  bit ao portavel nao pode usa-la onde o portavel faz multiplicacao e soma
  separadas.
- As regras de sempre: ASCII, arquivos com menos de 300 linhas, cabecalhos
  SPDX, nada de `unwrap`/`expect` em codigo de producao, docs bilingues,
  commits convencionais assinados.

## Deixado para quem constroi

- Onde o nome do adaptador e informado.
- Como os wrappers do adaptador alcancam seu token (dentro das regras
  acima).
- O layout AVX2, a largura do grupo e o fatiamento de tokens por formato.
- Como a deteccao se comporta sob o Miri.
- Se o `Gates arm64` mantem algum `RUSTFLAGS` (no minimo deve largar
  `+dotprod` e afirmar que o adaptador selecionado e `neon`).
- Como o spike de AVX-512 testa no CI, se chegar a esse ponto.

## Resultados

(Preenchido pela tarefa final.)
