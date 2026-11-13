<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Adaptadores de kernels escolhidos em tempo de execucao (Design)

Status: concluido, 2026-11-12. Segue o
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

Medido em 2026-11-12 na arvore final, as duas maquinas na tomada. Os
numeros de kernel sao G/s (pesos, isto e multiplica-somas, por segundo em
uma thread); prefill e decode sao tokens/s com o pool de threads padrao de
cada maquina. A coluna SIMD e o baseline regravado (mediana dos melhores
de tres sessoes); a coluna portavel e uma execucao do mesmo dia com
`RUSTICIDE_KERNELS=portable`.

### Apple M1 Pro (8 nucleos), `neon`

Baseline `apple-m1-pro-8.json`, gravado com a maquina quieta, carga de 1
minuto 1.27. A coluna portavel e uma execucao anterior do mesmo dia, com
carga 2.47, e nao foi repetida.

- Q4_K: row 47.53 vs 29.69 (1.6x), rows8 91.77 vs 30.29 (3.0x).
- Q6_K: row 28.66 vs 26.21 (1.1x), rows8 80.61 vs 27.21 (3.0x).
- Q5_0: row 29.83 vs 20.46 (1.5x), rows8 80.79 vs 24.50 (3.3x).
- Q8_0: row 59.58 vs 35.41 (1.7x), rows8 96.99 vs 35.71 (2.7x).
- Qwen2.5 prefill 993.8 vs 416.4 (2.4x); decode 144.6 vs 124.9.
- TinyLlama prefill 470.7 vs 166.1 (2.8x); decode 105.5 vs 90.7.
- Contra o baseline quieto anterior (gravado em `44ff283`) toda linha fica
  dentro de 1.6%, entao a selecao em tempo de execucao nao custa nada
  mensuravel ao M1, como [Sequencia](#sequencia) pretendia. Uma primeira
  regravacao com carga 2.45 leu as linhas e2e 2-7% abaixo; um A/B na
  mesma sessao contra `44ff283` mostrou o codigo igual, e este registro
  quieto o substituiu.
- O llama.cpp nao foi remedido no M1; vale a referencia do adaptador NEON
  (`llama-bench -ngl 0 -t 6 -p 128,512 -n 0 -r 5`, BLAS e Metal
  carregados): TinyLlama pp128 318.2, Qwen2.5 pp128 599.3 e pp512 722.8.
  Contra o numero mais rapido do llama.cpp para cada modelo (pp128 para
  o TinyLlama, pp512 para o Qwen2.5), o prefill de hoje e 148% e 137%.

### iMac Pro, Xeon W-2150B (10 nucleos / 20 threads), `avx2`

Baseline `intel-r-xeon-r-w-2150b-cpu-3-00ghz-20.json`, gravado com carga
de 1 minuto 1.98; a execucao portavel com 1.73.

- Q4_K: row 39.53 vs 4.31 (9.2x), rows8 63.31 vs 9.41 (6.7x).
- Q6_K: row 16.97 vs 3.54 (4.8x), rows8 43.41 vs 9.57 (4.5x).
- Q5_0: row 21.75 vs 3.36 (6.5x), rows8 41.93 vs 8.20 (5.1x).
- Q8_0: row 27.61 vs 6.95 (4.0x), rows8 45.57 vs 6.99 (6.5x).
- Todo kernel rows8 passa o piso de 2x em [Objetivos](#objetivos).
- Qwen2.5 prefill 561.0 vs 176.0 (3.2x); decode 56.9 vs 32.7 (1.7x).
- TinyLlama prefill 367.7 vs 65.4 (5.6x); decode 43.0 vs 22.3 (1.9x).
- Contra o baseline anterior (gravado em `f97c8c2`, antes dos produtos
  escalares `f32` com FMA e do alinhamento de loops da Tarefa 8) o
  prefill sobe 27% (Qwen2.5) e 64% (TinyLlama), o decode do Qwen2.5 sobe
  10%, e toda linha de kernel fica dentro de 5% exceto duas que subiram:
  Q4_K row (+6%) e Q8_0 rows8 (+9%).
- Esta maquina e ruidosa de ponta a ponta: sessoes seguidas perdem ate
  15% do prefill e do kernel portavel `Q4_0` para o calor, e o llama.cpp
  mostra dois modos de velocidade (abaixo). As tentativas de baseline
  daquele dia deram medianas de prefill de 557-632 (Qwen2.5) e 349-380
  (TinyLlama). O baseline gravado usou os proprios comandos da tarefa
  `bench:baseline` executados diretamente (tres sessoes de
  `rusticide-bench run`, depois `rusticide-bench baseline`); as
  tentativas via `mise run bench:baseline` mediram as mesmas linhas e2e,
  mas uma linha portavel `Q4_0` 10-16% abaixo do baseline anterior,
  enquanto execucoes avulsas do mesmo binario mediram 6.1-6.9 G/s, acima
  dele. Nenhuma causa foi encontrada; `Q4_0` nao tem kernel SIMD e nenhum
  modelo de referencia o usa.

llama.cpp no iMac Pro, build `8642541b2` (llama.cpp 0.8.0, 20 threads,
tamanhos de prompt iguais ao prompt do bench contados pelo
`llama-tokenize`: 161 tokens para o TinyLlama, 142 para o Qwen2.5):

- So o backend de CPU, a referencia do piso: um build so-CPU com AVX2 a
  partir do fonte do Homebrew (`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON
  -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_METAL=OFF -DLLAMA_CURL=OFF`),
  medido com `llama-bench -m <gguf> -t 20 -p <161|142> -n 0 -r 10
  -o json`. No mesmo dia: Qwen2.5 melhor 560.0, media 512.4 (carga
  1.91). A primeira execucao do TinyLlama (carga 1.95) ficou em cerca de
  220 em toda amostra, melhor 228, metade de toda execucao anterior; foi
  descartada. A nova execucao (carga 1.81) deu melhor 465, media 381.8,
  com as amostras divididas entre cerca de 220 e cerca de 450.
- Prefill contra ele, mediana do baseline sobre o melhor do llama.cpp:
  TinyLlama 367.7 / 465 = **79%**, Qwen2.5 561.0 / 560 = **100%**. Melhor
  contra melhor numa mesma sessao (Tarefa 8, depois da ultima mudanca):
  414 / 466 = **89%** e 667 / 560 = **119%**. O piso de 65% vale nos dois
  modelos de qualquer forma.
- Decode contra o mesmo build (`-p 0 -n 64`, Tarefa 8): TinyLlama 43.0
  vs 83.2 (52%), Qwen2.5 57.2 vs 86.0 (67%). Decode nao tem piso.
- O build do Homebrew instalado, so como contexto: seu backend de CPU e
  so SSE3 (nenhuma instrucao `ymm`) e seu backend BLAS (Accelerate) fica
  com os produtos em lote. `llama-bench -m <gguf> -t 20 -p <161|142>
  -n 64 -r 5`: TinyLlama pp161 167.3 e tg64 18.1, Qwen2.5 pp142 310.1 e
  tg64 13.7. Com `-ub 31`, que mantem todo produto no backend de CPU
  SSE3: 24.6 e 24.2.

### Desvio do nao-objetivo f32

A rodada de otimizacao da Tarefa 8, a mudanca que fechou o piso de 65% no
TinyLlama, adicionou corpos AVX2+FMA para os produtos `f32` de dot e mix
da atencao (`avx2::float`); um profile tinha posto cerca de 32% do
prefill do TinyLlama ali. Eles sao identicos bit a bit ao portavel so
para entradas que nao sao NaN (`f6db6d9`; o portavel calcula o mesmo
`mul_add` fundido, via `fmaf` da libm no x86_64). Isso se desvia do
[Nao-objetivo](#nao-objetivos) "todo adaptador mantem o conjunto `f32`
portavel" e estreita o [Objetivo](#objetivos) de identidade bit a bit
"para toda entrada" para entradas que nao sao NaN nesses dois produtos.
Os kernels quantizados mantem a identidade bit a bit completa.

### Paridade com a referencia

`mise run test:reference` passa 2/2 (tokens gulosos do TinyLlama e do
Qwen2.5 contra o llama.cpp) nas duas maquinas com o adaptador padrao
(`neon` no M1, `avx2` no iMac Pro, como `rusticide version` informa) e
com `RUSTICIDE_KERNELS=portable`. Cobertura de 100% no M1
(`mise run ci:quick`, 5276/5276 linhas) e, na Tarefa 8, no iMac Pro
(5304/5304, com os modulos `avx2`).

### AVX-512 (Tarefa 9)

Decidido como trabalho futuro; nenhum adaptador foi construido (Tarefa 10
pulada). Um spike adicionou um adaptador `avx512` acima do `avx2` com um
kernel `Q4_K` de 512 bits (os outros formatos delegavam ao `avx2`). No
W-2150B, melhor de quatro rodadas alternadas, cada uma iniciada com carga
de 1 minuto abaixo de 2.5:

- Kernel `Q4_K`: rows8 86.5 vs 65.5 G/s do `avx2` (1.32x), row 51.2 vs
  41.0 (1.25x).
- Prefill: Qwen2.5 612.5 vs 662.3 (-7.5%, `avx512` mais lento em 3 de 4
  rodadas), TinyLlama 432.6 vs 404.0 (+7%). Decode: 58.3 vs 57.7 e 42.0
  vs 42.8.

A condicao de construcao em [Sequencia](#sequencia) pede ao menos 10% de
prefill nos dois modelos; nenhum modelo chega la, e o AVX2 ja cumpre
todos os pisos, entao a excecao nao se aplica. A causa provavel (nao
provada): trabalho pesado de 512 bits baixa o clock do Skylake-SP para o
trabalho AVX2 e `f32` ao redor, e o Qwen2.5 passa mais do prefill fora do
`Q4_K`. A resposta de CI que o spike propos: um passo do Miri compilado
com `-C target-feature=+avx512f,+avx512bw,+avx512vl` e um `cfg` para o
Miri executar os corpos SIMD reais, mais um job isolado com Intel SDE
(SHA-256 fixado, sem permissoes; licenca de aceite por clique,
redistribuicao incerta); os runners hospedados do GitHub nao
garantem AVX-512, o QEMU nao tem, e um runner self-hosted e inseguro para
um repositorio publico.

**Achado sobre deteccao no macOS.** No iMac Pro, o
`is_x86_feature_detected!` do std informa `avx512f`, `avx512bw` e
`avx512vl` como ausentes, enquanto `sysctl hw.optional.avx512f` (e `bw`,
`vl`) informam 1; essas duas respostas sao o que foi observado. O
mecanismo e o comportamento conhecido do Darwin, nao medido aqui: ele
habilita o estado dos registradores AVX-512 sob demanda, no primeiro uso,
entao o XCR0 nao tem os bits de opmask e ZMM que a deteccao do std exige.
Um adaptador AVX-512 detectado so pelo std nunca seria ativado no macOS.
Com um fallback via `sysctlbyname` no macOS, os testes AVX-512 do spike
rodaram no hardware e passaram (7 + 9, 0 pulados).

### Trabalho futuro

- AVX-512: reabrir so com um fallback de deteccao via `sysctl` no macOS,
  a resposta de CI acima, e uma medicao em Ice Lake ou posterior (sem
  queda forte de clock). O spike fica em
  `.superpowers/sdd/2026-11-02-rusticide-runtime-adapters-plan/t9-out/spike-3aa66d9.patch`
  e `spike-macos-detection.diff` ao lado, so local (ignorado pelo git).
- O binario de release Linux aarch64 ganhar `neon` e provado so pelo job
  `arm` do CI; rodar o workflow da branch uma vez antes do merge (com o
  job `Gates` x86_64, que prova a afirmacao `avx2` nos runners do
  GitHub).
- O ruido de ponta a ponta do iMac Pro (calor entre sessoes, dois modos
  de velocidade) torna seu baseline uma guarda frouxa para o prefill; um
  descanso maior entre sessoes no `bench:baseline` vale a tentativa la.
- O job de MSRV do CI define `RUSTFLAGS` sem a flag de alinhamento de
  loops x86_64, entao a aceitacao de `-align-loops=32` pela 1.98 e
  provada so por builds locais.
- Herdado das revisoes das tarefas: nenhum teste envia uma ativacao
  `-128` pelos caminhos de entrada unica e na pilha; as checagens de
  endereco impar nunca deslocam as ativacoes; `Q5_0` rows8 no NEON fica
  cerca de 1% abaixo da 0.8.0; o ganho do `#[inline(never)]` no chunk do
  `Q4_K` depende do layout do codegen; `samples::format` esta copiado tres
  vezes (so em testes); `avx2/q5_0.rs` e `avx2/q8_0.rs` sao quase copias
  (um bloco `unsafe` por ponto de entrada); nenhum teste pega a perda de
  um `#[inline(always)]` nos corpos AVX2 `f32` (os bits sao os mesmos,
  entao so as linhas de prefill do bench o guardam).
- A diferenca no iMac Pro entre `mise run bench:baseline` e execucoes
  diretas na linha portavel `Q4_0` (acima) segue sem explicacao.
- Release 0.9.0 depois do merge ([Sequencia](#sequencia), passo 4).
- O resto da atencao e do RoPE (`f32`) continua sendo o proximo marco de
  desempenho.
